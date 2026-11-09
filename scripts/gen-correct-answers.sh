#!/bin/bash
#SBATCH --job-name=drawedu-answers
#SBATCH --time=2:00:00                     # full run ~1h; a crashed run continues with RESUME=1
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G                # 96G: images are preprocessed in RAM
#SBATCH --output=gen-correct-answers-%j.log
#
# Generates correct answers for every DrawEduMath image with Qwen3-VL-30B-A3B-Instruct.
# Submit from SP/:
#   sbatch scripts/gen-correct-answers.sh                          # all images
#   sbatch --export=ALL,LIMIT=20 scripts/gen-correct-answers.sh   # smoke test on the first 20 images
#   sbatch --export=ALL,RESUME=1 scripts/gen-correct-answers.sh   # continue a crashed run
# Writes (next to the dataset):
#   correct_answers/Qwen3-VL-30B-A3B-Instruct.raw.jsonl  one line per image, the model's raw output and parse
#   correct_answers/Qwen3-VL-30B-A3B-Instruct.json       {"_meta", "answers": {image_name: {...}}}, read by
#                                                        SP/feedback_pipeline.py --correct_answers_json
set -euo pipefail

export HF_HOME=/cluster/scratch/nbagan/hf_cache
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
DATASET_DIR="${HF_HOME}/hub/datasets--Heffernan-WPI-Lab--DrawEduMath"
OUT_DIR="${DATASET_DIR}/correct_answers"
mkdir -p "${OUT_DIR}"
PY="${TMPDIR:-/tmp}/gen_correct_answers_${SLURM_JOB_ID}.py"
LIMIT="${LIMIT:-0}"    # 0 = all images

cat > "${PY}" <<'PYEOF'
import base64, csv, glob, io, json, os, re, sys, time
from collections import Counter, defaultdict
from PIL import Image

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DATASET_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
LIMIT = int(sys.argv[3])
MAX_PIXELS = 1638400          # same downscale as feedback_pipeline.py
MAX_TOKENS = 2048
TP = int(os.environ.get("TP", "4"))
CHUNK = int(os.environ.get("CHUNK", "128"))   # images queued in vLLM at a time
RESUME = os.environ.get("RESUME", "0") == "1"  # 1 = keep the images already in the .raw.jsonl
CONSENSUS_MIN = 0.5          # share of a problem's answered images that must agree for a consensus answer
ANSWER_TYPES = ("exact", "open_ended", "drawing", "not_answerable")

SYSTEM = """You are an expert math teacher writing an answer key.
You get a screenshot of an online math problem. The left side ("Problem") shows the problem; the right side \
("Student Response") shows a student's handwritten work. IGNORE the student's work when solving: it may be wrong. \
Use it only to see which part of the problem the student is answering.
If the problem has several parts, the part the student response is for is usually boxed in red. Solve that part \
(earlier parts may give context). If nothing is boxed, solve the part the student response addresses.
Solve the problem yourself, step by step, then give the correct answer."""

USER = """Classify the answer type and give the correct answer:
- "exact": the part has one correct answer that can be written as text (a number, fraction, expression, equation, \
coordinate, comparison, yes/no, or short word answer). correct_answer = that answer, as short as possible \
(e.g. "3/8", "x = 4", "120,000", "1/4 > 1/8"). If the task is "draw/plot/shade X" but X has a single exact value \
(e.g. "plot 3/4 on the number line"), it is still "exact" with that value.
- "open_ended": several different answers are correct (e.g. "write a fraction greater than 1/2", "estimate"). \
correct_answer = one canonical correct answer followed by the rule all correct answers must satisfy, \
e.g. "3/4 (any fraction greater than 1/2)".
- "drawing": the task is to draw, model or label a diagram. correct_answer = what a correct drawing must show, in \
one sentence. If the drawing ends in or represents a value, START with that value, then the description in \
parentheses, e.g. "2 (number line: start at 0, move to -6, then to 6, then to 2)".
- "not_answerable": the problem text is unreadable, cut off, or missing. correct_answer = null.
Return JSON only, in this format:
{
"problem_text": "<the text of the part being answered, transcribed>",
"solution": "<your step-by-step solution>",
"answer_type": "<exact|open_ended|drawing|not_answerable>",
"correct_answer": "<the answer, or null>"
}"""


def data_url(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(raw):
    cands = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.S)]
    a, b = raw.find("{"), raw.rfind("}")
    if 0 <= a < b:
        cands.append(raw[a:b + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            try:  # models often leave LaTeX backslashes unescaped
                obj = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", c))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj
    return None


def norm(ans):
    """Key for comparing answers across images of the same problem: the part before any '(' explanation."""
    s = ans.split("(")[0] if not ans.lstrip().startswith("(") else ans
    s = s.lower().strip().rstrip(".")
    s = re.sub(r"\\\(|\\\)|\$|\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s.replace(",", "") if re.fullmatch(r"[\d,.\-]+", s) else s


def per_image_record(r, raw, finish_reason):
    obj = parse(raw) or {}
    atype = str(obj.get("answer_type") or "").strip().lower()
    ans = obj.get("correct_answer")
    ans = None if ans is None or str(ans).strip().lower() in ("", "null", "none") else str(ans).strip()
    status = "ok" if obj and atype in ANSWER_TYPES else "parse_error"
    if atype == "not_answerable":
        ans = None
    return {"image_name": r["Image Name"], "problem_id": r["Problem ID"], "status": status,
            "answer_type": atype if status == "ok" else None, "image_answer": ans,
            "problem_text": obj.get("problem_text"), "solution": obj.get("solution"),
            "finish_reason": finish_reason, "raw": raw}


def main():
    csv.field_size_limit(sys.maxsize)
    csv_path = glob.glob(f"{DATASET_DIR}/snapshots/*/Data/DrawEduMath_QA.csv")[0]
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if LIMIT:
        rows = rows[:LIMIT]
    images_dir = f"{DATASET_DIR}/images"
    print(f"{len(rows)} images", flush=True)

    base = MODEL.split("/")[-1]
    raw_path = f"{OUT_DIR}/{base}.raw.jsonl"
    done = {}
    if RESUME and os.path.exists(raw_path):
        for line in open(raw_path):
            d = json.loads(line)
            done[d["image_name"]] = d
        print(f"resuming: {len(done)} images already done", flush=True)
    elif os.path.exists(raw_path):
        os.remove(raw_path)
    todo = [r for r in rows if r["Image Name"] not in done]

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, tensor_parallel_size=TP, max_model_len=8192, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={"image": 1, "video": 0}, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    t0 = time.time()
    # Chunks: vLLM keeps every queued image's preprocessed tensors in RAM, so queueing all 2030 at once runs out.
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(f"{images_dir}/{r['Image Name']}")}},
                                               {"type": "text", "text": USER}]}] for r in chunk]
        outs = llm.chat(convs, params, use_tqdm=False)
        with open(raw_path, "a") as f:
            for r, o in zip(chunk, outs):
                d = per_image_record(r, o.outputs[0].text, o.outputs[0].finish_reason)
                done[r["Image Name"]] = d
                f.write(json.dumps(d) + "\n")
        print(f"{start + len(chunk)}/{len(todo)} images, {time.time() - t0:.0f}s", flush=True)
    print(f"generated in {time.time() - t0:.0f}s", flush=True)
    per_image = [done[r["Image Name"]] for r in rows]

    # Consensus per problem: the same problem appears in several images, so a majority answer is more reliable.
    by_problem = defaultdict(list)
    for d in per_image:
        by_problem[d["problem_id"]].append(d)
    final = {}
    for pid, ds in by_problem.items():
        answered = [d for d in ds if d["image_answer"]]
        counts = Counter(norm(d["image_answer"]) for d in answered)
        top, top_n = counts.most_common(1)[0] if counts else (None, 0)
        agreement = top_n / len(answered) if answered else 0.0
        type_counts = Counter(d["answer_type"] for d in ds if d["answer_type"])
        consensus = None
        if top and agreement >= CONSENSUS_MIN and len(answered) > 1:
            consensus = next(d["image_answer"] for d in answered if norm(d["image_answer"]) == top)
        for d in ds:
            final[d["image_name"]] = {
                "problem_id": pid,
                # What [CORRECT_ANSWER] gets: the problem's consensus answer, else this image's own answer.
                "correct_answer": consensus or d["image_answer"],
                "answer_type": d["answer_type"],
                "source": "problem_consensus" if consensus else ("image" if d["image_answer"] else None),
                "image_answer": d["image_answer"],
                "problem_agreement": round(agreement, 3),
                "problem_n_images": len(ds),
                "problem_answer_types": dict(type_counts),
                "problem_text": d["problem_text"],
                "status": d["status"],
            }
    meta = {"model": MODEL, "created": time.strftime("%Y-%m-%d %H:%M"), "csv": csv_path, "n_images": len(rows),
            "max_pixels": MAX_PIXELS, "temperature": 0.0, "consensus_min": CONSENSUS_MIN,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "script": "SP/scripts/gen-correct-answers.sh"}
    with open(f"{OUT_DIR}/{base}.json", "w") as f:
        json.dump({"_meta": meta, "answers": final}, f, indent=1)

    st = Counter(d["status"] for d in per_image)
    ty = Counter(d["answer_type"] for d in per_image)
    src = Counter(v["source"] for v in final.values())
    trunc = sum(d["finish_reason"] == "length" for d in per_image)
    print(f"status {dict(st)} | types {dict(ty)} | source {dict(src)} | truncated {trunc}", flush=True)
    print(f"wrote {OUT_DIR}/{base}.json", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # do not wait for vLLM worker processes
PYEOF

source "${HOME}/venvs/drawedumath/bin/activate"   # see SP/requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u "${PY}" "${DATASET_DIR}" "${OUT_DIR}" "${LIMIT}"
