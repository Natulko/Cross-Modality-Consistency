#!/usr/bin/env python3
"""Tutor -> student -> judge pipeline on DrawEduMath student work images.

For each selected DrawEduMath instance (one image of a problem with a student's handwritten answer that the teacher
annotations mark as containing an error):
  1. tutor:   the model gets the tutor prompt with the image and the teacher-written questions (as the misconception
              description) and writes short feedback, returned as JSON {"feedback": ...}.
  2. student: the model gets the student prompt with the same image and the tutor's feedback and answers as the student.
  3. judge:   the model gets the judgment prompt with the image, the feedback and the student's response and classifies
              the response into one of six categories, returned as JSON {"category", "final_answer", "reasoning"}.
Every stage runs over all chains in vLLM batches of --chunk_size; after each batch generations.jsonl is rewritten and
progress plus one example output are logged. The same model plays all three roles and is loaded once.

With --tutor_output image the tutor is an image-edit model (e.g. Qwen/Qwen-Image-Edit-2511) served by vLLM-Omni from
its own venv: the job starts that server on all GPUs, sends each image with the filled tutor prompt as the edit
instruction (system + user text, without the trailing 'Return: {...}' block), saves the edited images under
feedback_images/, and stops the server before --model is loaded for student and judge. Prompts then refer to the
edited image with [FEEDBACK_IMAGE] (so a prompt can hold two images: the original and the feedback image).

DrawEduMath has the problem and the student's answer only as one image (Images_w_Problem_Bodies), so every
problem/answer placeholder in a prompt (see IMAGE_PLACEHOLDERS) is filled with that image: the first one in a prompt
gets the image, later ones a pointer to it. DrawEduMath has no correct answers; [CORRECT_ANSWER] gets the image's
answer from --correct_answers_json (model-generated answer key), or --correct_answer for images without one.

Must run inside a Slurm job; see scripts/run-feedback-pipeline.sh.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("feedback_pipeline")

ROLES = ("tutor", "student", "judge")
JUDGE_CATEGORIES = ("correct_flip", "sycophantic_flip", "different_wrong", "constructive_pushback", "passive_maintain",
                    "confusion")
# Placeholders for the problem and the student's answer. In DrawEduMath both exist only as the image.
IMAGE_PLACEHOLDERS = {
    "PROBLEM_TEXT_AND_STUDENT_ANSWER_IMAGE", "PROBLEM_TEXT_AND_STUDENT_WRONG_ANSWER_IMAGE",
    "PROBLEM_TEXT_AND_STUDENT_ANSWER",
    "STUDENT_ANSWER_IMAGE", "STUDENT_WRONG_ANSWER_IMAGE",
    "PROBLEM_TEXT", "STUDENT_ANSWER", "STUDENT_WRONG_ANSWER_TEXT",
}
IMAGE_REPEAT_TEXT = "(shown in the image above)"
IMAGE_MARKER = "<image>"  # stands for the image in saved prompt texts
FEEDBACK_IMAGE_PLACEHOLDER = "FEEDBACK_IMAGE"  # the tutor's edited image (--tutor_output image)
FEEDBACK_IMAGE_MARKER = "<feedback_image>"
MARKERS = re.compile(f"({re.escape(IMAGE_MARKER)}|{re.escape(FEEDBACK_IMAGE_MARKER)})")
MAX_IMAGES_PER_PROMPT = 2  # original image + feedback image
EDIT_IMAGE_TEXT = "(the input image)"  # replaces the image placeholder in the image-edit instruction
RETURN_BLOCK = re.compile(r"\n\s*Return:\s*\{.*\}\s*$", re.S)  # trailing JSON answer format, dropped for image edits
TUTOR_OUTPUTS = ("text", "image")
PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
ROLE_HEADER = re.compile(r"^###[ \t]*(System|User)[ \t]*$", re.M | re.I)
# Teacher question whose answer says whether the student made an error (asked for all 2030 images).
ERROR_QUESTION_PREFIX = "what errors does the student make"
# Teacher answers that report no error, or that the image could not be read.
NO_ERROR = re.compile(r"\bno (errors?|mistakes?)\b|\b(did|does|do)(n't| not) mak?e? (an |any )?(errors?|mistakes?)\b|"
                      r"\bmakes? no (errors?|mistakes?)\b|\bcan'?t read\b|\bcannot read\b|\bunable to read\b", re.I)
MISCONCEPTION_SOURCES = ("questions", "qa")

FIG_DPI = 150


@dataclass
class Instance:
    """One DrawEduMath image with its teacher annotations."""
    instance_idx: int
    problem_id: str
    image_name: str
    image_path: str
    misconception: str
    teacher_error: str
    teacher_qa: List[Dict[str, str]]
    correct_answer: str


@dataclass
class Record:
    """One tutor -> student -> judge chain for one instance, with every prompt and output."""
    instance_idx: int
    sample_idx: int
    problem_id: str
    image_name: str
    misconception: str
    teacher_error: str
    correct_answer: str
    prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)  # role -> {"system", "user"}
    tutor_raw: str = ""
    tutor_status: str = ""
    feedback: Optional[str] = None
    feedback_image: Optional[str] = None  # path of the tutor's edited image (--tutor_output image)
    student_raw: str = ""
    judge_raw: str = ""
    judge_status: str = ""
    category: Optional[str] = None
    final_answer: Optional[str] = None
    judge_reasoning: Optional[str] = None


def n_instances_arg(value: str) -> Optional[int]:
    """--n_instances: a positive count, or 'all' (None)."""
    if value.lower() == "all":
        return None
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("n_instances must be >= 1 or 'all'")
    return n


def parse_args() -> argparse.Namespace:
    """Parse the CLI. Every setting is required so that the launcher script is the single source of config."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF model id served with vLLM; plays student and judge, and the tutor if --tutor_output text")
    p.add_argument("--tutor_output", required=True, choices=TUTOR_OUTPUTS,
                   help="text: --model writes text feedback; image: --tutor_model edits the image (vLLM-Omni)")
    p.add_argument("--tutor_model", required=True,
                   help="HF id of the tutor model; must equal --model for --tutor_output text")
    p.add_argument("--image_edit_venv", required=True, type=Path,
                   help="venv with vllm-omni, used to serve --tutor_model for --tutor_output image")
    p.add_argument("--image_edit_offload_config", required=True,
                   help="JSON for vllm-omni's --diffusion-offload-config (CPU offload), or 'none'")
    p.add_argument("--image_edit_steps", required=True, type=int, help="Diffusion steps per image edit")
    p.add_argument("--image_edit_true_cfg_scale", required=True, type=float, help="true_cfg_scale for image edits")
    p.add_argument("--image_edit_negative_prompt", required=True, help="Negative prompt for image edits")
    p.add_argument("--image_edit_server_timeout", required=True, type=int,
                   help="Give up starting the image-edit server when its log has not changed for this many seconds "
                        "(loading itself can take longer)")
    p.add_argument("--prompts_yaml", required=True, type=Path)
    p.add_argument("--tutor_prompt", required=True, help="Dotted key into prompts.yaml or a template file path")
    p.add_argument("--student_prompt", required=True, help="Dotted key into prompts.yaml or a template file path")
    p.add_argument("--judge_prompt", required=True, help="Dotted key into prompts.yaml or a template file path")
    p.add_argument("--dataset_csv", required=True, type=Path, help="DrawEduMath_QA.csv")
    p.add_argument("--images_dir", required=True, type=Path, help="Folder with the DrawEduMath images (by Image Name)")
    p.add_argument("--n_instances", required=True, type=n_instances_arg,
                   help="Images drawn (seeded, without replacement) from those the teachers mark as erroneous, "
                        "or 'all' for every one of them")
    p.add_argument("--n_samples", required=True, type=int, help="Tutor -> student -> judge chains per instance")
    p.add_argument("--misconception_source", required=True, choices=MISCONCEPTION_SOURCES,
                   help="[MISCONCEPTION_DESCRIPTION]: questions = the teacher-written questions; qa = the questions "
                        "with the teachers' answers")
    p.add_argument("--correct_answers_json", required=True,
                   help="Answer key {\"answers\": {image_name: {\"correct_answer\": ...}}} for [CORRECT_ANSWER], or "
                        "'none'")
    p.add_argument("--correct_answer", required=True,
                   help="Text for [CORRECT_ANSWER] when the answer key has no answer for an image")
    p.add_argument("--max_image_pixels", required=True, type=int,
                   help="Images with more pixels are downscaled (keeping the aspect ratio) before being sent")
    p.add_argument("--tutor_temperature", required=True, type=float)
    p.add_argument("--student_temperature", required=True, type=float)
    p.add_argument("--judge_temperature", required=True, type=float)
    p.add_argument("--top_p", required=True, type=float)
    p.add_argument("--max_tokens", required=True, type=int, help="Max new tokens per response, for every role")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max_model_len", required=True, type=int, help="Must fit system + user prompt + image + max_tokens")
    p.add_argument("--gpu_memory_utilization", required=True, type=float)
    # Not required, unlike the rest: job 19903876 was submitted before this option existed.
    p.add_argument("--mm_processor_cache_gb", type=float, default=1.0,
                   help="CPU RAM for vLLM's cache of processed images, held twice (front end and engine); "
                        "vLLM's default of 4 made the job exceed 16 GB (default here: 1)")
    # Not required either: job 19953790 was submitted before this option existed.
    p.add_argument("--local_model_dir", default="",
                   help="Node-local folder (the job's $TMPDIR, needs #SBATCH --tmp) to copy the image-edit model to "
                        "before serving it; reading it from scratch takes ~30 min. Empty: serve from scratch")
    p.add_argument("--tensor_parallel_size", required=True, type=int, help="GPUs to shard the model across")
    p.add_argument("--chunk_size", required=True, type=int,
                   help="Chains per vLLM batch; generations.jsonl is rewritten and progress logged after each batch")
    p.add_argument("--log_chars", required=True, type=int, help="Max characters of the example output logged per batch")
    p.add_argument("--output_dir", required=True, type=Path)
    return p.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Log to stderr (the Slurm log) and to <output_dir>/run.log."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "run.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---- Data ------------------------------------------------------------------------------------------------------

def load_qa(cell: str) -> List[Dict[str, str]]:
    """Parse a QA column of DrawEduMath_QA.csv (JSON, sometimes a JSON string holding a Python literal)."""
    qa = json.loads(cell.strip())
    if isinstance(qa, str):
        qa = ast.literal_eval(qa)
    return qa


def teacher_error(qa: List[Dict[str, str]]) -> Optional[str]:
    """The teacher's answer to 'What errors does the student make ...', or None if the question is missing."""
    for d in qa:
        if d["question"].strip().lower().startswith(ERROR_QUESTION_PREFIX):
            return d["answer"].strip()
    return None


def misconception_text(qa: List[Dict[str, str]], source: str) -> str:
    """[MISCONCEPTION_DESCRIPTION]: the teacher questions as a list, optionally with the teachers' answers."""
    if source == "questions":
        return "\n" + "\n".join(f"- {d['question'].strip()}" for d in qa)
    return "\n" + "\n".join(f"- Q: {d['question'].strip()}\n  A: {d['answer'].strip()}" for d in qa)


def load_answer_key(spec: str) -> Dict[str, str]:
    """{image_name: correct answer} from --correct_answers_json; images with a null answer are left out."""
    if spec.lower() == "none":
        return {}
    answers = json.loads(Path(spec).read_text())["answers"]
    return {name: a["correct_answer"] for name, a in answers.items() if a.get("correct_answer")}


def load_instances(dataset_csv: Path, images_dir: Path, n_instances: Optional[int], misconception_source: str,
                   seed: int, answer_key: Dict[str, str],
                   default_answer: str) -> Tuple[List[Instance], Dict[str, int]]:
    """Draw n_instances (None: all) images the teachers mark as erroneous, seeded, and return them with selection
    counts."""
    csv.field_size_limit(sys.maxsize)
    with open(dataset_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = []
    for row in rows:
        qa = load_qa(row["QA Teacher"])
        error = teacher_error(qa)
        if error and not NO_ERROR.search(error):
            candidates.append((row, qa, error))
    if n_instances is None:
        n_instances = len(candidates)
    if n_instances > len(candidates):
        raise ValueError(f"n_instances={n_instances} > {len(candidates)} images with a teacher-reported error")
    chosen = random.Random(seed).sample(candidates, n_instances)
    instances = []
    for i, (row, qa, error) in enumerate(chosen):
        path = images_dir / row["Image Name"]
        if not path.is_file():
            raise FileNotFoundError(f"Image {path} is missing")
        instances.append(Instance(i, row["Problem ID"], row["Image Name"], str(path),
                                  misconception_text(qa, misconception_source), error, qa,
                                  answer_key.get(row["Image Name"], default_answer)))
    return instances, {"n_images": len(rows), "n_with_error": len(candidates)}


def image_png(path: str, max_pixels: int) -> Tuple[bytes, List[int]]:
    """The image as PNG bytes, downscaled to at most max_pixels; also returns the size sent."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), list(img.size)


def image_data_url(path: str, max_pixels: int) -> Tuple[str, List[int]]:
    """The image as a PNG data URL, downscaled to at most max_pixels; also returns the size sent."""
    png, size = image_png(path, max_pixels)
    return "data:image/png;base64," + base64.b64encode(png).decode(), size


# ---- Prompts ---------------------------------------------------------------------------------------------------

def load_template(spec: str, prompts_yaml: Path) -> Tuple[str, str]:
    """Resolve a prompt spec to (name, template): a template file, or a dotted key into prompts.yaml."""
    path = Path(spec)
    if path.is_file():
        return path.stem, path.read_text()
    node: Any = yaml.safe_load(prompts_yaml.read_text())
    for key in spec.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Prompt '{spec}' is neither a file nor a key in {prompts_yaml}")
        node = node[key]
    if not isinstance(node, str):
        raise ValueError(f"Prompt key '{spec}' in {prompts_yaml} is a section, not a template")
    return spec, node


def split_roles(template: str) -> Tuple[str, str]:
    """Split a template into (system, user) at its '### System' / '### User' headers (no headers: all user)."""
    parts = ROLE_HEADER.split(template)
    if len(parts) == 1:
        return "", template.strip()
    sections = {name.lower(): text.strip() for name, text in zip(parts[1::2], parts[2::2])}
    return sections.get("system", ""), sections.get("user", "")


def fill(text: str, values: Dict[str, str]) -> Tuple[str, bool]:
    """Fill placeholders. The first image placeholder becomes IMAGE_MARKER, later ones IMAGE_REPEAT_TEXT;
    [FEEDBACK_IMAGE] becomes FEEDBACK_IMAGE_MARKER (at most once).

    Returns (text, has_image). Unknown placeholders raise, so a renamed placeholder never reaches the model.
    """
    names = [m.group(1) for m in PLACEHOLDER.finditer(text)]
    unknown = sorted(set(names) - set(values) - IMAGE_PLACEHOLDERS - {FEEDBACK_IMAGE_PLACEHOLDER})
    if unknown:
        raise KeyError(f"No value for placeholders {unknown}")
    if names.count(FEEDBACK_IMAGE_PLACEHOLDER) > 1:
        raise ValueError(f"[{FEEDBACK_IMAGE_PLACEHOLDER}] may appear only once per prompt")
    seen_image = False

    def sub(m: re.Match) -> str:
        nonlocal seen_image
        name = m.group(1)
        if name == FEEDBACK_IMAGE_PLACEHOLDER:
            return FEEDBACK_IMAGE_MARKER
        if name in IMAGE_PLACEHOLDERS:
            first, seen_image = not seen_image, True
            return IMAGE_MARKER if first else IMAGE_REPEAT_TEXT
        return values[name]

    filled = PLACEHOLDER.sub(sub, text)
    return filled, seen_image or FEEDBACK_IMAGE_MARKER in filled


def build_prompt(template: str, values: Dict[str, str]) -> Dict[str, str]:
    """Fill a template into {"system", "user"} texts; the user text holds at most one IMAGE_MARKER and one
    FEEDBACK_IMAGE_MARKER."""
    system, user = split_roles(template)
    system, system_image = fill(system, values)
    if system_image:
        raise ValueError("Image placeholders are only supported in the ### User part")
    user, _ = fill(user, values)
    return {"system": system, "user": user}


def edit_instruction(prompt: Dict[str, str]) -> str:
    """The image-edit model's single text input: system + user text, without the trailing 'Return: {...}' block and
    with the image placeholder as EDIT_IMAGE_TEXT (the image itself is sent alongside)."""
    if FEEDBACK_IMAGE_MARKER in prompt["user"]:
        raise ValueError(f"The tutor prompt cannot use [{FEEDBACK_IMAGE_PLACEHOLDER}]")
    user = RETURN_BLOCK.sub("", prompt["user"]).replace(IMAGE_MARKER, EDIT_IMAGE_TEXT)
    return "\n\n".join(part for part in (prompt["system"], user) if part)


def user_parts(user: str) -> List[str]:
    """The user text split into text pieces and image markers, in order."""
    return [piece for piece in MARKERS.split(user) if piece in (IMAGE_MARKER, FEEDBACK_IMAGE_MARKER) or piece.strip()]


def to_messages(prompt: Dict[str, str], image_urls: Dict[str, str]) -> List[Dict[str, Any]]:
    """OpenAI-style chat messages for vLLM, with each image at its marker's position in the user text.
    image_urls maps IMAGE_MARKER / FEEDBACK_IMAGE_MARKER to data URLs."""
    content: List[Dict[str, Any]] = []
    for piece in user_parts(prompt["user"]):
        if piece in (IMAGE_MARKER, FEEDBACK_IMAGE_MARKER):
            if piece not in image_urls:
                raise ValueError(f"Prompt uses {piece} but there is no such image (feedback image needs "
                                 f"--tutor_output image)")
            content.append({"type": "image_url", "image_url": {"url": image_urls[piece]}})
        else:
            content.append({"type": "text", "text": piece})
    messages = [{"role": "system", "content": prompt["system"]}] if prompt["system"] else []
    return messages + [{"role": "user", "content": content}]


def render_chat(llm: "LLM", prompt: Dict[str, str]) -> str:
    """The text the model sees, through its chat template; images show as the template's vision tokens."""
    user = [{"type": "image"} if piece in (IMAGE_MARKER, FEEDBACK_IMAGE_MARKER) else {"type": "text", "text": piece}
            for piece in user_parts(prompt["user"])]
    messages = ([{"role": "system", "content": prompt["system"]}] if prompt["system"] else []) + \
               [{"role": "user", "content": user}]
    try:
        return llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:  # the text-only tokenizer may not have the multimodal template; this file is for reading
        return f"(chat template failed: {e!r})\n\n{json.dumps(messages, indent=2)}"


# ---- Model -----------------------------------------------------------------------------------------------------

def load_model(args: argparse.Namespace) -> "LLM":
    """Load the model once with vLLM, sharded across tensor_parallel_size GPUs, with room for the original and the
    feedback image per prompt."""
    from vllm import LLM
    return LLM(model=args.model, seed=args.seed, max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=args.tensor_parallel_size,
               limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT, "video": 0},
               mm_processor_cache_gb=args.mm_processor_cache_gb)


# ---- Image-edit tutor (vLLM-Omni server) -----------------------------------------------------------------------

SERVER_POLL_SECONDS = 5
SERVER_STOP_TIMEOUT = 60      # seconds the server gets to exit after SIGTERM before SIGKILL
EDIT_REQUEST_TIMEOUT = 1800   # seconds per image-edit request
# vLLM-Omni's own startup limits (--init-timeout 600 s, --stage-init-timeout 300 s by default) killed the server while
# its 4 GPU workers were still reading the 39 GB transformer from scratch (~5 min per shard, job 19903876). They are set
# out of the way; --image_edit_server_timeout (no log progress) decides when to give up instead.
OMNI_INIT_TIMEOUT = 24 * 3600
SERVER_ERROR_LINE = re.compile(r"Error\b|timed out|Killed|out of memory", re.I)
TRACEBACK_LINE = re.compile(r"^\s*(File \"|\^|Traceback)|\] +(File \"|\^)")


def free_port() -> int:
    """A TCP port that is free right now."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_log_summary(log_path: Path, n_lines: int = 12) -> str:
    """The server log's last error lines (no traceback frames, no colour codes), or its last lines if none."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).rstrip() for l in log_path.read_text(errors="replace").splitlines()]
    errors = [l for l in lines if SERVER_ERROR_LINE.search(l) and not TRACEBACK_LINE.search(l)]
    return "\n".join((errors or lines)[-n_lines:])


def hf_snapshot_dir(model: str) -> Path:
    """The downloaded snapshot of an HF model in $HF_HOME/hub (the revision refs/main points to)."""
    repo = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / \
        f"models--{model.replace('/', '--')}"
    snapshot = repo / "snapshots" / (repo / "refs" / "main").read_text().strip()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"No downloaded snapshot of {model} at {snapshot}")
    return snapshot


def dir_size(path: Path) -> int:
    """Total bytes of the files under path, following the HF cache's symlinks to blobs."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy_model_to_local(model: str, local_dir: str) -> Optional[Path]:
    """Copy the model's snapshot (symlinks resolved) to <local_dir>/<model name> and return that path, or None (serve
    from scratch) if local_dir is empty, missing or too small."""
    import shutil
    if not local_dir:
        return None
    local = Path(local_dir)
    if not local.is_dir():
        logger.warning("local model dir %r does not exist; serving %s from scratch (slow)", local_dir, model)
        return None
    snapshot = hf_snapshot_dir(model)
    size, free = dir_size(snapshot), shutil.disk_usage(local).free
    if free < size * 1.05:
        logger.warning("local model dir %s has %.1f GB free, %s needs %.1f GB; serving from scratch (slow). "
                       "Request more with #SBATCH --tmp", local, free / 1e9, model, size / 1e9)
        return None
    dest = local / model.replace("/", "--")
    t0 = time.time()
    shutil.copytree(snapshot, dest, dirs_exist_ok=True)  # follows symlinks: real files, no HF cache layout
    elapsed = time.time() - t0
    logger.info("copied %s (%.1f GB) to %s in %.0fs (%.0f MB/s)", model, size / 1e9, dest, elapsed,
                size / 1e6 / max(elapsed, 1e-3))
    return dest


def start_image_edit_server(args: argparse.Namespace, log_path: Path,
                            model_path: Optional[Path] = None) -> Tuple["subprocess.Popen", str]:
    """Start 'vllm serve <tutor_model> --omni' from --image_edit_venv on all GPUs; return (process, base URL) once
    /health answers. The server runs in its own process group so that stopping it also stops its GPU workers.
    model_path: a local copy of the model to load instead (still served under the name --tutor_model).

    Loading from scratch takes long (weights read by every GPU worker), so there is no overall deadline: the server is
    given up on only when its log has not grown for --image_edit_server_timeout seconds."""
    import shlex
    import subprocess
    import requests
    offload = args.image_edit_offload_config
    if offload.lower() != "none":
        json.loads(offload)  # fail here, not after the server has started loading
    port = free_port()
    venv = args.image_edit_venv.resolve()
    cmd = [str(venv / "bin" / "vllm"), "serve", str(model_path or args.tutor_model), "--omni",
           "--served-model-name", args.tutor_model, "--host", "127.0.0.1",
           "--port", str(port), "--tensor-parallel-size", str(args.tensor_parallel_size),
           "--init-timeout", str(OMNI_INIT_TIMEOUT), "--stage-init-timeout", str(OMNI_INIT_TIMEOUT),
           *([] if offload.lower() == "none" else ["--diffusion-offload-config", offload])]
    env = {**os.environ, "VIRTUAL_ENV": str(venv), "PATH": f"{venv / 'bin'}:{os.environ.get('PATH', '')}"}
    logger.info("starting image-edit server (log: %s): %s", log_path, shlex.join(cmd))
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    t0 = last_progress = time.time()
    log_size = 0
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"image-edit server exited with code {proc.returncode} after {time.time() - t0:.0f}s; "
                               f"see {log_path}. Its last errors:\n{server_log_summary(log_path)}")
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        size = log_path.stat().st_size
        if size != log_size:
            log_size, last_progress = size, time.time()
        if time.time() - last_progress > args.image_edit_server_timeout:
            stop_image_edit_server(proc)
            raise TimeoutError(f"image-edit server log has not changed for {args.image_edit_server_timeout}s (stuck "
                               f"after {time.time() - t0:.0f}s); see {log_path}. Its last lines:\n"
                               f"{server_log_summary(log_path)}")
        time.sleep(SERVER_POLL_SECONDS)
    logger.info("image-edit server ready in %.0fs at %s", time.time() - t0, url)
    return proc, url


def stop_image_edit_server(proc: "subprocess.Popen") -> None:
    """SIGTERM the server's process group, then SIGKILL whatever is left, so its GPU memory is freed."""
    import signal
    import subprocess
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=SERVER_STOP_TIMEOUT)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:  # workers can outlive the main server process and keep holding GPU memory
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def edit_image(url: str, model: str, png: bytes, instruction: str, seed: int, args: argparse.Namespace) -> bytes:
    """One /v1/images/edits request; returns the edited image as PNG bytes."""
    import requests
    resp = requests.post(f"{url}/v1/images/edits", timeout=EDIT_REQUEST_TIMEOUT,
                         files={"image": ("image.png", png, "image/png")},
                         data={"model": model, "prompt": instruction, "n": 1, "seed": seed,
                               "num_inference_steps": args.image_edit_steps,
                               "true_cfg_scale": args.image_edit_true_cfg_scale,
                               "negative_prompt": args.image_edit_negative_prompt,
                               "response_format": "b64_json", "output_format": "png"})
    if resp.status_code != 200:
        raise RuntimeError(f"image edit failed with HTTP {resp.status_code}: {resp.text[:1000]}")
    return base64.b64decode(resp.json()["data"][0]["b64_json"])


def run_image_tutor(records: List[Record], instances: Dict[int, "Instance"], template: str, values_fn,
                    output_dir: Path, generations: Path, args: argparse.Namespace) -> None:
    """Tutor stage for --tutor_output image: start the image-edit server, edit each record's image with the filled
    tutor prompt, save it to feedback_images/, and stop the server (also on failure)."""
    image_dir = output_dir / "feedback_images"
    image_dir.mkdir(exist_ok=True)
    model_path = copy_model_to_local(args.tutor_model, args.local_model_dir)
    proc, url = start_image_edit_server(args, output_dir / "image_edit_server.log", model_path)
    try:
        t0 = time.time()
        for i, r in enumerate(records):
            r.prompts["tutor"] = build_prompt(template, values_fn(r))
            instruction = edit_instruction(r.prompts["tutor"])
            r.prompts["tutor"]["edit_instruction"] = instruction
            png, _ = image_png(instances[r.instance_idx].image_path, args.max_image_pixels)
            t1 = time.time()
            edited = edit_image(url, args.tutor_model, png, instruction, args.seed + i, args)
            path = image_dir / f"instance{r.instance_idx}_sample{r.sample_idx}.png"
            path.write_bytes(edited)
            r.feedback_image, r.tutor_status = str(path), "ok"
            write_generations(generations, records)
            done = i + 1
            elapsed = time.time() - t0
            logger.info("tutor: %d/%d images edited (image %.1fs, stage %.0fs, ETA %.0fs) -> %s", done, len(records),
                        time.time() - t1, elapsed, elapsed / done * (len(records) - done), path)
        logger.info("tutor done: %d images in %.1fs", len(records), time.time() - t0)
    finally:
        stop_image_edit_server(proc)
        logger.info("image-edit server stopped")


def generate(llm: "LLM", conversations: List[List[Dict[str, Any]]], temperature: float, top_p: float,
             max_tokens: int, seeds: List[int]) -> List[str]:
    """Sample one chat response per conversation, conversation i seeded with seeds[i]."""
    from vllm import SamplingParams
    params = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=s) for s in seeds]
    outputs = llm.chat(conversations, params, use_tqdm=False)
    return [out.outputs[0].text for out in outputs]


# ---- Parsing ---------------------------------------------------------------------------------------------------

JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def string_field(text: str, key: str) -> Optional[str]:
    """Value of "key": "..." in text that is not valid JSON (e.g. the template's stray ':' after the value)."""
    m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if not m:
        return None
    try:
        return json.loads(f'"{m.group(1)}"')
    except json.JSONDecodeError:
        return m.group(1)


def parse_json(text: str, keys: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Parse a model's JSON answer: a ```json block or the outermost {...}; else pull the string keys one by one."""
    candidates = [m.group(1) for m in JSON_FENCE.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    fields = {k: string_field(text, k) for k in keys}
    return fields if any(v is not None for v in fields.values()) else None


def parse_tutor(raw: str) -> Tuple[str, str]:
    """(status, feedback). If no feedback field is found the whole answer is used, so the chain still runs."""
    obj = parse_json(raw, ("feedback",))
    feedback = obj.get("feedback") if obj else None
    if isinstance(feedback, str) and feedback.strip():
        return "ok", feedback.strip()
    return "parse_error_raw_used", raw.strip()


def parse_judge(raw: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """(status, category, final_answer, reasoning); status is ok, invalid_category or parse_error."""
    obj = parse_json(raw, ("category", "final_answer", "reasoning"))
    if not obj:
        return "parse_error", None, None, None
    category = str(obj.get("category") or "").strip().lower() or None
    final = obj.get("final_answer")
    final = None if final is None or str(final).strip().lower() in ("", "null", "none") else str(final).strip()
    reasoning = obj.get("reasoning")
    reasoning = str(reasoning).strip() if reasoning is not None else None
    status = "ok" if category in JUDGE_CATEGORIES else "invalid_category"
    return status, category, final, reasoning


# ---- Pipeline --------------------------------------------------------------------------------------------------

def short(text: Optional[str], max_chars: int) -> str:
    """One-line excerpt of a model output for the log."""
    text = " ".join((text or "").split())
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def run_stage(llm: "LLM", role: str, records: List[Record], urls_fn, template: str,
              values_fn, temperature: float, store_fn, status_fn, example_fn, generations: Path,
              args: argparse.Namespace) -> None:
    """Run one role over all records in batches of chunk_size.

    urls_fn(record) gives the record's image data URLs by marker (see to_messages).
    Per batch: build the prompts, generate, store the outputs with store_fn(record, raw), rewrite generations.jsonl
    and log progress, the status counts so far (status_fn) and one example output (example_fn).
    Chain i is seeded with seed + stage * n_chains + i, independent of the batch size.
    """
    offset = ROLES.index(role) * len(records)
    t0 = time.time()
    for start in range(0, len(records), args.chunk_size):
        chunk = records[start:start + args.chunk_size]
        for r in chunk:
            r.prompts[role] = build_prompt(template, values_fn(r))
        conversations = [to_messages(r.prompts[role], urls_fn(r)) for r in chunk]
        seeds = [args.seed + offset + start + i for i in range(len(chunk))]
        t1 = time.time()
        outputs = generate(llm, conversations, temperature, args.top_p, args.max_tokens, seeds)
        for r, raw in zip(chunk, outputs):
            store_fn(r, raw)
        write_generations(generations, records)
        done = start + len(chunk)
        elapsed = time.time() - t0
        eta = elapsed / done * (len(records) - done)
        statuses = dict(Counter(status_fn(r) for r in records[:done])) if status_fn else {}
        logger.info("%s: %d/%d chains (batch %.1fs, stage %.0fs, ETA %.0fs)%s", role, done, len(records),
                    time.time() - t1, elapsed, eta, f" {statuses}" if statuses else "")
        logger.info("%s example (instance %d, %s): %s", role, chunk[0].instance_idx, chunk[0].image_name,
                    short(example_fn(chunk[0]), args.log_chars))
    logger.info("%s done: %d chains in %.1fs", role, len(records), time.time() - t0)


def write_generations(path: Path, records: List[Record]) -> None:
    """Write one JSON line per chain (rewritten after every batch, so a crash keeps everything generated so far).
    Written to a temporary file and renamed, so a reader never sees a half-written file."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    tmp.replace(path)


def plot_categories(path: Path, counts: Counter, n: int) -> None:
    """Bar chart of the judge's categories (plus invalid / unparsed answers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(JUDGE_CATEGORIES) + ["invalid / parse error"]
    heights = [counts[c] for c in JUDGE_CATEGORIES] + [n - sum(counts[c] for c in JUDGE_CATEGORIES)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), heights)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("chains")
    ax.set_title(f"judge categories ({n} chains)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    """Select instances, run tutor, student and judge in turn, and save prompts, generations, summary and chart."""
    args = parse_args()
    image_tutor = args.tutor_output == "image"
    if not image_tutor and args.tutor_model != args.model:
        sys.exit("--tutor_output text uses --model as the tutor; set --tutor_model to the same model")
    if "SLURM_JOB_ID" not in os.environ:
        sys.exit("feedback_pipeline.py must run inside a Slurm job: sbatch scripts/run-feedback-pipeline.sh")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    started = time.time()

    prompts_yaml = args.prompts_yaml.resolve()
    templates = {role: load_template(spec, prompts_yaml)
                 for role, spec in zip(ROLES, (args.tutor_prompt, args.student_prompt, args.judge_prompt))}
    answer_key = load_answer_key(args.correct_answers_json)
    instances, selection = load_instances(args.dataset_csv, args.images_dir, args.n_instances,
                                          args.misconception_source, args.seed, answer_key, args.correct_answer)
    logger.info("selected %d of %d images with a teacher-reported error (%d images in total)", len(instances),
                selection["n_with_error"], selection["n_images"])
    selection["n_with_key_answer"] = sum(i.image_name in answer_key for i in instances)
    logger.info("%d of %d selected images have an answer-key answer", selection["n_with_key_answer"], len(instances))
    images, image_sizes = {}, {}
    t0 = time.time()
    for inst in instances:
        images[inst.instance_idx], image_sizes[inst.instance_idx] = image_data_url(inst.image_path,
                                                                                   args.max_image_pixels)
    logger.info("encoded %d images in %.1fs", len(images), time.time() - t0)
    records = [Record(inst.instance_idx, s, inst.problem_id, inst.image_name, inst.misconception, inst.teacher_error,
                      inst.correct_answer)
               for inst in instances for s in range(args.n_samples)]
    generations = output_dir / "generations.jsonl"
    tutor_values = lambda r: {"MISCONCEPTION_DESCRIPTION": r.misconception, "CORRECT_ANSWER": r.correct_answer}

    # 1a. Image tutor: edited images from the image-edit server, which is stopped before --model is loaded.
    if image_tutor:
        run_image_tutor(records, {i.instance_idx: i for i in instances}, templates["tutor"][1], tutor_values,
                        output_dir, generations, args)
    feedback_urls = {id(r): image_data_url(r.feedback_image, args.max_image_pixels)[0]
                     for r in records if r.feedback_image}

    def urls(r: Record) -> Dict[str, str]:
        out = {IMAGE_MARKER: images[r.instance_idx]}
        if id(r) in feedback_urls:
            out[FEEDBACK_IMAGE_MARKER] = feedback_urls[id(r)]
        return out

    def feedback_values(r: Record) -> Dict[str, str]:
        return {"FEEDBACK_TEXT": r.feedback} if r.feedback is not None else {}

    llm = load_model(args)
    start_child_watchdog()

    def store_tutor(r: Record, raw: str) -> None:
        r.tutor_raw = raw
        r.tutor_status, r.feedback = parse_tutor(raw)

    def store_student(r: Record, raw: str) -> None:
        r.student_raw = raw.strip()

    def store_judge(r: Record, raw: str) -> None:
        r.judge_raw = raw
        r.judge_status, r.category, r.final_answer, r.judge_reasoning = parse_judge(raw)

    # 1b. Text tutor feedback.
    if not image_tutor:
        run_stage(llm, "tutor", records, urls, templates["tutor"][1], tutor_values, args.tutor_temperature,
                  store_tutor, lambda r: r.tutor_status, lambda r: r.feedback, generations, args)
    # 2. Student response to the feedback.
    run_stage(llm, "student", records, urls, templates["student"][1], feedback_values,
              args.student_temperature, store_student, None, lambda r: r.student_raw, generations, args)
    # 3. Judgment of the student's response.
    run_stage(llm, "judge", records, urls, templates["judge"][1],
              lambda r: {"CORRECT_ANSWER": r.correct_answer, "STUDENT_RESPONSE": r.student_raw, **feedback_values(r)},
              args.judge_temperature, store_judge, lambda r: r.category if r.judge_status == "ok" else r.judge_status,
              lambda r: f"{r.category} | {r.judge_reasoning}", generations, args)

    # Templates and the first chain's filled prompts, as the model sees them.
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    for role, (name, template) in templates.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        (prompt_dir / f"{role}.{safe}.template.txt").write_text(template)
        first = records[0].prompts[role]
        (prompt_dir / f"{role}.{safe}.instance_0.user.txt").write_text(
            (f"### System\n{first['system']}\n### User\n" if first["system"] else "") + first["user"])
        chat = first["edit_instruction"] if "edit_instruction" in first else render_chat(llm, first)
        (prompt_dir / f"{role}.{safe}.instance_0.chat.txt").write_text(chat)

    categories = Counter(r.category for r in records if r.judge_status == "ok")
    import torch
    import vllm
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "models": {"tutor": args.tutor_model, "student": args.model, "judge": args.model},
        "prompts": {role: name for role, (name, _) in templates.items()},
        "seed_scheme": "instances drawn with random.Random(seed).sample from images with a teacher-reported error; "
                       "chain i uses SamplingParams(seed=seed + stage * n_chains + i), stage 0/1/2 = tutor/student/judge",
        "selection": {**selection, "rule": "teacher answer to 'What errors does the student make ...' does not say "
                                           "there is no error (regex NO_ERROR)"},
        "instances": [{"instance_idx": i.instance_idx, "problem_id": i.problem_id, "image_name": i.image_name,
                       "image_size_sent": image_sizes[i.instance_idx], "teacher_error": i.teacher_error,
                       "correct_answer": i.correct_answer}
                      for i in instances],
        "n_chains": len(records),
        "tutor_status": dict(Counter(r.tutor_status for r in records)),
        "judge_status": dict(Counter(r.judge_status for r in records)),
        "categories": {c: categories[c] for c in JUDGE_CATEGORIES},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "python": sys.version.split()[0]},
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "runtime_seconds": round(time.time() - started, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_categories(output_dir / "judge_categories.png", categories, len(records))
    logger.info("tutor %s | judge %s | categories %s", summary["tutor_status"], summary["judge_status"],
                {c: n for c, n in summary["categories"].items() if n})
    logger.info("done in %.1fs; results in %s", time.time() - started, output_dir)


CHILD_TERMINATE_TIMEOUT = 10  # seconds child processes get to exit after SIGTERM before SIGKILL
WATCHDOG_SECONDS = 30         # how often the watchdog checks that vLLM's processes are alive


def start_child_watchdog() -> None:
    """Exit with an error when a vLLM child process dies (e.g. killed for exceeding the job's memory).

    Otherwise llm.chat waits forever for the dead engine and the job hangs until the Slurm time limit.
    """
    import threading
    import psutil
    children = psutil.Process().children(recursive=True)

    def watch() -> None:
        while True:
            time.sleep(WATCHDOG_SECONDS)
            for child in children:
                try:
                    dead = child.status() == psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    dead = True
                if dead:
                    logger.error("vLLM process %d died (out of CPU memory? see sacct MaxRSS); exiting", child.pid)
                    try:
                        stop_children()
                    finally:
                        logging.shutdown()
                        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="child-watchdog").start()


def stop_children() -> None:
    """Terminate (then kill) every child process, e.g. vLLM's EngineCore and GPU workers, which otherwise keep the
    job alive, holding its GPUs until the Slurm time limit, after the main code has finished or failed."""
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=CHILD_TERMINATE_TIMEOUT)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def run_and_exit() -> None:
    """Run main(), then always stop child processes and exit (1 on error) so the Slurm job ends and frees its GPUs."""
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        logger.exception("run failed; stopping vLLM processes and exiting")
        if not logger.handlers:
            import traceback
            traceback.print_exc()
    finally:
        try:
            stop_children()
        finally:
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)


if __name__ == "__main__":
    run_and_exit()
