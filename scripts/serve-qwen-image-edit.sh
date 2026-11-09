#!/bin/bash
#SBATCH --job-name=qwen-image-edit
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:4                 # keep equal to TENSOR_PARALLEL_SIZE below
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --output=qwen-image-edit-%j.log
#
# Serves Qwen/Qwen-Image-Edit-2511 with vLLM-Omni as an OpenAI-style API, sharded across GPUs. Submit from SP/:
#   sbatch scripts/serve-qwen-image-edit.sh
# Then find the node (squeue -j <JOBID> -o "%N") and send requests to http://<node>:8091/v1/images/edits,
# e.g. python scripts/test-qwen-image-edit.py <node> input.png "make the pen red"
#
# Memory: the model is ~55 GB in bf16 (39 GB diffusion transformer, 16 GB Qwen2.5-VL text encoder).
# Tensor parallelism splits only the transformer (39 GB / 4 ≈ 10 GB per GPU). The text encoder would otherwise be
# copied whole onto every GPU, so it is kept in CPU RAM (split across ranks) and streamed in layer by layer.
# With 2x a100_80gb you can set TENSOR_PARALLEL_SIZE=2 and OFFLOAD_TEXT_ENCODER=0 (faster prompt encoding).
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this through Slurm: sbatch scripts/serve-qwen-image-edit.sh" >&2
  exit 1
fi

source "${SLURM_SUBMIT_DIR}/scripts/telegram-hooks.sh"

TENSOR_PARALLEL_SIZE=4     # shard the transformer across this many GPUs (must match #SBATCH --gpus)
OFFLOAD_TEXT_ENCODER=1     # 1 = text encoder in CPU RAM, streamed per layer (needed on 24 GB GPUs)
PORT=8091
MODEL="Qwen/Qwen-Image-Edit-2511"

MIN_COMPUTE_CAP=8.0
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
echo "GPUs:"; echo "${GPU_INFO}"
if [[ $(wc -l <<<"${GPU_INFO}") -ne ${TENSOR_PARALLEL_SIZE} ]]; then
  echo "Allocated GPUs != TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}; fix #SBATCH --gpus." >&2
  exit 1
fi
if ! awk -F', ' -v min="${MIN_COMPUTE_CAP}" '$2 < min { bad = 1 } END { exit bad }' <<<"${GPU_INFO}"; then
  echo "A GPU has compute capability < ${MIN_COMPUTE_CAP}; vLLM will crash. Request rtx_3090/rtx_4090/a100." >&2
  exit 1
fi

# ---- Environment -------------------------------------------------------------
VENV="/cluster/scratch/${USER}/venvs/vllm-omni"   # vllm 0.32 + vllm-omni 0.32 (omni needs matching versions)
export HF_HOME="/cluster/scratch/${USER}/hf_cache"
export HF_HUB_OFFLINE=1    # weights are downloaded beforehand on a login node
export NCCL_P2P_DISABLE=1  # RTX 4090s have no GPU peer-to-peer; NCCL can hang trying it during tensor parallelism
if command -v module >/dev/null 2>&1; then
  module load eth_proxy
fi
set +u; source "${VENV}/bin/activate"; set -u

EXTRA_ARGS=()
if [[ "${OFFLOAD_TEXT_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--diffusion-offload-config
    '{"mode": "layer", "components": ["text_encoder"], "layer_options": {"text_encoder": {"weight_transfer": "allgather"}}}')
fi

echo "Serving ${MODEL} on http://$(hostname):${PORT}/v1/images/edits"
vllm serve "${MODEL}" --omni \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
