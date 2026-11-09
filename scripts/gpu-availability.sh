#!/bin/bash
# Free GPUs per type on Euler, with memory per GPU and a rough upper bound on the model size vLLM can load.
# Usage: bash scripts/gpu-availability.sh        (a cheap scontrol query, fine on a login node)
#
# Columns:
#   mem/GPU      memory of one GPU (from the node's gpumem gres)
#   free/total   GPUs free right now / all GPUs of that type; offline = drained, down or reserved nodes
#   max/node     most GPUs free on a single node: the largest tensor-parallel size you could start now
#   vLLM         "ok" if compute capability >= 8.0; older GPUs make vLLM fall back to FlexAttention, which crashes
#   ~bf16 1 GPU  / ~bf16 node: rough largest model in bf16 on 1 GPU / on max/node GPUs, assuming 2 bytes per
#                parameter and ~25% of memory left for KV cache and activations. FP8 roughly doubles it.
# Free GPUs do not guarantee an immediate start; check with: sbatch --test-only --gpus=<type>:N --time=... --wrap=true
set -euo pipefail

export SLURM_CONF="${SLURM_CONF:-/cluster/adm/slurm-24-production/slurm/etc/slurm.conf}"
KV_RESERVE=0.25

scontrol -o show nodes | awk -v reserve="${KV_RESERVE}" '
BEGIN {
  cc["nvidia_geforce_rtx_2080_ti"] = 7.5; cc["nvidia_titan_rtx"] = 7.5; cc["quadro_rtx_6000"] = 7.5
  cc["nvidia_geforce_rtx_3090"] = 8.6; cc["nvidia_geforce_rtx_4090"] = 8.9
  cc["nvidia_a100-pcie-40gb"] = 8.0; cc["nvidia_a100_80gb_pcie"] = 8.0; cc["nvidia_rtx_pro_6000"] = 12.0
}
/Gres=gpu:/ {
  match($0, /Gres=gpu:[^:]+:[0-9]+/); split(substr($0, RSTART, RLENGTH), g, ":"); t = g[2]; n = g[3] + 0
  if (match($0, /gpumem:[a-z_]*:?[0-9]+G/)) { m = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", m); mem[t] = m + 0 }
  used = 0
  if (match($0, /AllocTRES=[^ ]*gres\/gpu=[0-9]+/)) { s = substr($0, RSTART, RLENGTH); sub(/.*gres\/gpu=/, "", s); used = s + 0 }
  match($0, /State=[^ ]+/); st = substr($0, RSTART + 6, RLENGTH - 6)
  tot[t] += n
  if (st ~ /DRAIN|DOWN|MAINT|RESERVED|FAIL/) { off[t] += n; next }
  f = n - used; free[t] += f; if (f > maxnode[t]) maxnode[t] = f
}
END {
  printf "%-28s %8s %11s %8s %9s %5s %12s %11s\n", "type", "mem/GPU", "free/total", "offline", "max/node", "vLLM", "~bf16 1 GPU", "~bf16 node"
  for (t in tot) {
    per = mem[t] * (1 - reserve) / 2
    node = maxnode[t] > 0 ? sprintf("%.0fB", per * maxnode[t]) : "-"
    printf "%-28s %7dG %5d/%-5d %8d %9d %5s %11.0fB %11s\n", t, mem[t], free[t], tot[t], off[t], maxnode[t],
           (t in cc && cc[t] >= 8.0) ? "ok" : "no", per, node
  }
}' | sort -k2 -n
