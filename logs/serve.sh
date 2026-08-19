#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/victor/work/k3-vllm-venv/bin:$PATH"
export HF_TOKEN="$(cat /home/victor/.cache/huggingface/token)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
MODEL=/home/victor/models/Ornith-1.5-35B-A3B-NVFP4
OUT=/home/victor/work/ornith15-nvfp4-bench
mkdir -p "$OUT"
exec /home/victor/work/k3-vllm-venv/bin/vllm serve "$MODEL" \
  --served-model-name ornith-1.5-35b-a3b-nvfp4 \
  --host 127.0.0.1 \
  --port 8100 \
  --trust-remote-code \
  --enable-prefix-caching \
  --moe-backend marlin \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
