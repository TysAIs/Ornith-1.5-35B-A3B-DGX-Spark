#!/usr/bin/env bash
# Start Ornith-1.5-35B-A3B (NVFP4) on one DGX Spark (GB10) using the
# spark-vllm-docker B12X image (eugr/spark-vllm-b12x) with the GB10-native
# b12x MoE backend. (flashinfer_b12x is rejected for the unquantized MTP
# draft layers; plain b12x maps to the NvFP4-native path for the main model
# and to FlashInfer cutlass for the draft, which is valid.)
#
# Per README: ensures weights are in the user's home HF cache, then serves.
# The B12X Docker image is pulled automatically from Docker Hub if absent.
set -euo pipefail

IMAGE="eugr/spark-vllm-b12x:latest"
CONTAINER="ornith-b12x-serve"
REPO="ornith-ai/Ornith-1.5-35B-A3B-NVFP4"
SERVED_MODEL_NAME="ornith-1.5-35b-a3b-nvfp4"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8889}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/vllm}"
CONTAINER_HF="/root/.cache/huggingface"
CONTAINER_VLLM="/root/.cache/vllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- 1. HF token gate (README warning: bare hf can hang on HTTP 401 if not exported) --
if [[ ! -f "$HF_CACHE/token" ]]; then
  echo "No HF token found at $HF_CACHE/token" >&2
  exit 1
fi
export HF_TOKEN="$(cat "$HF_CACHE/token")"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# -- 2. Ensure weights are cached in the user's home HF cache (idempotent, resumable) --
echo "Ensuring $REPO is downloaded into $HF_CACHE/hub ..."
hf download "$REPO"

# Resolve the cached snapshot path.
SNAPSHOT_DIR="$HF_CACHE/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4"
MODEL=""
for cand in "$SNAPSHOT_DIR"/snapshots/*/; do
  if [[ -f "${cand}config.json" && -f "${cand}model.safetensors.index.json" ]]; then
    MODEL="${cand%/}"
    break
  fi
done
if [[ -z "$MODEL" ]]; then
  echo "Could not find a complete snapshot under $SNAPSHOT_DIR" >&2
  exit 1
fi
MODEL_IN_CONTAINER="$CONTAINER_HF/${MODEL#$HF_CACHE/}"
echo "Using cached weights: $MODEL"

# -- 3. Ensure the B12X Docker image is available locally; pull if not. --
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found locally; pulling (may take a while)..."
  docker pull "$IMAGE"
fi

# -- 4. Launch. Remove any stale container of the same name first, then probe
# free device memory — else the probe counts the stale server's held memory
# and we'd launch with a crippled util. (Container shares the device, the host
# probe is representative.) README's 0.85 assumes an idle Spark; we request
# what's actually free minus a margin instead of failing. --
PROBE_BIN="$(command -v vllm)"
if [[ -n "$PROBE_BIN" ]]; then
  PYBIN="$(head -1 "$PROBE_BIN" | sed 's/^#!//')"
else
  PYBIN="python3"
fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
GPU_UTIL=0.85
# Wait for the stale server's memory to actually be released (can lag a rm -f
# by many seconds), then probe.
for _ in $(seq 1 30); do
  DEVINFO="$("$PYBIN" -c 'import torch; f,t=torch.cuda.mem_get_info(); print(int(f//1048576), int(t//1048576))' 2>/dev/null || true)"
  read -r FREE_MIB TOTAL_MIB <<< "$DEVINFO"
  [[ -n "$FREE_MIB" && -n "$TOTAL_MIB" ]] && (( FREE_MIB > 64*1024 )) && break
  sleep 2
  [[ -n "$FREE_MIB" ]] || { unset FREE_MIB TOTAL_MIB; continue; }
done
if [[ -n "${FREE_MIB:-}" && -n "${TOTAL_MIB:-}" ]]; then
  MEM_RATIO=$(awk -v f="$FREE_MIB" -v t="$TOTAL_MIB" 'BEGIN { if (t>0) printf "%.3f", f/t; else printf "1.0" }')
  GPU_UTIL=$(awk -v r="$MEM_RATIO" 'BEGIN { u=r*0.95; if (u>0.85) u=0.85; if (u<0.2) u=0.2; printf "%.2f", u }')
  if awk -v u="$GPU_UTIL" 'BEGIN { exit !(u < 0.85) }'; then
    echo "Warning: only ${FREE_MIB} MiB / ${TOTAL_MIB} MiB device memory free at launch." >&2
    echo "         Lowering --gpu-memory-utilization to $GPU_UTIL. (README's 0.85 needs an idle Spark;" >&2
    echo "         stop competing servers for the full KV-cache size.)" >&2
  fi
else
  echo "Warning: device memory probe failed; using 0.85." >&2
fi
echo "Starting container $CONTAINER (b12x MoE backend)..."
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc=host \
  -v "$HF_CACHE:$CONTAINER_HF" \
  -v "$VLLM_CACHE:$CONTAINER_VLLM" \
  -v "$SCRIPT_DIR/patches/kernel.py:/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py:ro" \
  -v "$SCRIPT_DIR/patches/route_pack.py:/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py:ro" \
  "$IMAGE" \
  vllm serve "$MODEL_IN_CONTAINER" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --enable-prefix-caching \
    --spec-method mtp --spec-tokens 1 \
    --max-num-seqs 24 \
    --moe-backend b12x \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 262144 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt image=1,video=1

echo "Launched container $CONTAINER (image $IMAGE)."
echo "Following the live vLLM log until the server is healthy..."

# -- 5. Follow the container log in the background and block until the server
# is healthy (or clearly failed / timed out). Return to the shell only on
# success. Ctrl+C aborts the follow (container keeps running).
LOG_PID=""
cleanup() { [[ -n "$LOG_PID" ]] && kill "$LOG_PID" 2>/dev/null || true; }
trap cleanup EXIT
docker logs -f --tail=50 "$CONTAINER" &
LOG_PID=$!

HEALTH_URL="http://$HOST:$PORT/health"
TIMEOUT_S=1800
DEADLINE=$(( $(date +%s) + TIMEOUT_S ))
while :; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo >&2
    echo "ERROR: container $CONTAINER exited before becoming healthy." >&2
    docker logs --tail 50 "$CONTAINER" >&2 || true
    exit 1
  fi
  if curl -fs --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo
    echo "Health OK: $HEALTH_URL"
    echo "vLLM live — returning you to the shell. (Logs: docker logs -f $CONTAINER)"
    exit 0
  fi
  if (( $(date +%s) >= DEADLINE )); then
    echo >&2
    echo "ERROR: /health not green after ${TIMEOUT_S}s." >&2
    echo "         Check logs: docker logs -f $CONTAINER" >&2
    exit 1
  fi
  sleep 2
done
