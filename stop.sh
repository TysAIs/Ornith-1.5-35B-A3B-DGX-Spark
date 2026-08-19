#!/usr/bin/env bash
# Stop the Ornith-1.5 serve. Stops and removes the B12X Docker container if
# present; falls back to killing a bare-metal vLLM serve (legacy, pre-Docker).
set -euo pipefail

CONTAINER="ornith-b12x-serve"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker stop "$CONTAINER" >/dev/null
  docker rm "$CONTAINER" >/dev/null
  echo "Stopped and removed container $CONTAINER."
  exit 0
fi

if pkill -f "vllm serve .*ornith-1.5-35b-a3b-nvfp4" 2>/dev/null; then
  echo "Sent SIGTERM to vLLM serve (Ornith-1.5-35B-A3B-NVFP4)"
  for _ in $(seq 1 15); do
    pgrep -f "vllm serve .*ornith-1.5-35b-a3b-nvfp4" >/dev/null 2>&1 || { echo "Stopped."; exit 0; }
    sleep 2
  done
  pkill -9 -f "vllm serve .*ornith-1.5-35b-a3b-nvfp4" || true
  echo "Stopped (forced)."
  exit 0
fi

echo "No running Ornith serve found (no container $CONTAINER, no legacy vLLM process)." >&2
exit 1
