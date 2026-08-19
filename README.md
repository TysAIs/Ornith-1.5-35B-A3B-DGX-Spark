# Ornith-1.5 35B-A3B for DGX Spark

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Serve **[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** — a ~35B mixture-of-experts model (~3B active params/token) from the Ornith team — on a single **NVIDIA DGX Spark (GB10 / SM121, ~128 GB unified memory)**: NVFP4 checkpoint, vLLM, in-checkpoint MTP speculative decoding, and up to 256k context.

This repository contains a production `start.sh` recipe plus the **two patches required to make the GB10-native `b12x` MoE backend work with CUDA-graph capture** (a PFM / image-level b12x limitation — see [Patches](#patches)).

## Highlights

- **Multi-Token Prediction (MTP) enabled** — vLLM in-checkpoint MTP (`--spec-method mtp --spec-tokens 1`). Ornith-1.5 ships a single MTP layer; one speculative token is the correct (and only) setting for this checkpoint, yielding ~2x the query rate of the base model path.
- **256k context** — `--max-model-len 262144` (the model's native `max_position_embeddings`; further with YaRN scaling).
- **24 concurrent sequences** — `--max-num-seqs 24`, at the edge of KV-cache capacity for full 256k contexts (see [KV cache](#kv-cache)).
- **NVFP4 (~22 GB) fits one Spark** with a large KV cache — no multi-node cluster required.
- **Reproducible, idempotent launch script** that checks the HF token, ensures cached weights, tears down stale containers, measures truly-free device memory, and returns to your shell only once `/health` goes green.

## Decode throughput

Measured end-to-end against the live OpenAI-compatible `/v1/chat/completions` endpoint (single Spark, MTP on, 256k context, dynamic prefill+decode):

| Streams | TTFT | Aggregate | Per stream |
|---:|---:|---:|---:|
| ×1 | 903 ms | 87.7 tok/s | 87.7 tok/s |
| ×2 | 531 ms | 127.5 tok/s | 66.4 tok/s |
| ×4 | 917 ms | 189.3 tok/s | 48.5 tok/s |
| ×6 | 503 ms | 234.7 tok/s | 41.6 tok/s |

Scaling is decent on a single GB10: aggregate decode goes from ~88 tok/s (1 stream) to **~235 tok/s** at 6 concurrent streams with sub-second TTFT throughout — usable as a low-power household/single-board inference server, not a data-center node.

## KV cache

Measured from the live `/metrics` endpoint on the last run (`gpu-memory-utilization 0.83`):

| Metric | Value |
|---|---:|
| KV cache size | **6,479,260 tokens** (~24.7 × 256k) |
| GPU blocks | 3,312 × 2,112 (fp8_e4m3) |
| KV max concurrent 256k sequences | ~24.7 |
| SSM/linear-attention state dtype | float32 (separate from KV) |

- KV cache dtype is **fp8_e4m3** (vLLM default on GB10), ~1 byte/element; the SSM state stays in float32 and is unaffected.
- `--max-num-seqs 24` therefore sits essentially at the KV capacity for full-length contexts: at 262k tokens per sequence the KV budget is what it is, but typical mixed-length workloads never approach it.
- Values scale with the auto-computed `--gpu-memory-utilization` (0.8x on an idle Spark); if you raise max concurrency further, lower `--max-model-len` or accept KV-pressure warnings from the scheduler.

## Serve it yourself

Prerequisites:

- An authenticated Hugging Face token at `~/.cache/huggingface/token` (a bare `hf download` without it silently hangs on HTTP 401).
- Docker with NVIDIA Container Toolkit on the Spark.
- The container image `eugr/spark-vllm-b12x:latest` is pulled automatically on first run if absent.

```bash
# 1. Start (downloads weights on first run; tears down stale containers;
#    tails the live log and returns to the shell only once /health is green)
./start.sh

# 2. In another shell — health and a smoke test
curl -sf http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-1.5-35b-a3b-nvfp4","messages":[{"role":"user","content":"say hi"}],"max_tokens":32}'

# 3. Stop
./stop.sh
```

The script refuses to launch without a token, uses only memory that is actually free at launch (so it never OOMs against a competing server), and waits up to 30 minutes for health before exiting non-zero.

## Patches

Two bugs in the image's `b12x` MoE package crash startup the moment MTP + CUDA graphs are enabled (the pre-MTP `marlin` path did not exercise them). They are fixed here as **host-side file overlays** mounted read-only into the container — no modified Docker image is needed:

1. **`patches/kernel.py`** — undefined `metadata_row` in the W4A16 output-drain path (a leftover of a b12x refactor; the correct value is the loop `row` used two lines above for `route_index`).
2. **`patches/route_pack.py`** — the W4A16 route-packing workspace was `torch.empty`-allocated per kernel call, which is illegal under CUDA graph capture, so `pack_topk_routes_by_expert` raised. The patch substitutes a stable, grow-only per-`(name, dtype, device)` cache seeded by vLLM's pre-capture profile run and reused during capture — consistent with b12x's own documented intent that this workspace "may be shared across all layers on this stream."

Pristine copies of the upstream files are kept as `patches/*.b12x` for reference. The overlay only takes effect for the b12x MoE backend; every other backend is untouched.

## Benchmarks

### sixcat-eval

[`vcruz305/sixcat-eval`](https://github.com/vcruz305/sixcat-eval) — six community categories, one overall score, fast local OpenAI-compatible harness.

| Category | Score | n |
|---|---:|---:|
| **Overall** | **85.0** | 180 |
| Knowledge | 85.0 | 80 |
| Math | 95.0 | 20 |
| Truth | 75.0 | 20 |
| Instruct | 70.0 | 20 |
| Code | 85.0 | 20 |
| Tools | 100.0 | 20 |

Clean run, `timed_out: false`. Full receipt kept locally at `logs/sixcat-nvfp4-20260819.json`.

### hermes-agentic-bench

[`vcruz305/hermes-agentic-bench`](https://github.com/vcruz305/hermes-agentic-bench) — Hermes Agent test batteries. `hermes_loop_gate.py` contract test: 20 Hermes-shaped scripted-tool tasks scored on parse-ok, tool count, duplicate calls, and hitting the consecutive-tool cap.

```
SUMMARY {"n_tasks": 20, "n_pass": 14, "pass_rate": 0.7, "mean_tools": 3.05,
         "n_hit_cap": 0, "n_parse_fail_tasks": 0, "n_dup_tasks": 2}
```

**0/20 hit the tool-loop cap** — the model never runs away into a dead loop. Full receipt kept locally at `logs/hermes-agentic-gate-nvfp4-20260819.json`.

## Comparison — other models measured on the same Spark fleet

sixcat overall, same harness/limits, all local:

| Model | Overall | Knowledge | Math | Truth | Instruct | Code | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ornith-1.5-35B-A3B-NVFP4** | **85.0** | 85.0 | 95.0 | 75.0 | 70.0 | 85.0 | 100.0 |
| Qwen3.8-27B stock Q4 | 82.3 | 88.8 | 40.0 | 85.0 | 80.0 | 100.0 | 100.0 |
| Qwen3.8-27B AEON Q4+MTP | 80.0 | 90.0 | 25.0 | 85.0 | 80.0 | 100.0 | 100.0 |
| Nemotron 3.5 Lightning NVFP4 | 57.9 | 17.5 | 80.0 | 65.0 | 35.0 | 60.0 | 90.0 |

Ornith-1.5 leads overall and takes math by a wide margin (95 vs 25–40 for the Qwen builds).

## Hardware

- NVIDIA DGX Spark, GB10, compute capability 12.1, ~128 GB unified memory (reported ~124.6 GB usable)
- Driver 580.159.03, CUDA 13.0, aarch64
- vLLM build inside `eugr/spark-vllm-b12x:latest` (`v0.1.dev20003`), flashinfer_b12x family kernels
- No multi-node RPC, no NCCL mesh, no tensor parallelism — one board, one process

## Repository layout

```
start.sh                 # idempotent launcher: token gate → weight cache → memory probe → run → wait for health
stop.sh                  # stops/removes the serve container (or legacy bare-metal vLLM)
patches/
  kernel.py              # PATCHED b12x w4a16 kernel (metadata_row fix)
  route_pack.py          # PATCHED b12x route-packing workspace (capture-safe)
  *.b12x                 # pristine upstream copies for diffing
```

> Benchmark receipts and startup logs are kept locally in `logs/` (gitignored) — they are not part of this repository.

Weights are fetched by the script into your HF cache; model weights are governed by their own [Ornith-1.5 HF collection](https://huggingface.co/collections/ornith-ai/ornith-15) license, while this repository is MIT (see [LICENSE](LICENSE)).

## Credits

Recipe skeleton originally from [vcruz305/Ornith-1.5-35B-A3B-DGX-Spark](https://github.com/vcruz305/Ornith-1.5-35B-A3B-DGX-Spark).
