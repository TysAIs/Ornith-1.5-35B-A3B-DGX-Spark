# Ornith-1.5-35B-A3B on one DGX Spark (GB10)

[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Measured serve for [**Ornith-1.5-35B-A3B**](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — a ~35B mixture-of-experts model (~3B active params/token) from the Ornith team — on a single **NVIDIA DGX Spark (GB10 / SM121, ~128 GB unified memory)**. 2026-08-19, day of release.

**Engine: vLLM + NVFP4.** Single node, `--tensor-parallel-size 1`. No multi-Spark cluster required — the NVFP4 checkpoint is ~22 GB and fits comfortably in one Spark's unified memory alongside a large KV cache.

## Scoreboard

| Metric | Value |
|---|---:|
| Decode | **~78 tok/s** |
| Prefill | **~3891 tok/s** |
| sixcat-eval overall | **85.0** |
| hermes-agentic-bench (loop gate) | **14/20 pass (70%)** |

Raw speed measured with three single-stream chat-completion probes (short prompt / long prompt / medium prompt) against the live `/v1/chat/completions` endpoint — see [`logs/quick_tps_bench.py`](logs/quick_tps_bench.py). Not `llama-bench` tg128/pp512; this is real end-to-end vLLM serving throughput on one stream.

## Serve it yourself

Weights: [`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) (~22 GB, ModelOpt NVFP4 checkpoint).

```bash
# 1. Download (needs an authenticated HF token exported first — see note below)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
hf download ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --local-dir ./models/Ornith-1.5-35B-A3B-NVFP4

# 2. Serve (vLLM >= 0.19.1; official recipe uses --tensor-parallel-size 2 for 2x80GB GPUs —
#    a single Spark's 128GB unified memory covers this NVFP4 checkpoint alone, so TP1 is enough)
vllm serve ./models/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name ornith-1.5-35b-a3b-nvfp4 \
  --host 127.0.0.1 --port 8100 \
  --trust-remote-code \
  --enable-prefix-caching \
  --moe-backend marlin \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

Notes:

- **HF token gate.** A bare `hf download` can silently freeze around ~11 MB in if the token in `~/.cache/huggingface/token` isn't exported into the shell's env first (HTTP 401 masquerading as a hang). Export both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` before downloading.
- **`--moe-backend marlin`** — vLLM auto-selects `MarlinNvFp4LinearKernel` for the linear layers and the `MARLIN` NVFP4 MoE backend out of the available set (`FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`, `FLASHINFER_CUTLASS`, `VLLM_CUTLASS`, `MARLIN`, `HUMMING`, `EMULATION`) — Marlin was the fastest/most compatible on this GB10 board.
- **`--max-model-len 65536`** is a deliberate choice, not a ceiling — the model natively handles up to 262144 (and further with YaRN scaling per the model card). One idle Spark had ~92 GB RAM free at load time and reserved ~76 GB for KV cache at this setting; raise it if you want a longer window and have the headroom.
- Model loads in ~2 min on local NVMe (safetensors read is the long pole, not compute); CUDA graph capture + torch.compile warmup adds another ~1 min before `/health` goes green.
- **GGUF path exists too** (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`, Q4_K_M ~21.7GB) but is out of scope for this repo — that's a llama.cpp/Turing-card story, tracked separately, not a DGX Spark recipe.

### Quick health / smoke check

```bash
curl -sf http://127.0.0.1:8100/v1/models
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-1.5-35b-a3b-nvfp4","messages":[{"role":"user","content":"say hi"}],"max_tokens":32}'
```

## Benchmarks

### sixcat-eval

[`vcruz305/sixcat-eval`](https://github.com/vcruz305/sixcat-eval) — six community categories, one overall score, fast local OpenAI-compatible harness.

```bash
git clone https://github.com/vcruz305/sixcat-eval && cd sixcat-eval
python -m sixcat \
  --base-url http://127.0.0.1:8100/v1 \
  --model ornith-1.5-35b-a3b-nvfp4 \
  --limit 20 --max-minutes 30 \
  --out results/sixcat.json
```

| Category | Score | n |
|---|---:|---:|
| **Overall** | **85.0** | 180 |
| Knowledge | 85.0 | 80 |
| Math | 95.0 | 20 |
| Truth | 75.0 | 20 |
| Instruct | 70.0 | 20 |
| Code | 85.0 | 20 |
| Tools | 100.0 | 20 |

Clean run, `timed_out: false`. Full receipt: [`logs/sixcat-nvfp4-20260819.json`](logs/sixcat-nvfp4-20260819.json).

### hermes-agentic-bench

[`vcruz305/hermes-agentic-bench`](https://github.com/vcruz305/hermes-agentic-bench) — Hermes Agent test batteries. Ran the `hermes_loop_gate.py` contract test: 20 Hermes-shaped scripted-tool tasks scored on parse-ok, tool count, duplicate calls, and hitting the consecutive-tool cap. No live Hermes process required — model + OpenAI-compatible server only, so it isolates model weights from harness behavior.

```bash
git clone https://github.com/vcruz305/hermes-agentic-bench && cd hermes-agentic-bench
pip install -r requirements.txt
python hermes_loop_gate.py \
  --base-url http://127.0.0.1:8100/v1 \
  --model ornith-1.5-35b-a3b-nvfp4 \
  --output results_ornith15_nvfp4_gate.json
```

```
SUMMARY {"n_tasks": 20, "n_pass": 14, "pass_rate": 0.7, "mean_tools": 3.05,
         "n_hit_cap": 0, "n_parse_fail_tasks": 0, "n_dup_tasks": 2}
```

**0/20 hit the tool-loop cap** — the model never runs away into a dead loop, which is the failure mode this battery exists to catch. Full receipt: [`logs/hermes-agentic-gate-nvfp4-20260819.json`](logs/hermes-agentic-gate-nvfp4-20260819.json).

## Comparison — other models measured on the same Spark fleet

sixcat overall, same harness/limits, all local:

| Model | Overall | Knowledge | Math | Truth | Instruct | Code | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ornith-1.5-35B-A3B-NVFP4** | **85.0** | 85.0 | 95.0 | 75.0 | 70.0 | 85.0 | 100.0 |
| Qwen3.8-27B stock Q4 | 82.3 | 88.8 | 40.0 | 85.0 | 80.0 | 100.0 | 100.0 |
| Qwen3.8-27B AEON Q4+MTP | 80.0 | 90.0 | 25.0 | 85.0 | 80.0 | 100.0 | 100.0 |
| Nemotron 3.5 Lightning NVFP4 | 57.9 | 17.5 | 80.0 | 65.0 | 35.0 | 60.0 | 90.0 |

Ornith-1.5 leads overall and takes math by a wide margin (95 vs 25–40 for the Qwen builds). No other model in this set has hermes-agentic-bench numbers yet — Ornith-1.5 is the first.

## Hardware

- NVIDIA DGX Spark, GB10, compute capability 12.1, ~128 GB unified memory (reported ~124.6 GB usable)
- Driver 580.159.03, CUDA 13.0, aarch64
- vLLM `0.1.dev1+g75231eff2.d20260809`
- No multi-node RPC, no NCCL, no tensor parallelism — one board, one process

## Files

```
logs/
  sixcat-nvfp4-20260819.json              # full sixcat receipt (categories, items, log path)
  hermes-agentic-gate-nvfp4-20260819.json # full hermes-agentic-bench loop-gate receipt
  quick_tps_bench.py                      # raw prefill/decode probe script
```

## License

MIT — see [LICENSE](LICENSE). Model weights are governed by their own license on the [Ornith-1.5 HF collection](https://huggingface.co/collections/ornith-ai/ornith-15); this repo covers the serving recipe and benchmark receipts only.
