#!/usr/bin/env python3
"""Quick prefill/decode tok/s probe against an OpenAI-compatible endpoint."""
import argparse
import json
import time
import urllib.request


def chat(base_url, model, prompt, max_tokens, api_key="EMPTY"):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    usage = d.get("usage", {})
    return dt, usage, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    args = p.parse_args()

    short_prompt = "Write a one-sentence fact about the ocean."
    dt, usage, _ = chat(args.base_url, args.model, short_prompt, 200, args.api_key)
    ptoks = usage.get("prompt_tokens", 0)
    ctoks = usage.get("completion_tokens", 0)
    print(f"SHORT_PROMPT wall={dt:.2f}s prompt_tok={ptoks} completion_tok={ctoks} "
          f"approx_overall_tps={ctoks/dt:.2f}")

    long_prompt = ("Summarize the plot of a story about a lighthouse keeper. " * 400) + \
        " Now write a haiku about the sea."
    dt2, usage2, _ = chat(args.base_url, args.model, long_prompt, 16, args.api_key)
    ptoks2 = usage2.get("prompt_tokens", 0)
    ctoks2 = usage2.get("completion_tokens", 0)
    print(f"LONG_PROMPT wall={dt2:.2f}s prompt_tok={ptoks2} completion_tok={ctoks2} "
          f"approx_prefill_tps={ptoks2/dt2:.2f}")

    med_prompt = "Explain how a binary search tree works, with an example insertion sequence."
    dt3, usage3, resp3 = chat(args.base_url, args.model, med_prompt, 300, args.api_key)
    ptoks3 = usage3.get("prompt_tokens", 0)
    ctoks3 = usage3.get("completion_tokens", 0)
    print(f"MED_PROMPT wall={dt3:.2f}s prompt_tok={ptoks3} completion_tok={ctoks3} "
          f"approx_decode_tps={ctoks3/dt3:.2f}")


if __name__ == "__main__":
    main()
