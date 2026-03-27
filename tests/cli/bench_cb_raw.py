"""
Raw continuous batching benchmark — no HTTP, no serve layer.
2x2 matrix: {non_stream, stream} × {legacy get_result, optimized async}.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/cli/bench_cb_raw.py
    CUDA_VISIBLE_DEVICES=0 python tests/cli/bench_cb_raw.py --batch 10 50 100 500 1000 2000
"""

import argparse
import asyncio
import os
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig, GenerationConfig


def make_prompts(tokenizer, n, target_len=256):
    filler = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 100
    ids = tokenizer.encode(filler, add_special_tokens=False)
    return [ids[:max(10, int(target_len * (0.8 + 0.4 * (i % 5) / 4)))] for i in range(n)]


# ---------------------------------------------------------------------------
# Non-streaming (CB streaming=False → one output per request when finished)
# ---------------------------------------------------------------------------


def bench_ns_get_result(mgr, prompts, max_new_tokens):
    """Non-stream + get_result: batch add, poll shared queue."""
    N = len(prompts)
    t0 = time.perf_counter()
    mgr.add_requests(inputs=prompts, max_new_tokens=max_new_tokens, streaming=False)
    total = finished = 0
    while finished < N:
        r = mgr.get_result(timeout=1)
        if r is not None and r.is_finished():
            total += len(r.generated_tokens)
            finished += 1
    return total, time.perf_counter() - t0


async def bench_ns_handler(mgr, prompts, max_new_tokens):
    """Non-stream + handler: register_result_handler per request, resolve future on finish."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    futures = []
    for i, ids in enumerate(prompts):
        rid = f"nsh_{time.perf_counter_ns()}_{i}"
        future = loop.create_future()

        def _on_result(output, fut=future):
            if not fut.done() and output.is_finished():
                fut.set_result(len(output.generated_tokens))

        mgr.register_result_handler(rid, _on_result)
        mgr.add_request(ids, request_id=rid, max_new_tokens=max_new_tokens, streaming=False)
        futures.append(future)
    results = await asyncio.gather(*futures)
    return sum(results), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Streaming (CB streaming=True → one output per token per request)
# ---------------------------------------------------------------------------


def bench_s_get_result(mgr, prompts, max_new_tokens):
    """Stream + get_result: batch add, poll shared queue, skip intermediate outputs."""
    N = len(prompts)
    t0 = time.perf_counter()
    mgr.add_requests(inputs=prompts, max_new_tokens=max_new_tokens, streaming=True)
    total = finished = 0
    while finished < N:
        r = mgr.get_result(timeout=1)
        if r is not None and r.is_finished():
            total += len(r.generated_tokens)
            finished += 1
    return total, time.perf_counter() - t0


async def bench_s_handler(mgr, prompts, max_new_tokens):
    """Stream + handler: register_result_handler per request, await future on finish."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    futures = []
    for i, ids in enumerate(prompts):
        rid = f"sh_{time.perf_counter_ns()}_{i}"
        future = loop.create_future()

        def _on_output(output, fut=future):
            if not fut.done() and output.is_finished():
                fut.set_result(len(output.generated_tokens))

        mgr.register_result_handler(rid, _on_output)
        mgr.add_request(ids, request_id=rid, max_new_tokens=max_new_tokens, streaming=True)
        futures.append(future)

    results = await asyncio.gather(*futures)
    return sum(results), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

METHODS = {
    "ns_get_result": ("Non-stream + get_result", lambda mgr, p, m: bench_ns_get_result(mgr, p, m)),
    "ns_handler":    ("Non-stream + handler",     lambda mgr, p, m: asyncio.run(bench_ns_handler(mgr, p, m))),
    "s_get_result":  ("Stream + get_result",      lambda mgr, p, m: bench_s_get_result(mgr, p, m)),
    "s_handler":     ("Stream + handler",          lambda mgr, p, m: asyncio.run(bench_s_handler(mgr, p, m))),
}


def main():
    parser = argparse.ArgumentParser(description="Raw CB benchmark (2x2 matrix)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=list(METHODS.keys()), choices=list(METHODS.keys()))
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Batch: {args.batch} | Prompt: ~{args.prompt_tokens} tok | Gen: {args.max_new_tokens} tok")
    print(f"Warmup: {args.warmup} | Runs: {args.runs} | Methods: {args.methods}")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_3",
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_prompts = make_prompts(tokenizer, max(args.batch), args.prompt_tokens)

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    cb_config = ContinuousBatchingConfig()

    # Header
    col_w = 20
    header = f"{'N':>6}"
    for m in args.methods:
        label = METHODS[m][0]
        header += f" | {label:>{col_w}}"
    print(f"\n{header}")
    print("-" * len(header))
    sys.stdout.flush()

    # Per-batch-size context: each N gets fresh CUDA graph capture
    for N in args.batch:
        prompts = all_prompts[:N]

        row = f"{N:>6}"
        for method_key in args.methods:
            _, fn = METHODS[method_key]
            # Fresh CB context per method — each gets its own CUDA graph cache
            with model.continuous_batching_context_manager(
                generation_config=gen_config, continuous_batching_config=cb_config, block=True, timeout=5,
            ) as mgr:
                # Warmup with the same method being tested
                warmup_prompts = prompts[:min(200, N)]
                for _ in range(args.warmup):
                    fn(mgr, warmup_prompts, args.max_new_tokens)
                # Measured runs
                best = 0
                for _ in range(args.runs):
                    tokens, dt = fn(mgr, prompts, args.max_new_tokens)
                    best = max(best, tokens / dt if dt > 0 else 0)
            row += f" | {best:>{col_w - 4}.0f} t/s"
            print(row, flush=True)

    # Quality check
    print("\n--- Quality check ---")
    with model.continuous_batching_context_manager(
        generation_config=gen_config, continuous_batching_config=cb_config, block=True, timeout=5,
    ) as mgr:
        async def check():
            loop = asyncio.get_running_loop()
            for i in range(3):
                rid = f"qc_{i}"
                future = loop.create_future()
                def _on_qc(output, fut=future):
                    if not fut.done():
                        fut.set_result(output)
                mgr.register_result_handler(rid, _on_qc)
                mgr.add_request(all_prompts[i], request_id=rid, max_new_tokens=args.max_new_tokens, streaming=False)
                r = await future
                text = tokenizer.decode(r.generated_tokens, skip_special_tokens=True)[:80]
                print(f"  {r.request_id}: {len(r.generated_tokens)} tokens | {text}")
        asyncio.run(check())


if __name__ == "__main__":
    main()
