#!/usr/bin/env python3
"""
Benchmark the impact of attention mask skip logic on decoder-only inference throughput.

Measures tokens/sec and peak memory with and without the causal mask skip condition,
at varying batch sizes and sequence lengths.

Usage:
    python tests/utils/benchmark_attention_mask_skip.py              # CPU
    python tests/utils/benchmark_attention_mask_skip.py --device cuda  # GPU
    python tests/utils/benchmark_attention_mask_skip.py --model "meta-llama/Llama-3.2-1B"
"""

import argparse
import time
from contextlib import contextmanager

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, tokenizer


@contextmanager
def force_mask(sdpa_mask_skip):
    """Context manager to force mask creation by patching the skip function."""
    import transformers.masking_utils as mu

    original_fn = mu._ignore_causal_mask_sdpa

    def patched_fn(padding_mask, query_length, kv_length, kv_offset, local_attention_size=None):
        if sdpa_mask_skip:
            return original_fn(padding_mask, query_length, kv_length, kv_offset, local_attention_size)
        return False

    mu._ignore_causal_mask_sdpa = patched_fn
    try:
        yield
    finally:
        mu._ignore_causal_mask_sdpa = original_fn


@torch.no_grad()
def benchmark_throughput(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_new_tokens: int,
    device: str,
    skip_enabled: bool,
    num_runs: int = 5,
) -> dict:
    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_length // 10)
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length).to(device)

    with force_mask(skip_enabled):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        model.generate(
            **inputs,
            max_new_tokens=num_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

    total_tokens = batch_size * num_new_tokens
    throughput = total_tokens / elapsed

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with force_mask(skip_enabled):
            _ = model.generate(
                **inputs, max_new_tokens=num_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_mem = 0.0

    return {
        "skip_enabled": skip_enabled,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "num_new_tokens": num_new_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_tokens_per_sec": round(throughput, 1),
        "peak_memory_gib": round(peak_mem, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--num-new-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model, args.device)

    print(
        f"{'Skip':>6} | {'Batch':>5} | {'SeqLen':>6} | {'NewTok':>6} | {'Time(s)':>8} | {'Tok/s':>8} | {'Mem(GiB)':>8}"
    )
    print("-" * 70)

    results = []
    for batch_size in args.batch_sizes:
        for seq_length in args.seq_lengths:
            if batch_size * seq_length > 4096 and args.device == "cpu":
                print(f"{'SKIP':>6} | {batch_size:>5} | {seq_length:>6} | SKIPPED (too large for CPU)")
                continue

            for skip_enabled in [False, True]:
                try:
                    result = benchmark_throughput(
                        model=model,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        seq_length=seq_length,
                        num_new_tokens=args.num_new_tokens,
                        device=args.device,
                        skip_enabled=skip_enabled,
                        num_runs=args.num_runs,
                    )
                    results.append(result)
                    label = "SKIP" if skip_enabled else "NO-SKIP"
                    print(
                        f"{label:>6} | {result['batch_size']:>5} | {result['seq_length']:>6} | "
                        f"{result['num_new_tokens']:>6} | {result['elapsed_seconds']:>8.3f} | "
                        f"{result['throughput_tokens_per_sec']:>8.1f} | {result['peak_memory_gib']:>8.2f}"
                    )
                except Exception as e:
                    label = "SKIP" if skip_enabled else "NO-SKIP"
                    print(f"{label:>6} | {batch_size:>5} | {seq_length:>6} | ERROR: {e}")

    if len(results) >= 2:
        print("\n--- Speedup Summary ---")
        print(f"{'Batch':>5} | {'SeqLen':>6} | {'NoSkip(t/s)':>10} | {'Skip(t/s)':>10} | {'Speedup':>8}")
        print("-" * 50)
        for batch_size in args.batch_sizes:
            for seq_length in args.seq_lengths:
                no_skip = [
                    r
                    for r in results
                    if not r["skip_enabled"] and r["batch_size"] == batch_size and r["seq_length"] == seq_length
                ]
                skip = [
                    r
                    for r in results
                    if r["skip_enabled"] and r["batch_size"] == batch_size and r["seq_length"] == seq_length
                ]
                if no_skip and skip:
                    speedup = skip[0]["throughput_tokens_per_sec"] / no_skip[0]["throughput_tokens_per_sec"]
                    print(
                        f"{batch_size:>5} | {seq_length:>6} | {no_skip[0]['throughput_tokens_per_sec']:>10.1f} | "
                        f"{skip[0]['throughput_tokens_per_sec']:>10.1f} | {speedup:>7.2f}x"
                    )


if __name__ == "__main__":
    main()
