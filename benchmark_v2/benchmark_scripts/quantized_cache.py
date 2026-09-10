"""
Quantized cache benchmark.

Compares the backends of `QuantizedCache` against the default bf16 `DynamicCache`, on the size of the KV cache,
the peak device memory, the latency of `generate`, and how much the generation drifts from the bf16 one.
"""

import argparse
import gc
import statistics
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


# The additional options of the group-wise integer backends, which do not apply to `fp8`
BACKEND_CACHE_CONFIGS = {
    "quanto": {"backend": "quanto", "nbits": 4, "axis_key": 0, "axis_value": 0},
    "hqq": {"backend": "hqq", "nbits": 4, "axis_key": 1, "axis_value": 1},
    "fp8": {"backend": "fp8"},
}

PROMPT = "The French Revolution was a period of political and societal change in France that began in 1789. "


def cache_memory(cache) -> int:
    """Number of bytes held by the key and value states of `cache`, whether they are quantized or not."""
    states = ("keys", "values", "_quantized_keys", "_quantized_values")
    tensors = (getattr(layer, name, None) for layer in cache.layers for name in states)
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def run(model, inputs, generation_kwargs, warmup: int, iterations: int) -> dict:
    """Time `generate` and collect the memory used by the cache it returns."""
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()

    latencies = []
    for i in range(warmup + iterations):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        outputs = model.generate(**inputs, **generation_kwargs)
        torch.accelerator.synchronize()
        if i >= warmup:
            latencies.append(time.perf_counter() - start)

    return {
        "latency": statistics.median(latencies) * 1e3,
        "cache_memory": cache_memory(outputs.past_key_values) / 2**20,
        "peak_memory": torch.accelerator.max_memory_allocated() / 2**20,
        "sequences": outputs.sequences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--backends", type=str, nargs="+", default=list(BACKEND_CACHE_CONFIGS))
    parser.add_argument("--sequence-length", "-s", type=int, default=1024)
    parser.add_argument("--num-tokens-to-generate", "-n", type=int, default=256)
    parser.add_argument("--warmup", "-w", type=int, default=1)
    parser.add_argument("--iterations", "-i", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="auto")

    prompt = PROMPT * (args.sequence_length // len(tokenizer(PROMPT).input_ids) + 1)
    inputs = tokenizer(
        [prompt], return_tensors="pt", max_length=args.sequence_length, truncation=True, return_attention_mask=True
    ).to(model.device)
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": args.num_tokens_to_generate,
        "return_dict_in_generate": True,
        "disable_compile": True,
    }

    print(f"\n{args.model_id}, {args.sequence_length} prompt tokens, {args.num_tokens_to_generate} generated tokens\n")
    header = f"{'cache':>10} | {'kv cache':>10} | {'peak memory':>12} | {'latency':>10} | {'matching tokens':>16}"
    print(header + "\n" + "-" * len(header))

    # The bf16 `DynamicCache` is the baseline that the quantized backends are compared against
    baseline = run(model, inputs, generation_kwargs, args.warmup, args.iterations)
    results = {"bf16": baseline}
    for backend in args.backends:
        cache_kwargs = {"cache_implementation": "quantized", "cache_config": BACKEND_CACHE_CONFIGS[backend]}
        results[backend] = run(model, inputs, generation_kwargs | cache_kwargs, args.warmup, args.iterations)

    for name, result in results.items():
        # Quantizing the cache changes the numerics, so the greedy generations eventually diverge from the bf16 one
        matching_tokens = (result["sequences"] == baseline["sequences"]).int().cumprod(dim=-1).sum()
        print(
            f"{name:>10} | {result['cache_memory']:7.1f} MiB | {result['peak_memory']:9.1f} MiB "
            f"| {result['latency']:7.1f} ms | {matching_tokens - inputs.input_ids.numel():6d} "
            f"/ {args.num_tokens_to_generate:<7d}"
        )


if __name__ == "__main__":
    main()
