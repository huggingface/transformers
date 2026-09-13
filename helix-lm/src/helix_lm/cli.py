# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Command line entry point: ``helix info | demo | bench | recall``."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

from . import __author__, __version__
from .cache import HelixCache
from .config import HelixConfig
from .model import HelixBraid, HelixForCausalLM

SMALL = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "block_size": 64,
    "local_blocks": 3,
    "num_window_scales": 2,
    "index_layer_stride": 2,
    "index_topk": 8,
    "num_recurrent_heads": 2,
    "recurrent_head_dim": 64,
    "recurrent_value_head_dim": 64,
}


def _banner() -> str:
    return f"HELIX {__version__} — Hierarchical Episodic Linear IndeX — made by {__author__}"


def cmd_info(args: argparse.Namespace) -> None:
    config = HelixConfig(**SMALL)
    model = HelixForCausalLM(config)
    print(_banner())
    print("\nEvery block braids three sequence mixers over the same residual stream:\n")
    print(
        "  L  local      exact softmax attention over the last "
        f"{config.local_span} tokens, per-head windows {config.local_window_sizes}"
    )
    print(
        f"  R  recurrent  gated delta rule, {config.num_recurrent_heads} heads x "
        f"{config.recurrent_head_dim}x{config.recurrent_value_head_dim} matrix state"
    )
    print(
        f"  I  index      beam descent over a {config.index_branching}-ary landmark tree, "
        f"picks {config.index_topk} blocks of {config.block_size} tokens"
    )
    print(f"\nlayer schedule : {config.layer_types}")
    print(f"parameters     : {model.num_parameters():,}")
    keys_per_query = (config.local_blocks + 1 + config.index_topk) * config.block_size
    print(f"keys per query : {keys_per_query} on an indexed layer — constant, whatever the context length")
    print("\nRun `helix demo` to see it generate, `helix bench` for the scaling numbers.")


def cmd_demo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    config = HelixConfig(**{**SMALL, "vocab_size": args.vocab_size})
    model = HelixForCausalLM(config).eval()
    print(_banner())
    print(
        f"\nrandom-weight model, {model.num_parameters():,} parameters "
        f"(untrained, so the tokens below are noise — this shows the machinery runs)\n"
    )

    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_length))
    started = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    elapsed = time.perf_counter() - started
    print(f"prompt         : {prompt.shape[1]} tokens")
    print(f"generated      : {generated.shape[1] - prompt.shape[1]} tokens in {elapsed:.2f}s")
    print(f"first 16 ids   : {generated[0, prompt.shape[1] :][:16].tolist()}")

    # The property that makes HELIX worth using: the fixed-size decode state reproduces the full forward.
    reference = model(generated).logits
    cache = HelixCache(config)
    steps = [
        model(generated[:, i : i + 1], past_key_values=cache, use_cache=True).logits for i in range(generated.shape[1])
    ]
    delta = (torch.cat(steps, dim=1) - reference).abs().max().item()
    print(f"\nstep-by-step decoding vs one-shot forward: max difference {delta:.2e}")
    print(f"decode state after {generated.shape[1]} tokens: {cache.byte_size() / 1024:.1f} KiB")


def _count_pairs(model: HelixForCausalLM, length: int) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, HelixBraid):
            original = module._blocked_attention

            def counted(query_blocks, keys, *args, _original=original, **kwargs):
                nonlocal total
                batch, heads, blocks, block_size, _ = query_blocks.shape
                total += batch * heads * blocks * block_size * keys.shape[3]
                return _original(query_blocks, keys, *args, **kwargs)

            module._blocked_attention = counted
    with torch.no_grad():
        model(torch.zeros(1, length, dtype=torch.long))
    return total


def _slope(xs: list[int], ys: list[float]) -> float:
    lx, ly = [math.log(v) for v in xs], [math.log(v) for v in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / sum((a - mx) ** 2 for a in lx)


def cmd_bench(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    model = HelixForCausalLM(HelixConfig(**SMALL)).eval()
    print(_banner())
    print("\nAttention pairs are exact: the query/key products the forward pass actually forms, counted by")
    print("instrumenting the attention call. Flat per-token means linear in the context length.\n")
    print(f"{'tokens':>8}  {'pairs':>16}  {'per token':>10}  {'seconds':>8}  {'cache MiB':>10}")

    lengths, times = args.lengths, []
    for length in lengths:
        pairs = _count_pairs(model, length)
        ids = torch.zeros(1, length, dtype=torch.long)
        with torch.no_grad():
            model(ids)
            started = time.perf_counter()
            cache = model(ids, use_cache=True).past_key_values
            elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(
            f"{length:>8}  {pairs:>16,}  {pairs / length:>10,.0f}  {elapsed:>8.2f}  "
            f"{cache.byte_size() / 1024**2:>10.1f}",
            flush=True,
        )

    if len(lengths) > 1:
        print(
            f"\nwall clock fits N^{_slope(lengths, times):.2f}   "
            "(full attention is N^2.00 in pairs; this is a reference implementation with no fused kernel)"
        )


def cmd_recall(args: argparse.Namespace) -> None:
    """Train two tiny models on associative recall: with the index strand, and without it."""
    from .recall import run_recall

    print(_banner())
    run_recall(steps=args.steps, seq_len=args.seq_len, pairs=args.pairs, batch_size=args.batch_size, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helix", description=_banner())
    parser.add_argument("--version", action="version", version=f"helix-lm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="what the architecture is and what it costs").set_defaults(func=cmd_info)

    demo = sub.add_parser("demo", help="build an untrained model, generate, verify the decode state")
    demo.add_argument("--prompt-length", type=int, default=64)
    demo.add_argument("--max-new-tokens", type=int, default=32)
    demo.add_argument("--temperature", type=float, default=0.0)
    demo.add_argument("--vocab-size", type=int, default=512)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    bench = sub.add_parser("bench", help="cost against context length")
    bench.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    bench.set_defaults(func=cmd_bench)

    recall = sub.add_parser("recall", help="train on associative recall, with and without the index strand")
    recall.add_argument("--steps", type=int, default=600)
    recall.add_argument("--seq-len", type=int, default=256)
    recall.add_argument("--pairs", type=int, default=16)
    recall.add_argument("--batch-size", type=int, default=16)
    recall.add_argument("--seed", type=int, default=0)
    recall.set_defaults(func=cmd_recall)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:
        # `helix info | head` closes the pipe early; exit quietly instead of dumping a traceback.
        try:
            sys.stdout.close()
        finally:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
