#!/usr/bin/env python3
"""Benchmark: MPS graph cache policy impact on BERT fine-tuning (WikiText-2).

With DataCollatorWithPadding each batch is padded to max(lengths_in_batch).
The shape key is a single sequence-length dimension bounded by max_length.
Unlike GNN batching, shape space is 1-D and finite -- but with long-context
training and diverse document lengths, hundreds of unique shapes accumulate.

Each strategy runs in an isolated subprocess for clean RSS measurement
(macOS ru_maxrss is monotonic; psutil current RSS requires a fresh process).

Usage:
    python3 benchmark/mps_graph_cache_bert.py

Requires: torch >= 2.13 (pytorch/pytorch#182648), transformers, datasets, psutil, MPS.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import psutil
import torch
from torch.optim import AdamW

assert torch.backends.mps.is_available(), "MPS not available"
assert hasattr(torch.mps, 'clear_graph_cache'), \
    "clear_graph_cache requires PyTorch >= 2.13 (pytorch/pytorch#182648)"

DEVICE   = torch.device("mps")
ITERS    = 200
WARMUP   = 5
MAX_LEN  = 128
BS       = 8


def _cur_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _make_workload(quiet=False):
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              DataCollatorWithPadding)

    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE).train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    texts = [t for t in dataset["text"] if len(t.strip()) > 20][:10000]

    def tokenize(text):
        return tokenizer(text, truncation=True, max_length=MAX_LEN,
                         return_attention_mask=True)

    tok_cache = [tokenize(t) for t in texts]
    lengths   = [len(e["input_ids"]) for e in tok_cache]
    n_unique  = len(set(lengths))

    # warmup formula for 1-D shape space
    std_len   = statistics.stdev(lengths)
    eff_1d    = math.sqrt(BS) * std_len * 2 * math.pi
    freeze_at = min(ITERS // 4, max(5, int(math.sqrt(eff_1d))))

    if not quiet:
        print(f"  WikiText-2: {len(texts)} docs  max_len={MAX_LEN}  "
              f"std_len={std_len:.1f}  n_unique_lengths={n_unique}  "
              f"eff_1d={eff_1d:.0f}  freeze_at={freeze_at}")

    import random
    def step_fn(i):
        rng   = random.Random(i)
        batch = [tok_cache[rng.randint(0, len(tok_cache) - 1)] for _ in range(BS)]
        batch = collator(batch)
        # simple MLM: use input_ids as labels (not masking for bench simplicity)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        optimizer.zero_grad()
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()

    return step_fn, freeze_at, n_unique


# -- subprocess runner --------------------------------------------------------

def run_strategy(strat_name, freeze_at):
    if strat_name == "never":
        torch.mps.set_graph_cache_policy("never")
    else:
        torch.mps.set_graph_cache_policy("always")

    step_fn, _, _ = _make_workload(quiet=True)
    torch.mps.empty_cache()
    torch.mps.clear_graph_cache()
    torch.mps.synchronize()

    rss_pre = _cur_rss_mb()

    for i in range(WARMUP):
        step_fn(i)
    torch.mps.synchronize()

    rss_post_warmup = _cur_rss_mb()

    times = []
    drvs  = []
    iter_peak_rss = rss_pre
    rss_at_freeze = None

    for i in range(ITERS):
        t0 = time.perf_counter()
        step_fn(WARMUP + i)

        if strat_name == "clear_per_iter":
            iter_peak_rss = max(iter_peak_rss, _cur_rss_mb())
            torch.mps.clear_graph_cache()
        elif strat_name == "freeze_after_warmup" and i == freeze_at:
            rss_at_freeze = _cur_rss_mb()
            torch.mps.freeze_graph_cache()

        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        drvs.append(torch.mps.driver_allocated_memory())

    torch.mps.synchronize()
    rss_final = _cur_rss_mb()

    n = len(drvs)
    xm = (n - 1) / 2.0
    denom = max(1, sum((j - xm) ** 2 for j in range(n)))
    slope_kbpi = sum(
        (j - xm) * (drvs[j] - drvs[0]) for j in range(n)
    ) / denom / 1024

    return {
        "strat":        strat_name,
        "rss_delta":    rss_final - rss_pre,
        "warmup_swell": rss_post_warmup - rss_pre,
        "freeze_swell": (rss_at_freeze - rss_pre) if rss_at_freeze is not None else None,
        "iter_peak":    (iter_peak_rss - rss_pre) if strat_name == "clear_per_iter" else None,
        "mean_ms":      statistics.mean(times),
        "std_ms":       statistics.stdev(times) if len(times) > 1 else 0.0,
        "slope_kbpi":   slope_kbpi,
    }


# -- orchestrator -------------------------------------------------------------

def main_benchmark():
    _, freeze_at, n_unique = _make_workload()
    strategies = ["always", "freeze_after_warmup", "clear_per_iter", "never"]

    print()
    print(f"PyTorch {torch.__version__}  |  BERT/WikiText-2 MPS graph cache benchmark")
    print(f"  warmup={WARMUP}  freeze_at={freeze_at}  iters={ITERS}  BS={BS}  max_len={MAX_LEN}")
    print(f"  n_unique_padded_lengths={n_unique}  (subprocess-isolated, psutil RSS)")
    print()
    print(f"  {'Strategy':<22} {'ΔRSS MB':>9}  {'ms/iter':>9}  {'+-':>6}  note")
    print(f"  {'-'*22} {'-'*9}  {'-'*9}  {'-'*6}  ------")

    always_ms = None

    for strat_name in strategies:
        sys.stdout.write(f"  {strat_name:<22} running...\r")
        sys.stdout.flush()

        result = subprocess.run(
            [sys.executable, __file__, "--strategy", strat_name,
             "--freeze-at", str(freeze_at)],
            capture_output=True, text=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            print(f"  {strat_name:<22} FAILED: {result.stderr[-200:]}")
            continue

        metrics = None
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                metrics = json.loads(line[7:])
                break

        if metrics is None:
            print(f"  {strat_name:<22} no result")
            continue

        if always_ms is None:
            always_ms = metrics["mean_ms"]
            note = "(baseline, unbounded)"
        else:
            note = f"({metrics['mean_ms'] / always_ms:.2f}x, bounded)"

        print(f"  {strat_name:<22} {metrics['rss_delta']:>+9.1f}  "
              f"{metrics['mean_ms']:>9.1f}  {metrics['std_ms']:>6.1f}  {note}")

        if strat_name == "freeze_after_warmup" and metrics["freeze_swell"] is not None:
            print(f"    warmup loop: +{metrics['warmup_swell']:.1f} MB  |  "
                  f"at freeze point: +{metrics['freeze_swell']:.1f} MB")

        if strat_name == "clear_per_iter" and metrics["iter_peak"] is not None:
            print(f"    peak/iter before clear: +{metrics['iter_peak']:.1f} MB  "
                  f"(net: {metrics['rss_delta']:+.1f} MB)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None,
                        choices=["always", "freeze_after_warmup",
                                 "clear_per_iter", "never"])
    parser.add_argument("--freeze-at", type=int, default=10)
    args = parser.parse_args()

    if args.strategy:
        metrics = run_strategy(args.strategy, args.freeze_at)
        print("RESULT:" + json.dumps(metrics))
    else:
        main_benchmark()
