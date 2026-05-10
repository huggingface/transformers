#!/usr/bin/env python3
"""Sweep max_len to find when freeze_after_warmup vs clear_per_iter is better.

Runs always/freeze/clear for each max_len. Shows overhead ratio and crossover.

Usage:
    PYTHONPATH=~/pytorch python3 tmp_hf_bench_sweep.py
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

import psutil
import torch
from torch.optim import AdamW


assert torch.backends.mps.is_available(), "MPS not available"

DEVICE  = torch.device("mps")
ITERS   = 150
WARMUP  = 5
BS      = 8
MAX_LENS = [64, 128, 256, 512, 1024]


def _cur_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _make_workload(max_len, quiet=False):
    from datasets import load_dataset

    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding

    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE).train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    texts     = [t for t in dataset["text"] if len(t.strip()) > 20][:10000]
    tok_cache = [tokenizer(t, truncation=True, max_length=max_len,
                           return_attention_mask=True) for t in texts]
    lengths   = [len(e["input_ids"]) for e in tok_cache]
    n_unique  = len(set(lengths))
    std_len   = statistics.stdev(lengths)
    eff_1d    = math.sqrt(BS) * std_len * 2 * math.pi
    freeze_at = min(ITERS // 4, max(5, int(math.sqrt(eff_1d))))

    if not quiet:
        print(f"  max_len={max_len}: n_unique={n_unique}  std_len={std_len:.1f}  "
              f"eff_1d={eff_1d:.0f}  freeze_at={freeze_at}")

    def step_fn(i):
        rng   = random.Random(i)
        batch = [tok_cache[rng.randint(0, len(tok_cache) - 1)] for _ in range(BS)]
        batch = collator(batch)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        optimizer.zero_grad()
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()

    return step_fn, freeze_at, n_unique, std_len


def run_strategy(strat_name, freeze_at, max_len):
    if strat_name == "never":
        torch.mps.set_graph_cache_policy("never")
    else:
        torch.mps.set_graph_cache_policy("always")

    step_fn, _, _, _ = _make_workload(max_len, quiet=True)
    torch.mps.empty_cache()
    torch.mps.clear_graph_cache()
    torch.mps.synchronize()

    rss_pre = _cur_rss_mb()
    for i in range(WARMUP):
        step_fn(i)
    torch.mps.synchronize()

    times = []
    for i in range(ITERS):
        t0 = time.perf_counter()
        step_fn(WARMUP + i)

        if strat_name == "clear_per_iter":
            torch.mps.clear_graph_cache()
        elif strat_name == "freeze_after_warmup" and i == freeze_at:
            torch.mps.freeze_graph_cache()

        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    torch.mps.synchronize()
    rss_final = _cur_rss_mb()

    return {
        "strat":     strat_name,
        "rss_delta": rss_final - rss_pre,
        "mean_ms":   statistics.mean(times[5:]),
        "std_ms":    statistics.stdev(times[5:]) if len(times) > 6 else 0.0,
    }


def spawn(strat, freeze_at, max_len):
    result = subprocess.run(
        [sys.executable, __file__,
         "--run-one", "--strategy", strat,
         "--freeze-at", str(freeze_at),
         "--max-len", str(max_len)],
        capture_output=True, text=True, env=os.environ.copy()
    )
    if result.returncode != 0:
        print(f"    FAILED {strat}: {result.stderr[-150:]}")
        return None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    return None


def main():
    print(f"PyTorch {torch.__version__}  |  BERT/WikiText-2 freeze vs clear sweep")
    print(f"  iters={ITERS}  warmup={WARMUP}  BS={BS}")
    print()
    print(f"  {'max_len':>7}  {'n_uniq':>6}  {'frz_at':>6}  "
          f"{'always MB':>9}  {'always ms':>9}  "
          f"{'frz ms':>8}  {'frz ratio':>9}  "
          f"{'clr ms':>8}  {'clr ratio':>9}")
    print("  " + "-"*95)

    for max_len in MAX_LENS:
        _, freeze_at, n_unique, _ = _make_workload(max_len, quiet=True)

        sys.stdout.write(f"  max_len={max_len} running always...\r")
        sys.stdout.flush()
        m_always = spawn("always", freeze_at, max_len)
        if not m_always:
            continue

        sys.stdout.write(f"  max_len={max_len} running freeze...\r")
        sys.stdout.flush()
        m_freeze = spawn("freeze_after_warmup", freeze_at, max_len)

        sys.stdout.write(f"  max_len={max_len} running clear...\r ")
        sys.stdout.flush()
        m_clear = spawn("clear_per_iter", freeze_at, max_len)

        frz_ratio = f"{m_freeze['mean_ms']/m_always['mean_ms']:.2f}x" if m_freeze else "FAIL"
        clr_ratio = f"{m_clear['mean_ms']/m_always['mean_ms']:.2f}x"  if m_clear  else "FAIL"
        winner = "freeze" if (m_freeze and m_clear and
                               m_freeze["mean_ms"] < m_clear["mean_ms"]) else "clear"

        print(f"  {max_len:>7}  {n_unique:>6}  {freeze_at:>6}  "
              f"{m_always['rss_delta']:>+9.0f}  {m_always['mean_ms']:>9.1f}  "
              f"{m_freeze['mean_ms'] if m_freeze else 0:>8.1f}  {frz_ratio:>9}  "
              f"{m_clear['mean_ms'] if m_clear else 0:>8.1f}  {clr_ratio:>9}  "
              f"← {winner}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-one",   action="store_true")
    parser.add_argument("--strategy",  default="always")
    parser.add_argument("--freeze-at", type=int, default=10)
    parser.add_argument("--max-len",   type=int, default=128)
    args = parser.parse_args()

    if args.run_one:
        m = run_strategy(args.strategy, args.freeze_at, args.max_len)
        print("RESULT:" + json.dumps(m))
    else:
        main()
