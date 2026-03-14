#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generation Scheduler Benchmark Script
======================================

Measures the overhead of the generation scheduler across different modes
and configurations, comparing against the baseline (no scheduler).

Usage:
    # Quick benchmark with tiny model (CPU)
    python scripts/benchmark_scheduler.py

    # Benchmark with a specific model (GPU)
    python scripts/benchmark_scheduler.py --model gpt2 --device cuda

    # Full benchmark with multiple configurations
    python scripts/benchmark_scheduler.py --model gpt2 --device cuda --full

    # Custom parameters
    python scripts/benchmark_scheduler.py --model gpt2 --max-new-tokens 256 --batch-size 4 --warmup 3 --runs 10

Output:
    A table showing latency (ms), tokens/sec, and overhead (%) for each configuration.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync_device(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_generate(
    model,
    input_ids,
    max_new_tokens: int,
    do_sample: bool,
    scheduler=None,
    warmup: int = 2,
    runs: int = 5,
    device: torch.device = torch.device("cpu"),
):
    """Run a generation benchmark and return timing statistics."""
    from transformers.generation.generation_scheduler import GenerationScheduler

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if scheduler is not None:
        kwargs["scheduler"] = scheduler

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, **kwargs)
        _sync_device(device)

    # Timed runs
    latencies = []
    output_lengths = []
    for _ in range(runs):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _sync_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, **kwargs)
        _sync_device(device)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        output_lengths.append(output.shape[-1] - input_ids.shape[-1])

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_tokens = sum(output_lengths) / len(output_lengths)
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0

    return {
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "avg_tokens": avg_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def _build_html_report(
    model_name: str,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    warmup: int,
    runs: int,
    results,
) -> str:
    """Build an HTML report from benchmark results."""
    import html

    title = "Generation Scheduler Benchmark Report"
    rows = []
    for r in results:
        rows.append(
            (
                html.escape(r["name"]),
                f"{r['avg_latency_ms']:.1f}",
                f"{r['min_latency_ms']:.1f}",
                f"{r['max_latency_ms']:.1f}",
                f"{r['tokens_per_sec']:.1f}",
                f"{r['overhead_pct']:+.1f}%",
            )
        )

    table_rows = "\n".join(
        f"      <tr><td>{name}</td><td>{avg}</td><td>{min_}</td><td>{max_}</td><td>{tok}</td><td>{over}</td></tr>"
        for (name, avg, min_, max_, tok, over) in rows
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background-color: #fafafa;
    }}
    h1 {{
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .meta {{
      margin-bottom: 16px;
      color: #555;
      font-size: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #fff;
    }}
    th, td {{
      padding: 8px 12px;
      border: 1px solid #ddd;
      text-align: right;
    }}
    th:first-child,
    td:first-child {{
      text-align: left;
    }}
    thead {{
      background: #f0f0f0;
    }}
    tbody tr:nth-child(even) {{
      background: #f9f9f9;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    <div><strong>Model:</strong> {html.escape(model_name)}</div>
    <div><strong>Device:</strong> {html.escape(str(device))}</div>
    <div><strong>Max new tokens:</strong> {max_new_tokens}</div>
    <div><strong>Batch size:</strong> {batch_size}</div>
    <div><strong>Sampling:</strong> {do_sample}</div>
    <div><strong>Warmup runs:</strong> {warmup}</div>
    <div><strong>Timed runs:</strong> {runs}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Configuration</th>
        <th>Avg (ms)</th>
        <th>Min (ms)</th>
        <th>Max (ms)</th>
        <th>Tok/s</th>
        <th>Overhead</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</body>
</html>
"""
    return html_doc


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Generation Scheduler overhead")
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-gpt2",
                        help="Model name or path (default: tiny-random-gpt2)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device to run on")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--full", action="store_true", help="Run full benchmark with all configurations")
    parser.add_argument(
        "--html-report",
        type=str,
        metavar="PATH",
        help="Optional path to save an HTML summary report",
    )
    args = parser.parse_args()

    device = _get_device(args.device)
    print(f"\n{'='*72}")
    print(f"  Generation Scheduler Benchmark")
    print(f"{'='*72}")
    print(f"  Model:          {args.model}")
    print(f"  Device:         {device}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Sampling:       {args.do_sample}")
    print(f"  Warmup runs:    {args.warmup}")
    print(f"  Timed runs:     {args.runs}")
    print(f"{'='*72}\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog"
    if args.batch_size > 1:
        prompts = [prompt] * args.batch_size
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print(f"Input shape: {input_ids.shape}")
    print()

    from transformers.generation.generation_scheduler import GenerationScheduler, SchedulerCallback
    from transformers.generation.scheduler_callbacks import (
        EntropyMonitorCallback,
        GenerationLoggerCallback,
        StepBudgetCallback,
    )

    # Define configurations to benchmark
    configs = []

    # 1. Baseline (no scheduler)
    configs.append(("Baseline (no scheduler)", None))

    # 2. NONE mode
    configs.append(("mode=none", GenerationScheduler(mode="none")))

    # 3. FORCE mode (no callbacks)
    configs.append(("mode=force (0 callbacks)", GenerationScheduler(mode="force")))

    # 4. FORCE mode (1 lightweight callback)
    s = GenerationScheduler(mode="force")
    s.register_callback(SchedulerCallback())  # No-op callback
    configs.append(("mode=force (1 noop cb)", s))

    if args.full:
        # 5. FORCE mode (3 callbacks)
        s = GenerationScheduler(mode="force")
        s.register_callback(SchedulerCallback())
        s.register_callback(SchedulerCallback())
        s.register_callback(SchedulerCallback())
        configs.append(("mode=force (3 noop cbs)", s))

        # 6. FORCE mode with EntropyMonitor
        s = GenerationScheduler(mode="force")
        s.register_callback(EntropyMonitorCallback(entropy_threshold=100.0, action="log"))
        configs.append(("mode=force + entropy", s))

        # 7. FORCE mode with GenerationLogger
        s = GenerationScheduler(mode="force")
        s.register_callback(GenerationLoggerCallback(log_tokens=True, log_phases=True))
        configs.append(("mode=force + logger", s))

        # 8. FORCE mode with check_interval
        s = GenerationScheduler(mode="force")
        s.context.check_interval = 10
        configs.append(("mode=force + chk_int=10", s))

        # 9. INTERNAL mode (no parser)
        configs.append(("mode=internal (no parser)", GenerationScheduler(mode="internal")))

    # Run benchmarks
    results = []
    baseline_latency = None

    for name, scheduler in configs:
        print(f"  Benchmarking: {name:<30} ", end="", flush=True)
        result = benchmark_generate(
            model, input_ids, args.max_new_tokens, args.do_sample,
            scheduler=scheduler, warmup=args.warmup, runs=args.runs, device=device,
        )
        if baseline_latency is None:
            baseline_latency = result["avg_latency_ms"]
            overhead_pct = 0.0
        else:
            overhead_pct = ((result["avg_latency_ms"] - baseline_latency) / baseline_latency) * 100

        result["name"] = name
        result["overhead_pct"] = overhead_pct
        results.append(result)

        print(f"avg={result['avg_latency_ms']:8.1f}ms  "
              f"tok/s={result['tokens_per_sec']:8.1f}  "
              f"overhead={overhead_pct:+.1f}%")

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  Summary")
    print(f"{'='*72}")
    print(f"  {'Configuration':<30} {'Avg(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'Tok/s':>8} {'Overhead':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['name']:<30} {r['avg_latency_ms']:8.1f} {r['min_latency_ms']:8.1f} "
              f"{r['max_latency_ms']:8.1f} {r['tokens_per_sec']:8.1f} {r['overhead_pct']:+7.1f}%")
    print(f"{'='*72}\n")

    # Optional HTML report
    if args.html_report:
        report_path = Path(args.html_report)
        html_content = _build_html_report(
            args.model,
            device,
            args.max_new_tokens,
            args.batch_size,
            args.do_sample,
            args.warmup,
            args.runs,
            results,
        )
        report_path.write_text(html_content, encoding="utf-8")
        print(f"HTML report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
