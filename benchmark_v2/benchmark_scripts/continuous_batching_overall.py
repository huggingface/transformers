import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from tabulate import tabulate


SCRIPT_LOCATION = (Path(__file__).parent.parent.parent / "examples/pytorch/continuous_batching.py").as_posix()
COMMON_ARGS = "--log-level WARNING --seed 0 --force-max-length".split()
ERROR_OUTPUT = {"time_seconds": "X", "num_tokens": "X", "throughput_tok_per_sec": "ERROR"}
RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results/cb_overall/"


def run_and_parse_cb_example(args: str) -> dict:
    print(f"\nBenchmarking with args: {args}")
    output = subprocess.run(
        ["python", SCRIPT_LOCATION] + args.split() + COMMON_ARGS,
        stdout=subprocess.PIPE,
    )
    output = output.stdout.decode("utf-8")
    if "generate_batch despite unexpected termination" in output:
        return {"args": args, **ERROR_OUTPUT}
    pattern = r"CB generation took: ([\d.]+) seconds for (\d+) tokens\. ([\d.]+)tok/s"
    match = re.search(pattern, output)
    if match is not None:
        return {
            "args": args,
            "time_seconds": float(match.group(1)),
            "num_tokens": int(match.group(2)),
            "throughput_tok_per_sec": float(match.group(3)),
        }
    else:
        return {"args": args, **ERROR_OUTPUT}


def get_most_recent_file(prefix: str, exclude: Path | None = None) -> Path | None:
    """Find the most recent results file in RESULTS_DIR matching the given prefix, optionally excluding one."""
    candidates = sorted(RESULTS_DIR.glob(f"{prefix}__*.json"))
    if exclude:
        candidates = [c for c in candidates if c != exclude]
    return candidates[-1] if candidates else None


def build_comparison_table(results: list[dict], baseline_results: list[dict], baseline_label: str) -> list[dict]:
    """Build a table comparing current results against baseline results."""
    baseline_by_args = {r["args"]: r for r in baseline_results}
    comparison = [
        {
            "args": "Arguments",
            "baseline_tok_per_sec": f"{baseline_label} (tok/s)",
            "current_tok_per_sec": "Current (tok/s)",
            "diff_percent": "Diff (%)",
        }
    ]
    for result in results:
        baseline = baseline_by_args.get(result["args"])
        baseline_tp = baseline["throughput_tok_per_sec"] if baseline else None
        current_tp = result["throughput_tok_per_sec"]
        if isinstance(baseline_tp, (int, float)) and isinstance(current_tp, (int, float)):
            diff = (current_tp - baseline_tp) / baseline_tp * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"
        comparison.append(
            {
                "args": result["args"],
                "baseline_tok_per_sec": baseline_tp if baseline_tp is not None else "N/A",
                "current_tok_per_sec": current_tp,
                "diff_percent": diff_str,
            }
        )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", action="store_true", help="Save results as the main baseline to compare against.")
    args = parser.parse_args()

    results = [
        {
            "args": "Arguments",
            "time_seconds": "Duration (s)",
            "num_tokens": "Generated tokens",
            "throughput_tok_per_sec": "Throughput (tok/s)",
        }
    ]

    # Benchmark with low number of samples
    results.append(run_and_parse_cb_example("--samples 10"))
    results.append(run_and_parse_cb_example("--samples 20 --num-blocks 20"))  # and low number of blocks
    results.append(run_and_parse_cb_example("--samples 50"))

    # Benchmark with compile: default, flash attention 2 and sdpa
    results.append(run_and_parse_cb_example("--samples 100"))
    results.append(run_and_parse_cb_example("--samples 100 --attn flash_attention_2"))
    results.append(run_and_parse_cb_example("--samples 100 --attn sdpa"))

    # Benchmark with high number of samples and synchronous batching
    results.append(run_and_parse_cb_example("--samples 500 --no-use-async"))
    # Benchmark with high number of samples and asynchronous batching
    results.append(run_and_parse_cb_example("--samples 500 --use-async"))

    # Benchmark with low number of samples, asynchronous batching and decdode fast path
    results.append(run_and_parse_cb_example("--samples 32 --max-new-tokens 2048 --use-async"))
    # Benchmark with low number of samples, asynchronous batching and decdode fast path
    results.append(run_and_parse_cb_example("--samples 32 --max-new-tokens 2048 --use-async --block-table 32"))

    # Benchmark with prefix sharing and compile (best performance, but not reproducible due to compilation)
    results.append(run_and_parse_cb_example("--samples 500 --add-prefix --compile"))

    # Benchmark with parallel decoding
    results.append(run_and_parse_cb_example("--samples 50 --num-return-sequences 8 --do-sample"))
    results.append(run_and_parse_cb_example("--samples 100 --num-return-sequences 4 --do-sample"))

    # Print results
    print()
    print(tabulate(results, tablefmt="github"))

    # The header row is results[0], data rows are results[1:]
    data_results = results[1:]

    # Always save results to a new timestamped file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "main" if args.main else "run"
    results_file = RESULTS_DIR / f"{prefix}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(data_results, indent=2))
    print(f"\nResults saved to {results_file}")

    # Compare against baseline
    if args.main:
        # Compare against the previous main baseline (the one that was most recent before this new file)
        baseline_file = get_most_recent_file("main", exclude=results_file)
        baseline_label = "Previous main"
    else:
        # Compare against the most recent main baseline
        baseline_file = get_most_recent_file("main")
        baseline_label = "Main"

    if baseline_file:
        baseline_results = json.loads(baseline_file.read_text())
        comparison = build_comparison_table(data_results, baseline_results, baseline_label)
        print(f"\nComparing against: {baseline_file.name}")
        print(tabulate(comparison, tablefmt="github"))
    else:
        print("\nNo baseline results found for comparison.")
