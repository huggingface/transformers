import re
import subprocess
from pathlib import Path

from tabulate import tabulate


SCRIPT_LOCATION = (Path(__file__).parent.parent.parent / "examples/pytorch/continuous_batching.py").as_posix()
COMMON_ARGS = "--log-level WARNING --seed 0".split()


def run_and_parse_cb_example(args: list[str]) -> dict:
    print(f"Benchmarking with args: {args}")
    output = subprocess.check_output(
        ["python", SCRIPT_LOCATION] + args.split() + COMMON_ARGS,
        # stderr=subprocess.DEVNULL,
    )
    pattern = r"CB generation took: ([\d.]+) seconds for (\d+) tokens\. ([\d.]+)tok/s"
    match = re.search(pattern, output.decode("utf-8"))
    if match is not None:
        return {
            "args": args,
            "time_seconds": float(match.group(1)),
            "num_tokens": int(match.group(2)),
            "throughput_tok_per_sec": float(match.group(3)),
        }
    return {}


if __name__ == "__main__":
    results = [
        {
            "args": "Arguments",
            "time_seconds": "Duration (s)",
            "num_tokens": "Generated tokens",
            "throughput_tok_per_sec": "Throughput (tok/s)",
        }
    ]

    # Benchmark with different number of samples
    results.append(run_and_parse_cb_example("--samples 10"))
    results.append(run_and_parse_cb_example("--samples 50"))
    results.append(run_and_parse_cb_example("--samples 100"))
    results.append(run_and_parse_cb_example("--samples 500"))

    # Benchmark with compile: default, flash attention 2 and sdpa
    results.append(run_and_parse_cb_example("--samples 100 --compile"))
    results.append(run_and_parse_cb_example("--samples 100 --compile --attn flash_attention_2"))
    results.append(run_and_parse_cb_example("--samples 100 --compile --attn sdpa"))

    # Benchmark with parallel decoding
    results.append(run_and_parse_cb_example("--samples 50 --compile --num-return-sequences 8 --do-sample"))
    results.append(run_and_parse_cb_example("--samples 100 --compile --num-return-sequences 4 --do-sample"))

    # Benchmark with prefix sharing
    results.append(run_and_parse_cb_example("--samples 500 --add-prefix --compile"))

    print()
    print(tabulate(results, tablefmt="github"))
