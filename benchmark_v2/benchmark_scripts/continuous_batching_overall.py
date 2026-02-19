import re
import subprocess
from pathlib import Path

from tabulate import tabulate


SCRIPT_LOCATION = (Path(__file__).parent.parent.parent / "examples/pytorch/continuous_batching.py").as_posix()
COMMON_ARGS = "--log-level WARNING --seed 0".split()
ERROR_OUTPUT = {"time_seconds": "X", "num_tokens": "X", "throughput_tok_per_sec": "ERROR"}


def run_and_parse_cb_example(args: str) -> dict:
    print(f"Benchmarking with args: {args}")
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


if __name__ == "__main__":
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

    # Benchmark with high number of samples
    results.append(run_and_parse_cb_example("--samples 500"))

    # Benchmark with prefix sharing and compile (best performance, but not reproducible due to compilation)
    results.append(run_and_parse_cb_example("--samples 500 --add-prefix --compile"))

    # Benchmark with parallel decoding
    results.append(run_and_parse_cb_example("--samples 50 --num-return-sequences 8 --do-sample"))
    results.append(run_and_parse_cb_example("--samples 100 --num-return-sequences 4 --do-sample"))

    print()
    print(tabulate(results, tablefmt="github"))
