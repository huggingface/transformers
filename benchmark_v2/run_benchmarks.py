#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Top-level benchmarking script that automatically discovers and runs all benchmarks
in the ./benches directory, organizing outputs into model-specific subfolders.
"""

import argparse
import logging
import random
import sys
import uuid

from framework.benchmark_config import BenchmarkConfig, generate_all_configs
from framework.benchmark_runner import BenchmarkRunner


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output dir for benchmark results")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--model-id", type=str, help="Specific model ID to benchmark (if supported by benchmarks)")

    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations")

    parser.add_argument("--batch-size", "-b", type=int, nargs="+", help="Batch size")
    parser.add_argument("--sequence-length", "-s", type=int, nargs="+", help="Sequence length")
    parser.add_argument("--num-tokens-to-generate", "-n", type=int, nargs="+", help="Number of tokens to generate")

    parser.add_argument("--num-tokens-to-profile", "-p", type=int, default=0, help="Number of tokens to profile")

    parser.add_argument("--commit-id", type=str, help="Git commit ID (if not provided, will auto-detect from git)")
    args = parser.parse_args()

    # Setup logging
    benchmark_run_uuid = str(uuid.uuid4())[:8]
    numeric_level = getattr(logging, args.log_level.upper())

    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=numeric_level, format="[%(levelname)s - %(asctime)s] %(name)s: %(message)s", handlers=handlers
    )

    logger = logging.getLogger("benchmark_v2")
    logger.info("Starting benchmark discovery and execution")
    logger.info(f"Benchmark run UUID: {benchmark_run_uuid}")
    logger.info(f"Output directory: {args.output_dir}")

    # Error out if one of the arguments is not provided
    if len(args.batch_size) * len(args.sequence_length) * len(args.num_tokens_to_generate) == 0:
        raise ValueError(
            "At least one of the arguments --batch-size, --sequence-length, or --num-tokens-to-generate is required"
        )

    # If there is only one (batch_size, sequence_length, num_tokens_to_generate), we benchmark across configs
    elif len(args.batch_size) * len(args.sequence_length) * len(args.num_tokens_to_generate) == 1:
        benchmark_configs = generate_all_configs(
            warmup_iterations=args.warmup,
            measurement_iterations=args.iterations,
            batch_size=args.batch_size[0],
            sequence_length=args.sequence_length[0],
            num_tokens_to_generate=args.num_tokens_to_generate[0],
        )
        random.shuffle(benchmark_configs)

    # Otherwise, we benchmark across all combinations of dimensions
    else:
        kwargs = {
            "warmup_iterations": args.warmup,
            "measurement_iterations": args.iterations,
            "gpu_monitoring": False,
            "batch_size": args.batch_size[0],
            "sequence_length": args.sequence_length[0],
            "num_tokens_to_generate": args.num_tokens_to_generate[0],
            "attn_implementation": "flex_attention",
            "sdpa_backend": None,
            "compile_mode": "default",
            "kernelize": False,
        }
        benchmark_configs = []
        for num_tokens_to_generate in args.num_tokens_to_generate:
            for sequence_length in args.sequence_length:
                for batch_size in args.batch_size:
                    kwargs["batch_size"] = batch_size
                    kwargs["sequence_length"] = sequence_length
                    kwargs["num_tokens_to_generate"] = num_tokens_to_generate
                    benchmark_configs.append(BenchmarkConfig(**kwargs))

    runner = BenchmarkRunner(logger, args.output_dir, args.commit_id)
    results = runner.run_benchmarks(
        args.model_id,
        benchmark_configs[:3],
        args.num_tokens_to_profile,
        pretty_print_summary=True,
    )
    # runner.save_results(args.model_id, results)
