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
import sys
import uuid

from framework.benchmark_config import adapt_configs, get_config_by_level
from framework.benchmark_runner import BenchmarkRunner


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for benchmark results")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--model-id", type=str, help="Specific model ID to benchmark (if supported by benchmarks)")
    parser.add_argument("--warmup", "-w", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of measurement iterations")

    parser.add_argument("--batch-size", "-b", type=int, nargs="+", help="Batch size")
    parser.add_argument("--sequence-length", "-s", type=int, nargs="+", help="Sequence length")
    parser.add_argument("--num-tokens-to-generate", "-n", type=int, nargs="+", help="Number of tokens to generate")

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Level of coverage for the benchmark. 0: only the main config, 1: a few important configs, 2: a config for"
        " each attn implementation an option, 3: cross-generate all combinations of configs, 4: cross-generate all"
        " combinations of configs w/ all compile modes",
    )
    parser.add_argument("--num-tokens-to-profile", "-p", type=int, default=0, help="Number of tokens to profile")

    parser.add_argument("--branch-name", type=str, help="Git branch name")
    parser.add_argument("--commit-id", type=str, help="Git commit ID (if not provided, will auto-detect from git)")
    parser.add_argument("--commit-message", type=str, help="Git commit message")

    parser.add_argument(
        "--no-gpu-monitoring", action="store_true", help="Disables GPU monitoring during benchmark runs"
    )

    parser.add_argument(
        "--push-result-to-dataset",
        type=str,
        default=None,
        help="Name of the dataset to push results to. If not provided, results are not pushed to the Hub.",
    )
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

    # We cannot compute ITL if we don't have at least two measurements
    if any(n <= 1 for n in args.num_tokens_to_generate):
        raise ValueError("--num_tokens_to_generate arguments should be larger than 1")

    # Error out if one of the arguments is not provided
    if len(args.batch_size) * len(args.sequence_length) * len(args.num_tokens_to_generate) == 0:
        raise ValueError(
            "At least one of the arguments --batch-size, --sequence-length, or --num-tokens-to-generate is required"
        )

    # Get the configs for the given coverage level
    configs = get_config_by_level(args.level)
    # Adapt the configs to the given arguments
    configs = adapt_configs(
        configs,
        args.warmup,
        args.iterations,
        args.batch_size,
        args.sequence_length,
        args.num_tokens_to_generate,
        not args.no_gpu_monitoring,
    )

    runner = BenchmarkRunner(logger, args.output_dir, args.branch_name, args.commit_id, args.commit_message)
    timestamp, results = runner.run_benchmarks(
        args.model_id, configs, args.num_tokens_to_profile, pretty_print_summary=True
    )

    dataset_id = args.push_result_to_dataset
    if dataset_id is not None and len(results) > 0:
        runner.push_results_to_hub(dataset_id, results, timestamp)
