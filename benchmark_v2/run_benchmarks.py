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

from framework.benchmark_config import cross_generate_configs, smart_generate_configs
from framework.benchmark_runner import BenchmarkRunner


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output dir for benchmark results")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--model-id", type=str, help="Specific model ID to benchmark (if supported by benchmarks)")

    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations")

    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence-length", "-s", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-tokens-to-generate", "-n", type=int, default=128, help="Number of tokens to generate")

    parser.add_argument("--commit-id", type=str, help="Git commit ID (if not provided, will auto-detect from git)")
    args = parser.parse_args()

    # Setup logging
    benchmark_run_uuid = str(uuid.uuid4())[:8]
    numeric_level = getattr(logging, args.log_level.upper())

    handlers = [logging.StreamHandler(sys.stdout)]
    # handlers.append(logging.FileHandler(f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logging.basicConfig(
        level=numeric_level, format="[%(levelname)s - %(asctime)s] %(name)s: %(message)s", handlers=handlers
    )

    logger = logging.getLogger("benchmark_v2")
    logger.info("Starting benchmark discovery and execution")
    logger.info(f"Benchmark run UUID: {benchmark_run_uuid}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create benchmark configs
    benchmark_configs = cross_generate_configs(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    runner = BenchmarkRunner(logger, args.output_dir, args.commit_id)
    results = runner.run_benchmarks(args.model_id, benchmark_configs)
    # runner.save_results(args.model_id, results)
