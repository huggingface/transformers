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
import importlib.util
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def setup_logging(log_level: str = "INFO", enable_file_logging: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    handlers = [logging.StreamHandler(sys.stdout)]

    if enable_file_logging:
        handlers.append(logging.FileHandler(f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))

    logging.basicConfig(
        level=numeric_level, format="[%(levelname)s - %(asctime)s] %(name)s: %(message)s", handlers=handlers
    )

    return logging.getLogger(__name__)


def discover_benchmarks(benches_dir: str) -> list[dict[str, Any]]:
    """
    Discover all benchmark modules in the benches directory.

    Returns:
        List of dictionaries containing benchmark module info
    """
    benchmarks = []
    benches_path = Path(benches_dir)

    if not benches_path.exists():
        raise FileNotFoundError(f"Benches directory not found: {benches_dir}")

    for py_file in benches_path.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        module_name = py_file.stem

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if it has a benchmark runner function
            if hasattr(module, f"run_{module_name}"):
                benchmarks.append(
                    {
                        "name": module_name,
                        "path": str(py_file),
                        "module": module,
                        "runner_function": getattr(module, f"run_{module_name}"),
                    }
                )
            elif hasattr(module, "run_benchmark"):
                benchmarks.append(
                    {
                        "name": module_name,
                        "path": str(py_file),
                        "module": module,
                        "runner_function": getattr(module, "run_benchmark"),
                    }
                )
            else:
                logging.warning(f"No runner function found in {py_file}")

        except Exception as e:
            logging.error(f"Failed to import {py_file}: {e}")

    return benchmarks


def run_single_benchmark(
    benchmark_info: dict[str, Any], output_dir: str, logger: logging.Logger, **kwargs
) -> Optional[str]:
    """
    Run a single benchmark and return the output file path.

    Args:
        benchmark_info: Dictionary containing benchmark module info
        output_dir: Base output directory
        logger: Logger instance
        **kwargs: Additional arguments to pass to the benchmark

    Returns:
        Path to the output file if successful, None otherwise
    """
    benchmark_name = benchmark_info["name"]
    runner_func = benchmark_info["runner_function"]

    logger.info(f"Running benchmark: {benchmark_name}")

    try:
        # Check function signature to determine what arguments to pass
        import inspect

        sig = inspect.signature(runner_func)

        # Prepare arguments based on function signature
        func_kwargs = {"logger": logger, "output_dir": output_dir}

        # Add other kwargs if the function accepts them
        for param_name in sig.parameters:
            if param_name in kwargs:
                func_kwargs[param_name] = kwargs[param_name]

        # Filter kwargs to only include parameters the function accepts
        # If function has **kwargs, include all provided kwargs
        has_var_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
        if has_var_kwargs:
            valid_kwargs = {**func_kwargs, **kwargs}
        else:
            valid_kwargs = {k: v for k, v in func_kwargs.items() if k in sig.parameters}

        # Run the benchmark
        result = runner_func(**valid_kwargs)

        if isinstance(result, str):
            # Function returned a file path
            return result
        else:
            logger.info(f"Benchmark {benchmark_name} completed successfully")
            return "completed"

    except Exception as e:
        logger.error(f"Benchmark {benchmark_name} failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def generate_summary_report(
    output_dir: str,
    benchmark_results: dict[str, Any],
    logger: logging.Logger,
    benchmark_run_uuid: Optional[str] = None,
) -> str:
    """Generate a summary report of all benchmark runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")

    summary_data = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmark_run_uuid": benchmark_run_uuid,
            "total_benchmarks": len(benchmark_results),
            "successful_benchmarks": len([r for r in benchmark_results.values() if r is not None]),
            "failed_benchmarks": len([r for r in benchmark_results.values() if r is None]),
        },
        "benchmark_results": benchmark_results,
        "output_directory": output_dir,
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info(f"Summary report saved to: {summary_file}")
    return summary_file


def upload_results_to_hf_dataset(
    output_dir: str,
    summary_file: str,
    dataset_name: str,
    run_id: Optional[str] = None,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Upload benchmark results to a HuggingFace Dataset.
    Based on upload_collated_report() from utils/collated_reports.py
    Args:
        output_dir: Local output directory containing results
        summary_file: Path to the summary file
        dataset_name: Name of the HuggingFace dataset to upload to
        run_id: Unique run identifier (if None, will generate one)
        token: HuggingFace token for authentication (if None, will use environment variables)
        logger: Logger instance
    Returns:
        The run_id used for the upload, None if upload failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    import os

    from huggingface_hub import HfApi

    api = HfApi()

    if run_id is None:
        github_run_number = os.getenv("GITHUB_RUN_NUMBER")
        github_run_id = os.getenv("GITHUB_RUN_ID")
        if github_run_number and github_run_id:
            run_id = f"{github_run_number}-{github_run_id}"

    date_folder = datetime.now().strftime("%Y-%m-%d")

    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    if github_event_name != "schedule":
        # Non-scheduled runs go under a runs subfolder
        repo_path = f"{date_folder}/runs/{run_id}/benchmark_results"
    else:
        # Scheduled runs go directly under the date
        repo_path = f"{date_folder}/{run_id}/benchmark_results"

    logger.info(f"Uploading benchmark results to dataset '{dataset_name}' at path '{repo_path}'")

    try:
        # Upload all files in the output directory
        from pathlib import Path

        output_path = Path(output_dir)

        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                # Calculate relative path from output_dir
                relative_path = file_path.relative_to(output_path)
                path_in_repo = f"{repo_path}/{relative_path}"

                logger.debug(f"Uploading {file_path} to {path_in_repo}")

                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Upload benchmark results for run {run_id}",
                )

        logger.info(
            f"Successfully uploaded results to: https://huggingface.co/datasets/{dataset_name}/tree/main/{repo_path}"
        )

        return run_id

    except Exception as upload_error:
        logger.error(f"Failed to upload results: {upload_error}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def main():
    """Main entry point for the benchmarking script."""
    # Generate a unique UUID for this benchmark run
    benchmark_run_uuid = str(uuid.uuid4())[:8]

    parser = argparse.ArgumentParser(
        description="Run all benchmarks in the ./benches directory",
        epilog="""
Examples:
  # Run all available benchmarks
  python3 run_benchmarks.py
  
  # Run with specific model and upload to HuggingFace Dataset
  python3 run_benchmarks.py --model-id meta-llama/Llama-2-7b-hf --upload-to-hf username/benchmark-results
  
  # Run with custom run ID and upload to HuggingFace Dataset
  python3 run_benchmarks.py --run-id experiment_v1 --upload-to-hf org/benchmarks
  
  # Run only specific benchmarks with file logging
  python3 run_benchmarks.py --include llama --enable-file-logging
        """,  # noqa: W293
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Base output directory for benchmark results (default: benchmark_results)",
    )

    parser.add_argument(
        "--benches-dir",
        type=str,
        default="./benches",
        help="Directory containing benchmark implementations (default: ./benches)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--model-id", type=str, help="Specific model ID to benchmark (if supported by benchmarks)")

    parser.add_argument("--warmup-iterations", type=int, default=3, help="Number of warmup iterations (default: 3)")

    parser.add_argument(
        "--measurement-iterations", type=int, default=5, help="Number of measurement iterations (default: 5)"
    )

    parser.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=100,
        help="Number of tokens to generate in benchmarks (default: 100)",
    )

    parser.add_argument("--include", type=str, nargs="*", help="Only run benchmarks matching these names")

    parser.add_argument("--exclude", type=str, nargs="*", help="Exclude benchmarks matching these names")

    parser.add_argument("--enable-file-logging", action="store_true", help="Enable file logging (disabled by default)")

    parser.add_argument(
        "--commit-id", type=str, help="Git commit ID for metadata (if not provided, will auto-detect from git)"
    )

    parser.add_argument(
        "--push-to-hub",
        type=str,
        help="Upload results to HuggingFace Dataset (provide dataset name, e.g., 'username/benchmark-results')",
    )

    parser.add_argument(
        "--run-id", type=str, help="Custom run ID for organizing results (if not provided, will generate a unique ID)"
    )

    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token for dataset uploads (if not provided, will use HF_TOKEN environment variable)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level, args.enable_file_logging)

    logger.info("Starting benchmark discovery and execution")
    logger.info(f"Benchmark run UUID: {benchmark_run_uuid}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Benches directory: {args.benches_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Discover benchmarks
        benchmarks = discover_benchmarks(args.benches_dir)
        logger.info(f"Discovered {len(benchmarks)} benchmark(s): {[b['name'] for b in benchmarks]}")

        if not benchmarks:
            logger.warning("No benchmarks found!")
            return 1

        # Filter benchmarks based on include/exclude
        filtered_benchmarks = benchmarks

        if args.include:
            filtered_benchmarks = [
                b for b in filtered_benchmarks if any(pattern in b["name"] for pattern in args.include)
            ]
            logger.info(f"Filtered to include: {[b['name'] for b in filtered_benchmarks]}")

        if args.exclude:
            filtered_benchmarks = [
                b for b in filtered_benchmarks if not any(pattern in b["name"] for pattern in args.exclude)
            ]
            logger.info(f"After exclusion: {[b['name'] for b in filtered_benchmarks]}")

        if not filtered_benchmarks:
            logger.warning("No benchmarks remaining after filtering!")
            return 1

        # Prepare common kwargs for benchmarks
        benchmark_kwargs = {
            "warmup_iterations": args.warmup_iterations,
            "measurement_iterations": args.measurement_iterations,
            "num_tokens_to_generate": args.num_tokens_to_generate,
        }

        if args.model_id:
            benchmark_kwargs["model_id"] = args.model_id

        # Add commit_id if provided
        if args.commit_id:
            benchmark_kwargs["commit_id"] = args.commit_id

        # Run benchmarks
        benchmark_results = {}
        successful_count = 0

        for benchmark_info in filtered_benchmarks:
            result = run_single_benchmark(benchmark_info, args.output_dir, logger, **benchmark_kwargs)

            benchmark_results[benchmark_info["name"]] = result

            if result is not None:
                successful_count += 1

        # Generate summary report
        summary_file = generate_summary_report(args.output_dir, benchmark_results, logger, benchmark_run_uuid)

        # Upload results to HuggingFace Dataset if requested
        upload_run_id = None
        if args.push_to_hub:
            logger.info("=" * 60)
            logger.info("UPLOADING TO HUGGINGFACE DATASET")
            logger.info("=" * 60)
            # Use provided run_id or fallback to benchmark run UUID
            effective_run_id = args.run_id or benchmark_run_uuid
            upload_run_id = upload_results_to_hf_dataset(
                output_dir=args.output_dir,
                summary_file=summary_file,
                dataset_name=args.push_to_hub,
                run_id=effective_run_id,
                token=args.token,
                logger=logger,
            )
            if upload_run_id:
                logger.info(f"Upload completed with run ID: {upload_run_id}")
            else:
                logger.warning("Upload failed - continuing with local results")

        # Final summary
        total_benchmarks = len(filtered_benchmarks)
        failed_count = total_benchmarks - successful_count

        logger.info("=" * 60)
        logger.info("BENCHMARK RUN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total benchmarks: {total_benchmarks}")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Summary report: {summary_file}")

        if args.push_to_hub:
            if upload_run_id:
                logger.info(f"HuggingFace Dataset: {args.push_to_hub}")
                logger.info(f"Run ID: {upload_run_id}")
                logger.info(
                    f"View results: https://huggingface.co/datasets/{args.push_to_hub}/tree/main/{datetime.now().strftime('%Y-%m-%d')}/runs/{upload_run_id}"
                )
            else:
                logger.warning("Upload to HuggingFace Dataset failed")

        if failed_count > 0:
            logger.warning(f"{failed_count} benchmark(s) failed. Check logs for details.")
            return 1
        else:
            logger.info("All benchmarks completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
