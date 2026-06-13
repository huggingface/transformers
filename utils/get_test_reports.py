# coding=utf-8
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
This util provides a way to manually run the tests of the transformers repo as they would be run by the CI.
It was mainly used for models tests, so if you find features missing for another suite, do not hesitate to open a PR.

Functionnalities:
- Running specific test suite (models, tokenizers, etc.)
- Parallel execution across multiple processes (each has to be launched separately with different `--processes` argument)
- GPU/CPU test filtering and slow tests filter
- Temporary cache management for isolated test runs
- Resume functionality for interrupted test runs
- Important models subset testing

Example usages are below.
"""

import argparse
import contextlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch

from .important_files import IMPORTANT_MODELS


def is_valid_test_dir(path: Path) -> bool:
    """Check if a given path represents a valid test dir: the path must point to a dir, not start with '__' or '.'"""
    return path.is_dir() and not path.name.startswith("__") and not path.name.startswith(".")


def run_pytest(
    suite: str, subdir: Path, root_test_dir: Path, machine_type: str, dry_run: bool, tmp_cache: str, cpu_tests: bool
) -> None:
    """
    Execute pytest on a specific test directory with configured options:
        - suite (str): name of the test suite being run (e.g., 'models', 'tokenizers')
        - subdir (Path): the specific directory containing tests to run
        - root_test_dir (Path): the root directory of all tests, used for relative paths
        - machine_type (str): type of machine/environment (e.g., 'cpu', 'single-gpu', 'multi-gpu')
        - dry_run (bool): if True, only print the command without executing it
        - tmp_cache (str): prefix for temporary cache directory. If empty, no temp cache is used
        - cpu_tests (bool): if True, include CPU-only tests; if False, exclude non-device tests
    """
    relative_path = subdir.relative_to(root_test_dir)
    report_name = f"{machine_type}_{suite}_{relative_path}_test_reports"
    print(f"Suite: {suite} | Running on: {relative_path}")

    cmd = ["python3", "-m", "pytest", "-rsfE", "-v", f"--make-reports={report_name}", str(subdir)]
    if not cpu_tests:
        cmd = cmd + ["-m", "not not_device_test"]

    ctx_manager = tempfile.TemporaryDirectory(prefix=tmp_cache) if tmp_cache else contextlib.nullcontext()
    with ctx_manager as tmp_dir:
        env = os.environ.copy()
        if tmp_cache:
            env["HUGGINGFACE_HUB_CACHE"] = tmp_dir

            print(f"Using temporary cache located at {tmp_dir = }")

        print("Command:", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=False, env=env)


def handle_suite(
    suite: str,
    test_root: Path,
    machine_type: str,
    dry_run: bool,
    tmp_cache: str = "",
    resume_at: Optional[str] = None,
    only_in: Optional[list[str]] = None,
    cpu_tests: bool = False,
    process_id: int = 1,
    total_processes: int = 1,
) -> None:
    """
    Handle execution of a complete test suite with advanced filtering and process distribution.
    Args:
        - suite (str): Name of the test suite to run (corresponds to a directory under test_root).
        - test_root (Path): Root directory containing all test suites.
        - machine_type (str): Machine/environment type for report naming and identification.
        - dry_run (bool): If True, only print commands without executing them.
        - tmp_cache (str, optional): Prefix for temporary cache directories. If empty, no temp cache is used.
        - resume_at (str, optional): Resume execution starting from this subdirectory name.
            Useful for restarting interrupted test runs. Defaults to None (run from the beginning).
        - only_in (list[str], optional): Only run tests in these specific subdirectories.
            Can include special values like IMPORTANT_MODELS. Defaults to None (run all tests).
        - cpu_tests (bool, optional): Whether to include CPU-only tests. Defaults to False.
        - process_id (int, optional): Current process ID for parallel execution (1-indexed). Defaults to 1.
        - total_processes (int, optional): Total number of parallel processes. Defaults to 1.
    """
    # Check path to suite
    full_path = test_root / suite
    if not full_path.exists():
        print(f"Test folder does not exist: {full_path}")
        return

    # Establish the list of subdir to go through
    subdirs = sorted(full_path.iterdir())
    subdirs = [s for s in subdirs if is_valid_test_dir(s)]
    if resume_at is not None:
        subdirs = [s for s in subdirs if s.name >= resume_at]
    if only_in is not None:
        subdirs = [s for s in subdirs if s.name in only_in]
    if subdirs and total_processes > 1:
        # This interleaves the subdirs / files. For instance for subdirs = [A, B, C, D, E] and 2 processes:
        # - script launcehd with `--processes 0 2` will run A, C, E
        # - script launcehd with `--processes 1 2` will run B, D
        subdirs = subdirs[process_id::total_processes]

    # If the subdir list is not empty, go through each
    if subdirs:
        for subdir in subdirs:
            run_pytest(suite, subdir, test_root, machine_type, dry_run, tmp_cache, cpu_tests)
    # Otherwise, launch pytest from the full path
    else:
        run_pytest(suite, full_path, test_root, machine_type, dry_run, tmp_cache, cpu_tests)


if __name__ == "__main__":
    """Command-line interface for running test suite with comprehensive reporting. Check handle_suite for more details.

    Command-line Arguments:
        folder: Path to the root test directory (required)
        --suite: Test suite name to run (default: "models")
        --cpu-tests: Include CPU-only tests in addition to device tests
        --run-slow: Execute slow tests instead of skipping them
        --resume-at: Resume execution from a specific subdirectory
        --only-in: Run tests only in specified subdirectories (supports IMPORTANT_MODELS)
        --processes: Process distribution as "process_id total_processes"
        --dry-run: Print commands without executing them
        --tmp-cache: Use temporary cache directories for isolated runs
        --machine-type: Override automatic machine type detection

    Machine Type Detection:
        - 'cpu': No CUDA available
        - 'single-gpu': CUDA available with 1 GPU
        - 'multi-gpu': CUDA available with multiple GPUs

    Process Distribution:
        Use --processes to split work across multiple parallel processes:
        --processes 0 4  # This is process 0 of 4 total processes
        --processes 1 4  # This is process 1 of 4 total processes
        ...

    Usage Examples:
        # Basic model testing
        python3 -m utils.get_test_reports tests/ --suite models

        # Run slow tests for important models only
        python3 -m utils.get_test_reports tests/ --suite models --run-slow --only-in IMPORTANT_MODELS

        # Parallel execution across 4 processes, second process to launch (processes are 0-indexed)
        python3 -m utils.get_test_reports tests/ --suite models --processes 1 4

        # Resume interrupted run from 'bert' subdirectory with a tmp cache
        python3 -m utils.get_test_reports tests/ --suite models --resume-at bert --tmp-cache /tmp/

        # Run specific models with CPU tests
        python3 -m utils.get_test_reports tests/ --suite models --only-in bert gpt2 --cpu-tests

        # Run slow tests for only important models with a tmp cache
        python3 -m utils.get_test_reports tests/ --suite models --run-slow --only-in IMPORTANT_MODELS --tmp-cache /tmp/
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to test root folder (e.g., ./tests)")

    # Choose which tests to run (broad picture)
    parser.add_argument("--suite", type=str, default="models", help="Test suit to run")
    parser.add_argument("--cpu-tests", action="store_true", help="Also runs non-device tests")
    parser.add_argument("--run-slow", action="store_true", help="Run slow tests instead of skipping them")
    parser.add_argument("--collect-outputs", action="store_true", help="Collect outputs of the tests")

    # Fine-grain control over the tests to run
    parser.add_argument("--resume-at", type=str, default=None, help="Resume at a specific subdir / file in the suite")
    parser.add_argument(
        "--only-in",
        type=str,
        nargs="+",
        help="Only run tests in the given subdirs / file. Use IMPORTANT_MODELS to run only the important models tests.",
    )

    # How to run the test suite: is the work divided among processes, do a try run, use temp cache?
    parser.add_argument(
        "--processes",
        type=int,
        nargs="+",
        help="Inform each CI process as to the work to do: format as `process_id total_processes`. "
        "In order to run with multiple (eg. 3) processes, you need to run the script multiple times (eg. 3 times).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without running them")
    parser.add_argument("--tmp-cache", type=str, help="Change HUGGINGFACE_HUB_CACHE to a tmp dir for each test")

    # This is a purely decorative argument, but it can be useful to distinguish between runs
    parser.add_argument(
        "--machine-type", type=str, default="", help="Machine type, automatically inferred if not provided"
    )
    args = parser.parse_args()

    # Handle run slow
    if args.run_slow:
        os.environ["RUN_SLOW"] = "yes"
        print("[WARNING] Running slow tests.")
    else:
        print("[WARNING] Skipping slow tests.")

    # Handle multiple CI processes
    if args.processes is None:
        process_id, total_processes = 1, 1
    elif len(args.processes) == 2:
        process_id, total_processes = args.processes
    else:
        raise ValueError(f"Invalid processes argument: {args.processes}")

    # Assert test root exists
    test_root = Path(args.folder).resolve()
    if not test_root.exists():
        print(f"Root test folder not found: {test_root}")
        exit(1)

    # Handle collection of outputs
    if args.collect_outputs:
        os.environ["PATCH_TESTING_METHODS_TO_COLLECT_OUTPUTS"] = "yes"
        reports_dir = test_root.parent / "reports"
        os.environ["_PATCHED_TESTING_METHODS_OUTPUT_DIR"] = str(reports_dir)

    # Infer machine type if not provided
    if args.machine_type == "":
        if not torch.cuda.is_available():
            machine_type = "cpu"
        else:
            machine_type = "multi-gpu" if torch.cuda.device_count() > 1 else "single-gpu"
    else:
        machine_type = args.machine_type

    # Reduce the scope for models if necessary
    only_in = args.only_in if args.only_in else None
    if only_in == ["IMPORTANT_MODELS"]:
        only_in = IMPORTANT_MODELS

    # Launch suite
    handle_suite(
        suite=args.suite,
        test_root=test_root,
        machine_type=machine_type,
        dry_run=args.dry_run,
        tmp_cache=args.tmp_cache,
        resume_at=args.resume_at,
        only_in=only_in,
        cpu_tests=args.cpu_tests,
        process_id=process_id,
        total_processes=total_processes,
    )
