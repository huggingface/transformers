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
Generation Scheduler — Test Deployment Script
==============================================

This script runs all scheduler-related tests (unit + integration) and
generates a comprehensive test report.

Usage:
    # Run all tests (quick — unit + fast integration)
    python scripts/run_scheduler_tests.py

    # Run with slow tests included
    python scripts/run_scheduler_tests.py --slow

    # Run only unit tests
    python scripts/run_scheduler_tests.py --unit-only

    # Run only integration (black-box) tests
    python scripts/run_scheduler_tests.py --integration-only

    # Run with verbose output
    python scripts/run_scheduler_tests.py --verbose

    # Run with HTML report (requires pytest-html)
    python scripts/run_scheduler_tests.py --html-report

    # Run specific mode tests only
    python scripts/run_scheduler_tests.py --mode none
    python scripts/run_scheduler_tests.py --mode force
    python scripts/run_scheduler_tests.py --mode internal

    # Dry run (show what would be executed)
    python scripts/run_scheduler_tests.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ==============================================================================
# Constants
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests" / "generation"

# Test files
UNIT_TEST_FILES = [
    TESTS_DIR / "test_state_machine.py",
    TESTS_DIR / "test_generation_scheduler.py",
]

INTEGRATION_TEST_FILE = TESTS_DIR / "test_scheduler_integration.py"

# Test class → mode mapping for selective execution
MODE_TEST_CLASSES = {
    "none": [
        "TestNoneModeIntegration",
    ],
    "force": [
        "TestForceModeIntegration",
    ],
    "internal": [
        "TestInternalModeIntegration",
    ],
    "cross": [
        "TestCrossModeIntegration",
    ],
    "edge": [
        "TestEdgeCasesIntegration",
    ],
    "slow": [
        "TestSchedulerSlowIntegration",
    ],
}


# ==============================================================================
# Helpers
# ==============================================================================


def print_banner(title: str):
    """Print a formatted banner."""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_section(title: str):
    """Print a section header."""
    print(f"\n--- {title} ---\n")


def run_pytest(args: list, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run pytest with the given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=run_env)


def check_dependencies():
    """Check that required dependencies are available."""
    print_section("Checking Dependencies")

    checks = []

    # Check torch
    try:
        import torch
        checks.append(("torch", torch.__version__, True))
    except ImportError:
        checks.append(("torch", "NOT FOUND", False))

    # Check transformers
    try:
        import transformers
        checks.append(("transformers", transformers.__version__, True))
    except ImportError:
        checks.append(("transformers", "NOT FOUND", False))

    # Check pytest
    try:
        import pytest
        checks.append(("pytest", pytest.__version__, True))
    except ImportError:
        checks.append(("pytest", "NOT FOUND", False))

    # Check scheduler modules (import modules only to avoid F401 unused names)
    try:
        import transformers.generation.generation_scheduler
        import transformers.generation.scheduler_callbacks
        import transformers.generation.state_machine
        checks.append(("scheduler modules", "OK", True))
    except ImportError as e:
        checks.append(("scheduler modules", f"IMPORT ERROR: {e}", False))

    # Print results
    max_name = max(len(c[0]) for c in checks)
    for name, version, ok in checks:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name:<{max_name}} : {version}")

    all_ok = all(c[2] for c in checks)
    if not all_ok:
        print("\n  [FAIL] Some dependencies are missing. Please install them first.")
        print("    pip install -e '.[dev]'")
        return False

    print("\n  [OK] All dependencies satisfied.")
    return True


def verify_test_files():
    """Verify that all test files exist."""
    print_section("Verifying Test Files")

    all_files = UNIT_TEST_FILES + [INTEGRATION_TEST_FILE]
    all_exist = True

    for f in all_files:
        exists = f.exists()
        status = "[OK]" if exists else "[FAIL]"
        rel = f.relative_to(REPO_ROOT)
        print(f"  {status} {rel}")
        if not exists:
            all_exist = False

    if all_exist:
        print(f"\n  [OK] All {len(all_files)} test files found.")
    else:
        print("\n  [FAIL] Some test files are missing!")

    return all_exist


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run Generation Scheduler tests (unit + integration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_scheduler_tests.py                    # Run all quick tests
  python scripts/run_scheduler_tests.py --slow             # Include slow tests
  python scripts/run_scheduler_tests.py --unit-only        # Unit tests only
  python scripts/run_scheduler_tests.py --integration-only # Integration tests only
  python scripts/run_scheduler_tests.py --mode force       # Only FORCE mode tests
  python scripts/run_scheduler_tests.py --verbose          # Verbose output
  python scripts/run_scheduler_tests.py --html-report      # Generate HTML report
        """,
    )
    parser.add_argument("--slow", action="store_true", help="Include slow tests (RUN_SLOW=1)")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--mode", choices=["none", "internal", "force", "cross", "edge", "slow"],
                        help="Run only tests for a specific mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose test output")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML report (requires pytest-html)")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--parallel", "-n", type=int, default=0,
                        help="Number of parallel workers (requires pytest-xdist)")

    args = parser.parse_args()

    print_banner("Generation Scheduler — Test Suite")

    # Pre-flight checks
    if not args.dry_run:
        if not check_dependencies():
            sys.exit(1)
        if not verify_test_files():
            sys.exit(1)

    # Build pytest arguments
    pytest_args = []
    env = {}

    # Test file selection
    if args.unit_only:
        pytest_args.extend([str(f) for f in UNIT_TEST_FILES])
    elif args.integration_only:
        pytest_args.append(str(INTEGRATION_TEST_FILE))
    elif args.mode:
        # Run specific mode tests from integration file
        pytest_args.append(str(INTEGRATION_TEST_FILE))
        classes = MODE_TEST_CLASSES.get(args.mode, [])
        if classes:
            # Use pytest -k to filter by class name
            k_expr = " or ".join(classes)
            pytest_args.extend(["-k", k_expr])
    else:
        # All tests
        pytest_args.extend([str(f) for f in UNIT_TEST_FILES])
        pytest_args.append(str(INTEGRATION_TEST_FILE))

    # Slow tests
    if args.slow:
        env["RUN_SLOW"] = "1"

    # Verbosity
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-v")  # Always use -v for clarity

    # Short traceback
    pytest_args.extend(["--tb", "short"])

    # Fail fast
    if args.failfast:
        pytest_args.append("-x")

    # Parallel execution
    if args.parallel > 0:
        pytest_args.extend(["-n", str(args.parallel)])

    # HTML report
    if args.html_report:
        report_dir = REPO_ROOT / "test_reports"
        report_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"scheduler_test_report_{timestamp}.html"
        pytest_args.extend(["--html", str(report_path), "--self-contained-html"])

    # Display what will be run
    print_section("Test Configuration")
    print(f"  Mode filter:    {args.mode or 'all'}")
    print(f"  Unit tests:     {'yes' if not args.integration_only else 'no'}")
    print(f"  Integration:    {'yes' if not args.unit_only else 'no'}")
    print(f"  Slow tests:     {'yes' if args.slow else 'no'}")
    print(f"  Fail fast:      {'yes' if args.failfast else 'no'}")
    print(f"  Parallel:       {args.parallel if args.parallel > 0 else 'no'}")
    print(f"  HTML report:    {'yes' if args.html_report else 'no'}")

    cmd_str = f"{sys.executable} -m pytest {' '.join(pytest_args)}"
    if env:
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        cmd_str = f"{env_str} {cmd_str}"
    print(f"\n  Command: {cmd_str}")

    if args.dry_run:
        print("\n  [DRY RUN] Command not executed.")
        return 0

    # Run tests
    print_banner("Running Tests")
    start_time = time.time()
    result = run_pytest(pytest_args, env=env)
    elapsed = time.time() - start_time

    # Summary
    print_banner("Test Summary")
    print(f"  Exit code:      {result.returncode}")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Status:         {'PASSED' if result.returncode == 0 else 'FAILED'}")

    if args.html_report:
        print(f"  HTML report:    {report_path}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
