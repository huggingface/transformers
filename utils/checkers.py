#!/usr/bin/env python
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Unified runner for check/fix scripts.

Usage:
    python utils/checkers.py copies,modular_conversion,doc_toc
    python utils/checkers.py copies,modular_conversion,doc_toc --fix
    python utils/checkers.py copies,doc_toc --keep-going
    python utils/checkers.py all
    python utils/checkers.py all --fix
"""

import argparse
import hashlib
import itertools
import os
import subprocess
import sys
import threading
from pathlib import Path


UTILS_DIR = Path(__file__).parent
REPO_ROOT = UTILS_DIR.parent

# Each checker maps to (label, script_path, extra_check_args, extra_fix_args).
# When fix_args is None, the checker has no fix mode.
# Custom checkers use None instead of the tuple.
CHECKERS = {
    "copies": ("Copied code consistency", "check_copies.py", [], ["--fix_and_overwrite"]),
    "modular_conversion": ("Modular file conversions", "check_modular_conversion.py", [], ["--fix_and_overwrite"]),
    "doc_toc": ("Documentation table of contents", "check_doc_toc.py", [], ["--fix_and_overwrite"]),
    "docstrings": ("Docstring formatting", "check_docstrings.py", [], ["--fix_and_overwrite"]),
    "dummies": ("Dummy objects", "check_dummies.py", [], ["--fix_and_overwrite"]),
    "pipeline_typing": ("Pipeline type hints", "check_pipeline_typing.py", [], ["--fix_and_overwrite"]),
    "doctest_list": ("Doctest list", "check_doctest_list.py", [], ["--fix_and_overwrite"]),
    "repo": ("Repository structure", "check_repo.py", [], None),
    "inits": ("Init files", "check_inits.py", [], None),
    "config_docstrings": ("Config docstrings", "check_config_docstrings.py", [], None),
    "config_attributes": ("Config attributes", "check_config_attributes.py", [], None),
    "init_isort": ("Import ordering", "custom_init_isort.py", ["--check_only"], []),
    "auto_mappings": ("Auto mappings", "sort_auto_mappings.py", ["--check_only"], []),
    "update_metadata": ("Model metadata", "update_metadata.py", ["--check-only"], []),
    "add_dates": ("Model dates", "add_dates.py", ["--check-only"], []),
    "types": (
        "Type annotations",
        "check_types.py",
        [
            "src/transformers/_typing.py",
            "src/transformers/utils",
            "src/transformers/generation",
            "src/transformers/quantizers",
        ],
        None,
    ),
    "modeling_structure": ("Modeling file structure", "check_modeling_structure.py", [], None),
    "deps_table": ("Dependency versions table", None, None, None),
    "imports": ("Public imports", None, None, None),
    "ruff_check": ("Ruff linting", None, None, None),
    "ruff_format": ("Ruff formatting", None, None, None),
}


def _file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _run_cmd(cmd):
    """Run a command, capturing output. Returns (returncode, output)."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.decode("utf-8", errors="replace")


def run_deps_table_checker(fix=False):
    """Check or fix the dependency versions table."""
    deps_table = REPO_ROOT / "src" / "transformers" / "dependency_versions_table.py"
    setup_py = REPO_ROOT / "setup.py"
    cmd = [sys.executable, str(setup_py), "deps_table_update"]

    if fix:
        return _run_cmd(cmd)

    before = _file_md5(deps_table)
    rc, output = _run_cmd(cmd)
    if rc != 0:
        return rc, output
    after = _file_md5(deps_table)
    if before != after:
        msg = (
            "Error: the version dependency table is outdated.\n"
            "Please run 'make fix-repo' and commit the changes. This requires Python 3.10.\n"
        )
        return 1, output + msg
    return 0, output


def run_imports_checker(fix=False):
    """Check that all public imports work."""
    rc, output = _run_cmd([sys.executable, "-c", "from transformers import *"])
    if rc != 0:
        return rc, output + "Import failed, this means you introduced unprotected imports!\n"
    return 0, output


RUFF_TARGETS = ["examples", "tests", "src", "utils", "scripts", "benchmark", "benchmark_v2", "setup.py", "conftest.py"]


def run_ruff_check(fix=False):
    """Run ruff linting."""
    cmd = ["ruff", "check", *RUFF_TARGETS]
    if fix:
        cmd += ["--fix", "--exclude", ""]
    return _run_cmd(cmd)


def run_ruff_format(fix=False):
    """Run ruff formatting."""
    cmd = ["ruff", "format", *RUFF_TARGETS]
    if not fix:
        cmd += ["--check"]
    else:
        cmd += ["--exclude", ""]
    return _run_cmd(cmd)


CUSTOM_RUNNERS = {
    "deps_table": run_deps_table_checker,
    "imports": run_imports_checker,
    "ruff_check": run_ruff_check,
    "ruff_format": run_ruff_format,
}


def run_checker(name, fix=False):
    if name in CUSTOM_RUNNERS:
        return CUSTOM_RUNNERS[name](fix=fix)

    _, script, check_args, fix_args = CHECKERS[name]
    script_path = UTILS_DIR / script

    if fix and fix_args is None:
        return 0, "skipped (no fix mode)"

    cmd = [sys.executable, str(script_path)]
    cmd += fix_args if fix else check_args

    return _run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run check/fix scripts.")
    parser.add_argument(
        "checkers",
        nargs="+",
        help='Comma-separated checker names, or "all". Use --list to see available checkers.',
    )
    parser.add_argument("--fix", action="store_true", help="Run in fix mode instead of check mode.")
    parser.add_argument(
        "--keep-going", action="store_true", help="Run all checkers even if some fail (report failures at the end)."
    )
    parser.add_argument("--list", action="store_true", help="List available checkers and exit.")

    args = parser.parse_args()

    if args.list:
        for name, entry in sorted(CHECKERS.items()):
            label, script, _, fix_args = entry
            fixable = "fixable" if fix_args is not None else "check-only"
            script_display = script or "custom"
            print(f"  {name:25s} {label:35s} ({script_display}, {fixable})")
        return

    # Join all positional args (shell line continuations may split them) and parse checker names
    raw = " ".join(args.checkers)
    if raw.strip() == "all":
        names = list(CHECKERS.keys())
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]

    unknown = [n for n in names if n not in CHECKERS]
    if unknown:
        print(f"Unknown checkers: {', '.join(unknown)}")
        print(f"Available: {', '.join(sorted(CHECKERS.keys()))}")
        sys.exit(1)

    is_ci = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CIRCLECI") == "true"
    is_tty = sys.stdout.isatty() and not is_ci
    spinner_chars = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    stop_spinner = threading.Event()

    def spin(label):
        while not stop_spinner.is_set():
            print(f"\r{label}... {next(spinner_chars)}", end="", flush=True)
            stop_spinner.wait(0.08)

    failures = []
    for name in names:
        label = CHECKERS[name][0]
        t = None
        if is_tty:
            stop_spinner.clear()
            t = threading.Thread(target=spin, args=(label,))
            t.start()
        else:
            print(f"{label}... ", end="", flush=True)
        rc, output = run_checker(name, fix=args.fix)
        if t is not None:
            stop_spinner.set()
            t.join()
        if rc == 0:
            print(f"\r{label}... OK" if is_tty else "OK")
        else:
            print(f"\r{label}... FAILED" if is_tty else "FAILED")
            print(output)
            failures.append(name)
            if not args.keep_going:
                sys.exit(1)

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)

    print(f"\nAll {len(names)} checks passed.")


if __name__ == "__main__":
    main()
