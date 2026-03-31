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
import shutil
import subprocess
import sys
import threading
from collections import deque
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
    "import_complexity": ("Import complexity", "check_import_complexity.py", [], None),
    "ruff_check": ("Ruff linting", None, None, None),
    "ruff_format": ("Ruff formatting", None, None, None),
}


def _file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# ANSI helpers
ORANGE = "\033[38;5;214m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class SlidingWindow:
    """Displays a spinning title + sliding window of the last N output lines in a TTY."""

    def __init__(self, label, max_lines=10):
        self.label = label
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        self.displayed = 0  # number of output lines currently on screen
        self.term_width = shutil.get_terminal_size().columns
        self._spinner = itertools.cycle(SPINNER_CHARS)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Print initial title line (will be overwritten by spinner)
        print(f"{ORANGE}{next(self._spinner)} {label}{RESET}")
        self._title_on_screen = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            self._stop.wait(0.08)
            if self._stop.is_set():
                break
            with self._lock:
                self._redraw()

    def _redraw(self):
        """Clear output lines + title, redraw everything."""
        # Move up over output lines + title line
        for _ in range(self.displayed + (1 if self._title_on_screen else 0)):
            sys.stdout.write("\033[A\033[2K")
        self.displayed = 0
        # Redraw title with next spinner frame
        print(f"{ORANGE}{next(self._spinner)} {self.label}{RESET}")
        self._title_on_screen = True
        # Redraw output lines
        for line in self.lines:
            print(line)
        self.displayed = len(self.lines)
        sys.stdout.flush()

    def add_line(self, line):
        with self._lock:
            self.lines.append(line.rstrip()[: self.term_width])
            self._redraw()

    def finish(self, success):
        """Stop spinner and print final status title."""
        self._stop.set()
        self._thread.join()
        with self._lock:
            # Clear output lines + title
            for _ in range(self.displayed + (1 if self._title_on_screen else 0)):
                sys.stdout.write("\033[A\033[2K")
            self._title_on_screen = False
            self.displayed = 0
            # Print final title with status
            if success:
                print(f"{GREEN}✓ {self.label}{RESET}")
            else:
                print(f"{RED}✗ {self.label}{RESET}")
            # Reprint output lines
            for line in self.lines:
                print(line)
            sys.stdout.flush()


def _run_cmd(cmd, line_callback=None):
    """Run a command, capturing output. Returns (returncode, output)."""
    if line_callback is None:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.returncode, result.stdout.decode("utf-8", errors="replace")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output_lines = []
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace")
        output_lines.append(line)
        line_callback(line)
    proc.wait()
    return proc.returncode, "".join(output_lines)


def run_deps_table_checker(fix=False, line_callback=None):
    """Check or fix the dependency versions table."""
    deps_table = REPO_ROOT / "src" / "transformers" / "dependency_versions_table.py"
    setup_py = REPO_ROOT / "setup.py"
    cmd = [sys.executable, str(setup_py), "deps_table_update"]

    if fix:
        return _run_cmd(cmd, line_callback=line_callback)

    before = _file_md5(deps_table)
    rc, output = _run_cmd(cmd, line_callback=line_callback)
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


def run_imports_checker(fix=False, line_callback=None):
    """Check that all public imports work."""
    rc, output = _run_cmd([sys.executable, "-c", "from transformers import *"], line_callback=line_callback)
    if rc != 0:
        return rc, output + "Import failed, this means you introduced unprotected imports!\n"
    return 0, output


RUFF_TARGETS = ["examples", "tests", "src", "utils", "scripts", "benchmark", "benchmark_v2", "setup.py", "conftest.py"]


def run_ruff_check(fix=False, line_callback=None):
    """Run ruff linting."""
    cmd = ["ruff", "check", *RUFF_TARGETS]
    if fix:
        cmd += ["--fix", "--exclude", ""]
    return _run_cmd(cmd, line_callback=line_callback)


def run_ruff_format(fix=False, line_callback=None):
    """Run ruff formatting."""
    cmd = ["ruff", "format", *RUFF_TARGETS]
    if not fix:
        cmd += ["--check"]
    else:
        cmd += ["--exclude", ""]
    return _run_cmd(cmd, line_callback=line_callback)


CUSTOM_RUNNERS = {
    "deps_table": run_deps_table_checker,
    "imports": run_imports_checker,
    "ruff_check": run_ruff_check,
    "ruff_format": run_ruff_format,
}


def get_checker_command(name, fix=False):
    """Return a shell-friendly command string for a checker."""
    if name == "deps_table":
        return "python setup.py deps_table_update"
    if name == "imports":
        return 'python -c "from transformers import *"'
    if name == "ruff_check":
        cmd = ["ruff", "check", *RUFF_TARGETS]
        if fix:
            cmd += ["--fix", "--exclude", ""]
        return " ".join(cmd)
    if name == "ruff_format":
        cmd = ["ruff", "format", *RUFF_TARGETS]
        if not fix:
            cmd += ["--check"]
        else:
            cmd += ["--exclude", ""]
        return " ".join(cmd)

    _, script, check_args, fix_args = CHECKERS[name]
    if fix and fix_args is None:
        return None
    args = fix_args if fix else check_args
    return " ".join(["python", f"utils/{script}"] + args)


def run_checker(name, fix=False, line_callback=None):
    if name in CUSTOM_RUNNERS:
        return CUSTOM_RUNNERS[name](fix=fix, line_callback=line_callback)

    _, script, check_args, fix_args = CHECKERS[name]
    script_path = UTILS_DIR / script

    if fix and fix_args is None:
        return 0, "skipped (no fix mode)"

    cmd = [sys.executable, str(script_path)]
    cmd += fix_args if fix else check_args

    return _run_cmd(cmd, line_callback=line_callback)


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

    failures = []
    for name in names:
        label = CHECKERS[name][0]
        cmd_str = get_checker_command(name, fix=args.fix)

        if is_tty:
            window = SlidingWindow(label, max_lines=10)
            if cmd_str:
                window.add_line(f"$ {cmd_str}")
            rc, output = run_checker(name, fix=args.fix, line_callback=window.add_line)
            window.finish(success=(rc == 0))
            print()
            if rc != 0:
                failures.append(name)
                if not args.keep_going:
                    sys.exit(1)
        else:
            print(f"{label}")
            if cmd_str:
                print(f"$ {cmd_str}")
            rc, output = run_checker(name, fix=args.fix)
            tail = output.splitlines()[-10:]
            if tail:
                print("\n".join(tail))
            status = "OK" if rc == 0 else "FAILED"
            print(status)
            print()
            if rc != 0:
                failures.append(name)
                if not args.keep_going:
                    sys.exit(1)

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)

    print(f"\nAll {len(names)} checks passed.")


if __name__ == "__main__":
    main()
