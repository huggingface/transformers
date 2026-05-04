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

Plugin system
-------------
Each checker module declares a ``CHECKER_CONFIG`` dict (extracted via ``ast.literal_eval``,
no import needed — this keeps discovery fast and avoids executing checker code at scan time).
See any ``check_*.py`` file for the schema.

Cache semantics of ``file_globs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``file_globs`` lists the file patterns whose content is hashed to decide whether a checker
can be skipped. **Not all globs are exact reflections of the checker's runtime behaviour.**

* Some checkers introspect the live ``transformers`` module (``check_repo``,
  ``check_config_docstrings``, ``check_config_attributes``, ``update_metadata``), so their
  globs are necessarily *approximations* of the true dependency set.
* Some checkers over-approximate (``check_dummies``, ``check_doctest_list``): any change
  inside the broad glob forces a re-run even if the checker wouldn't look at that file.
  This is safe—just less cache-efficient.
* Some checkers rely on external state (network, git history, installed packages) that
  cannot be captured by file globs at all (``add_dates``, ``imports``).

Each ``CHECKER_CONFIG`` that is an approximation has an inline comment explaining the
gap. When in doubt, use ``--no-cache`` to force a full run.
"""

import argparse
import ast
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path


UTILS_DIR = Path(__file__).parent
REPO_ROOT = UTILS_DIR.parent
CACHE_PATH = UTILS_DIR / ".checkers_cache.json"

# Required keys in each module's CHECKER_CONFIG dict.
_CHECKER_CONFIG_KEYS = {"name", "label", "file_globs", "check_args", "fix_args"}


def _discover_checkers() -> tuple[dict, dict]:
    """Scan utils/*.py for CHECKER_CONFIG dicts using AST (no imports).

    Each checker module may define a top-level ``CHECKER_CONFIG`` dict with
    keys: name, label, file_globs, check_args, fix_args.

    Returns (checkers_dict, file_globs_dict) matching the shapes of
    the old CHECKERS and CHECKER_FILE_GLOBS registries.
    """
    checkers = {}
    file_globs = {}

    for py_file in sorted(UTILS_DIR.glob("*.py")):
        if py_file.name == Path(__file__).name:
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        config = None
        for node in ast.iter_child_nodes(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "CHECKER_CONFIG"
            ):
                try:
                    config = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
                break

        if config is None:
            continue

        missing = _CHECKER_CONFIG_KEYS - set(config)
        if missing:
            warnings.warn(
                f"CHECKER_CONFIG in {py_file.name} is missing keys: {', '.join(sorted(missing))}. Skipping.",
                stacklevel=1,
            )
            continue

        name = config["name"]
        if name in checkers:
            warnings.warn(
                f"Duplicate checker name {name!r} in {py_file.name}, already defined by {checkers[name][1]}",
                stacklevel=1,
            )

        checkers[name] = (
            config["label"],
            py_file.name,
            config["check_args"],
            config["fix_args"],
        )
        if config["file_globs"] is not None:
            file_globs[name] = config["file_globs"]

    return checkers, file_globs


# Inline checkers have no separate script file; they use custom runner functions below.
# fix_args=[] marks a checker as fix-capable (its custom runner handles --fix internally);
# fix_args=None marks a check-only entry that `make fix-repo` should silently skip.
_INLINE_CHECKERS = {
    "deps_table": ("Dependency versions table", None, None, []),
    "imports": ("Public imports", None, None, None),
    "import_complexity": ("Import complexity", "check_import_complexity.py", [], None),
    "ruff_check": ("Ruff linting", None, None, []),
    "ruff_format": ("Ruff formatting", None, None, []),
}

_INLINE_FILE_GLOBS = {
    # Also generates/checks src/transformers/dependency_versions_table.py.
    "deps_table": ["setup.py", "pyproject.toml", "src/transformers/dependency_versions_table.py"],
    # Approximate: runs `from transformers import *` at runtime; depends on the full
    # Python environment, not just these files. Broad globs used as a safe upper bound.
    "imports": ["src/transformers/**/__init__.py", "src/transformers/**/*.py"],
    # Approximate: ruff applies its own ignore rules from pyproject.toml at runtime.
    "ruff_check": [
        "examples/**/*.py",
        "tests/**/*.py",
        "src/**/*.py",
        "utils/**/*.py",
        "scripts/**/*.py",
        ".circleci/create_circleci_config.py",
        "benchmark/**/*.py",
        "benchmark_v2/**/*.py",
        "setup.py",
        "conftest.py",
    ],
    "ruff_format": [
        "examples/**/*.py",
        "tests/**/*.py",
        "src/**/*.py",
        "utils/**/*.py",
        "scripts/**/*.py",
        ".circleci/create_circleci_config.py",
        "benchmark/**/*.py",
        "benchmark_v2/**/*.py",
        "setup.py",
        "conftest.py",
    ],
}

# Build the registries: discovered modules + inline custom runners.
_discovered_checkers, _discovered_globs = _discover_checkers()

CHECKERS = {**_discovered_checkers, **_INLINE_CHECKERS}
CHECKER_FILE_GLOBS = {**_discovered_globs, **_INLINE_FILE_GLOBS}


def get_checker_cache_globs(checker_name: str) -> list[str] | None:
    """Return the cache inputs for a checker, including its implementation files."""
    globs = CHECKER_FILE_GLOBS.get(checker_name)
    if globs is None:
        return None

    cache_globs = [*globs, str(Path("utils") / Path(__file__).name)]
    script = CHECKERS[checker_name][1]
    if script is not None:
        cache_globs.append(str(Path("utils") / script))
    return cache_globs


class CheckerCache:
    """Disk-backed cache that tracks file content hashes per checker.

    For each checker that declares file globs in CHECKER_FILE_GLOBS, we compute
    a single digest over all matching files.  If the digest matches the stored
    value from the last clean (rc == 0) run, the checker can be skipped.
    """

    def __init__(self, path: Path | None = None):
        self._path = CACHE_PATH if path is None else path
        self._data = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _digest_files(globs: list[str]) -> str:
        """Compute a single SHA-256 over sorted file paths + contents."""
        h = hashlib.sha256()
        paths = set()
        for pattern in globs:
            paths.update(REPO_ROOT.glob(pattern))
        for p in sorted(paths):
            if p.is_file():
                h.update(str(p.relative_to(REPO_ROOT)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def is_current(self, checker_name: str) -> bool:
        """Return True if the checker's files haven't changed since last clean run."""
        globs = get_checker_cache_globs(checker_name)
        if globs is None:
            return False
        return self._data.get(checker_name) == self._digest_files(globs)

    def update(self, checker_name: str) -> None:
        """Record current digest for a checker (call after a clean run)."""
        globs = get_checker_cache_globs(checker_name)
        if globs is None:
            return
        self._data[checker_name] = self._digest_files(globs)

    def invalidate(self, checker_name: str) -> None:
        """Remove a checker from the cache (call after a failed run)."""
        self._data.pop(checker_name, None)


def _file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# ANSI helpers
ORANGE = "\033[38;5;214m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def format_elapsed(seconds: float) -> str:
    """Format a duration for status output."""
    if seconds >= 60:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m{seconds:05.2f}s"
    return f"{seconds:.2f}s"


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

    def finish(self, success, elapsed=None, show_lines=True):
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
            suffix = f" ({format_elapsed(elapsed)})" if elapsed is not None else ""
            if success:
                print(f"{GREEN}✓ {self.label}{suffix}{RESET}")
            else:
                print(f"{RED}✗ {self.label}{suffix}{RESET}")
            # Reprint output lines when we want to preserve the tail summary.
            if show_lines:
                for line in self.lines:
                    print(line)
            sys.stdout.flush()


def _print_output(output: str) -> None:
    """Print captured output without truncation."""
    if not output:
        return

    print(output, end="" if output.endswith("\n") else "\n", flush=True)


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


RUFF_TARGETS = [
    "examples",
    "tests",
    "src",
    "utils",
    "scripts",
    ".circleci/create_circleci_config.py",
    "benchmark",
    "benchmark_v2",
    "setup.py",
    "conftest.py",
]


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
    parser.add_argument("--no-cache", action="store_true", help="Ignore the disk cache and re-run every checker.")

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

    # In --fix mode, drop checkers that have no fix capability (fix_args is None) so
    # they don't print bogus "(0.00s)" lines or inflate the final pass count. Print
    # one transparency line listing what we're skipping.
    if args.fix:
        not_fixable = [n for n in names if CHECKERS[n][3] is None]
        if not_fixable:
            names = [n for n in names if CHECKERS[n][3] is not None]
            print(
                f"Skipping {len(not_fixable)} check-only checker(s) in fix mode: {', '.join(not_fixable)}\n",
                flush=True,
            )

    is_ci = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CIRCLECI") == "true"
    is_tty = sys.stdout.isatty() and not is_ci

    if not is_tty and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    use_cache = not args.no_cache and not args.fix
    cache = CheckerCache() if use_cache else None

    failures = []
    skipped = 0
    total_start = time.perf_counter()
    for name in names:
        label = CHECKERS[name][0]

        # Skip if all relevant files are unchanged since last clean run
        if cache is not None and cache.is_current(name):
            skipped += 1
            if is_tty:
                print(f"{GREEN}✓ {label} (cached){RESET}\n")
            else:
                print(f"{label} (cached)\n", flush=True)
            continue

        cmd_str = get_checker_command(name, fix=args.fix)
        checker_start = time.perf_counter()

        if is_tty:
            window = SlidingWindow(label, max_lines=10)
            if cmd_str:
                window.add_line(f"$ {cmd_str}")
            rc, output = run_checker(name, fix=args.fix, line_callback=window.add_line)
            elapsed = time.perf_counter() - checker_start
            window.finish(success=(rc == 0), elapsed=elapsed, show_lines=(rc == 0))
            if rc != 0:
                print()
                _print_output(output)
            print()
            if rc == 0 and cache is not None:
                cache.update(name)
            elif rc != 0:
                if cache is not None:
                    cache.invalidate(name)
                failures.append(name)
                if not args.keep_going:
                    if cache is not None:
                        cache.save()
                    sys.exit(1)
        else:
            print(f"{label}", flush=True)
            if cmd_str:
                print(f"$ {cmd_str}", flush=True)
            if is_ci:
                streamed_output = []

                def print_line(line):
                    streamed_output.append(line)
                    print(line, end="", flush=True)

                rc, output = run_checker(name, fix=args.fix, line_callback=print_line)
                if rc != 0 and output:
                    streamed_text = "".join(streamed_output)
                    if output.startswith(streamed_text):
                        _print_output(output[len(streamed_text) :])
                    elif output != streamed_text:
                        _print_output(output)
            else:
                rc, output = run_checker(name, fix=args.fix)
                if rc == 0:
                    tail = output.splitlines()[-10:]
                    if tail:
                        print("\n".join(tail), flush=True)
                else:
                    _print_output(output)
            elapsed = time.perf_counter() - checker_start
            status = "OK" if rc == 0 else "FAILED"
            print(f"{status} ({format_elapsed(elapsed)})", flush=True)
            print(flush=True)
            if rc == 0 and cache is not None:
                cache.update(name)
            elif rc != 0:
                if cache is not None:
                    cache.invalidate(name)
                failures.append(name)
                if not args.keep_going:
                    if cache is not None:
                        cache.save()
                    sys.exit(1)

    if cache is not None:
        cache.save()

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", flush=True)
        sys.exit(1)

    total_elapsed = format_elapsed(time.perf_counter() - total_start)
    passed = len(names) - skipped
    if skipped:
        print(f"\nAll {len(names)} checks passed in {total_elapsed} ({passed} ran, {skipped} cached).", flush=True)
    else:
        print(f"\nAll {len(names)} checks passed in {total_elapsed}.", flush=True)


if __name__ == "__main__":
    main()
