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

import io
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
utils_path = os.path.join(git_repo_path, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import checkers  # noqa: E402


@contextmanager
def patch_checkers_paths(repo_root: Path):
    cache_path = repo_root / "utils" / ".checkers_cache.json"
    with ExitStack() as stack:
        stack.enter_context(patch.object(checkers, "REPO_ROOT", repo_root))
        stack.enter_context(patch.object(checkers, "CACHE_PATH", cache_path))
        stack.enter_context(patch.object(checkers, "CHECKERS", {"demo": ("Demo checker", "fake_checker.py", [], [])}))
        stack.enter_context(patch.object(checkers, "CHECKER_FILE_GLOBS", {"demo": ["tracked/**/*.txt"]}))
        yield cache_path


class CheckersCacheTest(unittest.TestCase):
    class _TTYStringIO(io.StringIO):
        def isatty(self) -> bool:
            return True

    def _create_fake_repo(self, tmpdir: str) -> Path:
        """Create a minimal repo layout for exercising checker cache inputs."""
        repo_root = Path(tmpdir)
        (repo_root / "tracked").mkdir()
        (repo_root / "tracked" / "input.txt").write_text("tracked\n", encoding="utf-8")
        (repo_root / "utils").mkdir()
        (repo_root / "utils" / "fake_checker.py").write_text("# fake checker\n", encoding="utf-8")
        return repo_root

    def _run_main(self, *args: str, stdout=None) -> tuple[int | None, str]:
        """Run `checkers.main()` with patched argv/stdout and return the exit code and captured output."""
        stdout = io.StringIO() if stdout is None else stdout
        with (
            patch.object(sys, "argv", ["checkers.py", *args]),
            patch.object(sys, "stdout", new=stdout),
        ):
            exit_code = None
            try:
                checkers.main()
            except SystemExit as e:
                exit_code = e.code
            return exit_code, stdout.getvalue()

    def test_checker_cache_detects_checker_script_changes(self):
        """Cache entries should become stale when the checker implementation file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            with patch_checkers_paths(repo_root) as cache_path:
                cache = checkers.CheckerCache(path=cache_path)
                self.assertFalse(cache.is_current("demo"))

                cache.update("demo")
                self.assertTrue(cache.is_current("demo"))

                (repo_root / "utils" / "fake_checker.py").write_text("# fake checker changed\n", encoding="utf-8")
                self.assertFalse(cache.is_current("demo"))

    def test_main_skips_cached_runs(self):
        """Main should reuse cached results for repeated runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            with (
                patch_checkers_paths(repo_root),
                patch.object(
                    checkers,
                    "run_checker",
                    return_value=(0, "first run"),
                ) as run_checker,
            ):
                exit_code, _ = self._run_main("demo")
                self.assertIsNone(exit_code)
                self.assertEqual(run_checker.call_count, 1)

                exit_code, output = self._run_main("demo")
                self.assertIsNone(exit_code)
                self.assertEqual(run_checker.call_count, 1)
                self.assertIn("(cached)", output)

    def test_main_reruns_with_no_cache(self):
        """Main should rerun when `--no-cache` is passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            with (
                patch_checkers_paths(repo_root),
                patch.object(
                    checkers,
                    "run_checker",
                    side_effect=[(0, "first run"), (0, "forced rerun")],
                ) as run_checker,
            ):
                exit_code, _ = self._run_main("demo")
                self.assertIsNone(exit_code)
                self.assertEqual(run_checker.call_count, 1)

                exit_code, _ = self._run_main("demo", "--no-cache")
                self.assertIsNone(exit_code)
                self.assertEqual(run_checker.call_count, 2)

    def test_main_prints_full_output_on_failure_without_tty(self):
        """Local non-TTY failures should print the full checker output instead of a cropped tail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            output = "\n".join(f"line {i}" for i in range(12)) + "\n"
            with (
                patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "CIRCLECI": "false"}),
                patch_checkers_paths(repo_root),
                patch.object(checkers, "run_checker", return_value=(1, output)),
            ):
                exit_code, stdout = self._run_main("demo", "--keep-going")

            self.assertEqual(exit_code, 1)
            self.assertIn("line 0", stdout)
            self.assertIn("line 11", stdout)

    def test_main_prints_full_output_on_failure_with_tty(self):
        """TTY failures should print the full checker output without reprinting the cropped window tail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            output = "\n".join(f"line {i}" for i in range(12)) + "\n"

            class FakeSlidingWindow:
                def __init__(self, label, max_lines=10):
                    self.label = label
                    self.max_lines = max_lines

                def add_line(self, line):
                    pass

                def finish(self, success, elapsed=None, show_lines=True):
                    print(f"window finished: {self.label} ({success}, {show_lines})")

            with (
                patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "CIRCLECI": "false"}),
                patch_checkers_paths(repo_root),
                patch.object(checkers, "run_checker", return_value=(1, output)),
                patch.object(checkers, "SlidingWindow", FakeSlidingWindow),
            ):
                exit_code, stdout = self._run_main("demo", "--keep-going", stdout=self._TTYStringIO())

            self.assertEqual(exit_code, 1)
            self.assertIn("window finished: Demo checker (False, False)", stdout)
            self.assertIn("line 0", stdout)
            self.assertIn("line 11", stdout)

    def test_main_prints_failure_suffix_in_ci(self):
        """CI failures should still print any extra captured output that was not streamed live."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = self._create_fake_repo(tmpdir)
            streamed_output = "line 0\nline 1\n"
            failure_suffix = "summary line\n"

            def run_checker(name, fix=False, line_callback=None):
                self.assertEqual(name, "demo")
                self.assertFalse(fix)
                self.assertIsNotNone(line_callback)
                for line in streamed_output.splitlines(keepends=True):
                    line_callback(line)
                return 1, streamed_output + failure_suffix

            with (
                patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "CIRCLECI": "false"}),
                patch_checkers_paths(repo_root),
                patch.object(checkers, "run_checker", side_effect=run_checker),
            ):
                exit_code, stdout = self._run_main("demo", "--keep-going")

            self.assertEqual(exit_code, 1)
            self.assertIn("line 0", stdout)
            self.assertIn("line 1", stdout)
            self.assertIn("summary line", stdout)
