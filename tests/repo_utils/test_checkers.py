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
    def _create_fake_repo(self, tmpdir: str) -> Path:
        """Create a minimal repo layout for exercising checker cache inputs."""
        repo_root = Path(tmpdir)
        (repo_root / "tracked").mkdir()
        (repo_root / "tracked" / "input.txt").write_text("tracked\n", encoding="utf-8")
        (repo_root / "utils").mkdir()
        (repo_root / "utils" / "fake_checker.py").write_text("# fake checker\n", encoding="utf-8")
        return repo_root

    def _run_main(self, *args: str) -> str:
        """Run `checkers.main()` with patched argv/stdout and return captured output."""
        with (
            patch.object(sys, "argv", ["checkers.py", *args]),
            patch.object(sys, "stdout", new=io.StringIO()) as stdout,
        ):
            checkers.main()
            return stdout.getvalue()

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

    def test_main_skips_cached_runs_unless_no_cache_is_used(self):
        """Main should reuse cached results by default and rerun when `--no-cache` is passed."""
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
                self._run_main("demo")
                self.assertEqual(run_checker.call_count, 1)

                output = self._run_main("demo")
                self.assertEqual(run_checker.call_count, 1)
                self.assertIn("(cached)", output)

                self._run_main("demo", "--no-cache")
                self.assertEqual(run_checker.call_count, 2)
