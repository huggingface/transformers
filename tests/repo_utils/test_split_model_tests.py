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

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from split_model_tests import COVERED_BY_DEDICATED_JOB, NO_TEST_FILES  # noqa: E402


SCRIPT = Path(git_repo_path) / "utils" / "split_model_tests.py"
TESTS_DIR = Path(git_repo_path) / "tests"


class SplitModelTestsTester(unittest.TestCase):
    def test_default_run_skips_the_duplicated_directories(self):
        """
        The default run is what every CI caller uses, so it is the one that must drop the directories a dedicated job
        already covers. An inverted flag or a moved filter shows up here instead of as a silently doubled daily CI.
        """
        command = [sys.executable, str(SCRIPT), "--num_splits", "1"]
        stdout = subprocess.run(command, cwd=TESTS_DIR, capture_output=True, check=True, text=True).stdout
        folders = [folder for split in ast.literal_eval(stdout.strip()) for folder in split]

        for folder in [*COVERED_BY_DEDICATED_JOB, *NO_TEST_FILES]:
            self.assertNotIn(folder, folders, f"`tests/{folder}` should not be in the auto-discovered folder list")
        # `models/<model>` entries are never filtered, whatever a model is called.
        self.assertIn("models/bert", folders)

    def test_no_test_files_dirs_really_hold_no_test_file(self):
        """`NO_TEST_FILES` claims a directory collects nothing. Prove it, so the list cannot silently diverge."""
        for folder in NO_TEST_FILES:
            path = TESTS_DIR / folder
            self.assertTrue(path.is_dir(), f"`NO_TEST_FILES` names `{folder}`, which is not a directory under `tests`")
            offenders = sorted(str(p.relative_to(TESTS_DIR)) for p in path.rglob("test_*.py"))
            self.assertEqual(
                offenders,
                [],
                f"`tests/{folder}` is in `NO_TEST_FILES` but holds test files: {offenders}. Those tests never run.",
            )

    def test_covered_dirs_exist_under_tests(self):
        """A typo here skips nothing, so the duplicate run it was meant to remove keeps happening unnoticed."""
        for folder in COVERED_BY_DEDICATED_JOB:
            self.assertTrue(
                (TESTS_DIR / folder).is_dir(),
                f"`COVERED_BY_DEDICATED_JOB` names `{folder}`, which is not a directory under `tests`",
            )
