# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformers import testing_utils


class PatchedTestingMethodsOutputFileTest(unittest.TestCase):
    def test_get_output_file_without_xdist_worker(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(os.environ, {"_PATCHED_TESTING_METHODS_OUTPUT_DIR": tmpdir}, clear=True),
        ):
            output_path = testing_utils._get_patched_testing_methods_output_file()

        self.assertEqual(output_path, Path(tmpdir) / "captured_info.txt")

    def test_get_output_file_with_xdist_worker(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "_PATCHED_TESTING_METHODS_OUTPUT_DIR": tmpdir,
                    "PYTEST_XDIST_WORKER": "gw2",
                },
                clear=True,
            ),
        ):
            output_path = testing_utils._get_patched_testing_methods_output_file()

        self.assertEqual(output_path, Path(tmpdir) / "captured_info_gw2.txt")

    def test_prepare_debugging_info_writes_worker_specific_file(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {
                    "_PATCHED_TESTING_METHODS_OUTPUT_DIR": tmpdir,
                    "PYTEST_XDIST_WORKER": "gw1",
                },
                clear=True,
            ),
        ):
            output_path = Path(tmpdir) / "captured_info_gw1.txt"
            rendered_info = testing_utils._prepare_debugging_info("test-info", "payload")
            self.assertEqual(rendered_info, "test-info\n\npayload")
            self.assertTrue(output_path.exists())
            self.assertIn("test-info\n\npayload", output_path.read_text())

    def test_reset_only_clears_current_worker_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_worker_path = Path(tmpdir) / "captured_info_gw0.txt"
            other_worker_path = Path(tmpdir) / "captured_info_gw1.txt"
            current_worker_path.write_text("current worker")
            other_worker_path.write_text("other worker")

            with mock.patch.dict(
                os.environ,
                {
                    "_PATCHED_TESTING_METHODS_OUTPUT_DIR": tmpdir,
                    "PYTEST_XDIST_WORKER": "gw0",
                },
                clear=True,
            ):
                output_path = testing_utils._reset_patched_testing_methods_output_file()
                self.assertEqual(output_path, current_worker_path)
                self.assertFalse(current_worker_path.exists())
                self.assertTrue(other_worker_path.exists())
