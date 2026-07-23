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

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from transformers import testing_utils


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_notification_service_module():
    module_path = REPO_ROOT / "utils" / "notification_service.py"
    spec = importlib.util.spec_from_file_location("notification_service_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    stub_modules = {
        "compare_test_runs": types.SimpleNamespace(compare_job_sets=lambda *args, **kwargs: None),
        "get_ci_error_statistics": types.SimpleNamespace(get_jobs=lambda *args, **kwargs: []),
        "get_previous_daily_ci": types.SimpleNamespace(
            get_last_daily_ci_reports=lambda *args, **kwargs: None,
            get_last_daily_ci_run=lambda *args, **kwargs: None,
            get_last_daily_ci_workflow_run_id=lambda *args, **kwargs: None,
        ),
        "huggingface_hub": types.SimpleNamespace(HfApi=object),
        "slack_sdk": types.SimpleNamespace(WebClient=object),
    }
    with mock.patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


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


class RetrieveArtifactCapturedInfoTest(unittest.TestCase):
    def test_retrieve_artifact_preserves_legacy_captured_info_file(self):
        notification_service = _load_notification_service_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "captured_info.txt").write_text("legacy info")

            artifact = notification_service.retrieve_artifact(str(artifact_dir), gpu=None)

        self.assertEqual(artifact["captured_info"], "legacy info")

    def test_retrieve_artifact_merges_worker_specific_captured_info_files(self):
        notification_service = _load_notification_service_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "captured_info_gw1.txt").write_text("gw1 info")
            (artifact_dir / "captured_info_gw0.txt").write_text("gw0 info")
            (artifact_dir / "summary_short.txt").write_text("FAILED test_example\n")

            artifact = notification_service.retrieve_artifact(str(artifact_dir), gpu="multi")

        self.assertEqual(artifact["summary_short"], "FAILED test_example\n")
        self.assertIn("captured_info_gw0.txt", artifact["captured_info"])
        self.assertIn("gw0 info", artifact["captured_info"])
        self.assertIn("captured_info_gw1.txt", artifact["captured_info"])
        self.assertIn("gw1 info", artifact["captured_info"])
        self.assertNotIn("captured_info_gw0", artifact)
        self.assertNotIn("captured_info_gw1", artifact)
