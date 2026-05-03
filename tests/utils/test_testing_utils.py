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

from transformers.testing_utils import (
    _clear_patched_testing_methods_output_files,
    _get_patched_testing_methods_output_path,
)


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


class PatchedTestingMethodsOutputPathTester(unittest.TestCase):
    @mock.patch.dict(os.environ, {"_PATCHED_TESTING_METHODS_OUTPUT_DIR": "/tmp/reports"}, clear=True)
    def test_output_path_keeps_legacy_name_without_xdist(self):
        self.assertEqual(_get_patched_testing_methods_output_path(), Path("/tmp/reports/captured_info.txt"))

    @mock.patch.dict(
        os.environ,
        {"_PATCHED_TESTING_METHODS_OUTPUT_DIR": "/tmp/reports", "PYTEST_XDIST_WORKER": "gw1"},
        clear=True,
    )
    def test_output_path_is_worker_specific_with_xdist(self):
        self.assertEqual(_get_patched_testing_methods_output_path(), Path("/tmp/reports/captured_info_gw1.txt"))

    def test_clear_output_files_removes_all_matching_files_without_xdist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "captured_info.txt").write_text("legacy info")
            (tmp_path / "captured_info_gw0.txt").write_text("gw0 info")
            (tmp_path / "summary_short.txt").write_text("FAILED test_example\n")

            with mock.patch.dict(os.environ, {"_PATCHED_TESTING_METHODS_OUTPUT_DIR": tmp_dir}, clear=True):
                _clear_patched_testing_methods_output_files()

            self.assertFalse((tmp_path / "captured_info.txt").exists())
            self.assertFalse((tmp_path / "captured_info_gw0.txt").exists())
            self.assertTrue((tmp_path / "summary_short.txt").exists())


class RetrieveArtifactTester(unittest.TestCase):
    def test_retrieve_artifact_merges_worker_specific_captured_info_files(self):
        notification_service = _load_notification_service_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "captured_info_gw1.txt").write_text("gw1 info")
            (tmp_path / "captured_info_gw0.txt").write_text("gw0 info")
            (tmp_path / "summary_short.txt").write_text("FAILED test_example\n")

            artifact = notification_service.retrieve_artifact(str(tmp_path), gpu="multi")

        self.assertEqual(artifact["summary_short"], "FAILED test_example\n")
        self.assertIn("captured_info_gw0.txt", artifact["captured_info"])
        self.assertIn("gw0 info", artifact["captured_info"])
        self.assertIn("captured_info_gw1.txt", artifact["captured_info"])
        self.assertIn("gw1 info", artifact["captured_info"])
        self.assertNotIn("captured_info_gw0", artifact)
        self.assertNotIn("captured_info_gw1", artifact)

    def test_retrieve_artifact_preserves_legacy_captured_info_file(self):
        notification_service = _load_notification_service_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "captured_info.txt").write_text("legacy info")

            artifact = notification_service.retrieve_artifact(str(tmp_path), gpu=None)

        self.assertEqual(artifact["captured_info"], "legacy info")


if __name__ == "__main__":
    unittest.main()
