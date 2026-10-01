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

import os
import sys
import unittest
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
utils_path = os.path.join(git_repo_path, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import check_modular_conversion  # noqa: E402


class ConverterChangedInDiffTest(unittest.TestCase):
    """Regression guard for PR #45492: changes to the converter alone must force a full check."""

    def _patch_modified(self, files):
        return patch.object(check_modular_conversion, "_get_modified_files", return_value=files)

    def test_returns_true_when_modular_model_converter_changed(self):
        with self._patch_modified(
            [
                "utils/modular_model_converter.py",
                "src/transformers/models/llava_onevision/modular_llava_onevision.py",
            ]
        ):
            self.assertTrue(check_modular_conversion.converter_changed_in_diff())

    def test_returns_true_when_create_dependency_mapping_changed(self):
        with self._patch_modified(["utils/create_dependency_mapping.py"]):
            self.assertTrue(check_modular_conversion.converter_changed_in_diff())

    def test_returns_false_for_model_only_diff(self):
        with self._patch_modified(
            [
                "src/transformers/models/llama/modular_llama.py",
                "src/transformers/models/llama/modeling_llama.py",
            ]
        ):
            self.assertFalse(check_modular_conversion.converter_changed_in_diff())

    def test_returns_false_for_unrelated_utils_change(self):
        with self._patch_modified(["utils/check_modular_conversion.py", "utils/check_copies.py"]):
            self.assertFalse(check_modular_conversion.converter_changed_in_diff())

    def test_converter_files_set_includes_expected_entries(self):
        # Keep the allow-list grounded: if either file is renamed/removed, this test fails loudly
        # so the detection logic is updated alongside the rename.
        self.assertIn("utils/modular_model_converter.py", check_modular_conversion.CONVERTER_FILES)
        self.assertIn("utils/create_dependency_mapping.py", check_modular_conversion.CONVERTER_FILES)
        for rel_path in check_modular_conversion.CONVERTER_FILES:
            self.assertTrue(
                os.path.exists(os.path.join(git_repo_path, rel_path)),
                f"{rel_path} listed in CONVERTER_FILES but does not exist on disk",
            )


if __name__ == "__main__":
    unittest.main()
