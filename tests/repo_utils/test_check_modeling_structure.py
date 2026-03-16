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
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_modeling_structure as cms  # noqa: E402


class CheckModelingStructureTest(unittest.TestCase):
    # --- TRF001: config_class naming consistency (old TRF003) ---

    def test_trf001_valid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF001})
        trf001 = [v for v in violations if v.rule_id == cms.TRF001]
        self.assertEqual(trf001, [])

    def test_trf001_invalid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = BarConfig
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF001})
        trf001 = [v for v in violations if v.rule_id == cms.TRF001]
        self.assertEqual(len(trf001), 1)
        self.assertIn("config_class is BarConfig, expected FooConfig", trf001[0].message)

    # --- TRF002: base_model_prefix (old TRF004) ---

    def test_trf002_valid_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF002})
        trf002 = [v for v in violations if v.rule_id == cms.TRF002]
        self.assertEqual(trf002, [])

    def test_trf002_invalid_empty_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = ""
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF002})
        trf002 = [v for v in violations if v.rule_id == cms.TRF002]
        self.assertEqual(len(trf002), 1)
        self.assertIn("non-empty canonical token", trf002[0].message)

    # --- TRF003: capture_output enforcement (reworked old TRF005) ---

    def test_trf003_flags_old_return_dict_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        if not return_dict:
            return (x,)
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF003})
        trf003 = [v for v in violations if v.rule_id == cms.TRF003]
        self.assertEqual(len(trf003), 1)
        self.assertIn("old return_dict branching pattern", trf003[0].message)

    def test_trf003_allows_no_return_dict_arg(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF003})
        trf003 = [v for v in violations if v.rule_id == cms.TRF003]
        self.assertEqual(trf003, [])

    def test_trf003_allows_return_dict_without_branching(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, x, return_dict=None):
        return x
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF003})
        trf003 = [v for v in violations if v.rule_id == cms.TRF003]
        self.assertEqual(trf003, [])

    # --- TRF004: tie_weights hard ban (reworked old TRF007) ---

    def test_trf004_flags_any_tie_weights_override(self):
        source = """
class FooModel:
    def tie_weights(self):
        super().tie_weights()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF004})
        trf004 = [v for v in violations if v.rule_id == cms.TRF004]
        self.assertEqual(len(trf004), 1)
        self.assertIn("overrides tie_weights", trf004[0].message)

    def test_trf004_allows_no_tie_weights(self):
        source = """
class FooModel:
    _tied_weights_keys = ["lm_head.weight"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF004})
        trf004 = [v for v in violations if v.rule_id == cms.TRF004]
        self.assertEqual(trf004, [])

    # --- TRF005: _no_split_modules (old TRF008) ---

    def test_trf005_valid_no_split_modules(self):
        source = """
class FooModel:
    _no_split_modules = ["FooDecoderLayer"]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF005})
        trf005 = [v for v in violations if v.rule_id == cms.TRF005]
        self.assertEqual(trf005, [])

    def test_trf005_invalid_empty_string(self):
        source = """
class FooModel:
    _no_split_modules = [""]
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF005})
        trf005 = [v for v in violations if v.rule_id == cms.TRF005]
        self.assertEqual(len(trf005), 1)

    # --- TRF006: cache args usage (old TRF010) ---

    def test_trf006_catches_unused_cache_args(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF006})
        trf006 = [v for v in violations if v.rule_id == cms.TRF006]
        self.assertEqual(len(trf006), 1)
        self.assertIn("past_key_values/use_cache", trf006[0].message)

    # --- TRF007: post_init order (old TRF011) ---

    def test_trf007_flags_assignment_after_post_init(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        self.proj = None
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF007})
        trf007 = [v for v in violations if v.rule_id == cms.TRF007]
        self.assertEqual(len(trf007), 1)
        self.assertIn("assigns self.* after self.post_init()", trf007[0].message)

    def test_trf007_allows_post_init_at_end(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.proj = None
        self.post_init()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF007})
        trf007 = [v for v in violations if v.rule_id == cms.TRF007]
        self.assertEqual(trf007, [])

    # --- TRF009: cross-model imports (old TRF013) ---

    def test_trf009_flags_cross_model_import_in_modeling_file(self):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF009})
        trf009 = [v for v in violations if v.rule_id == cms.TRF009]
        self.assertEqual(len(trf009), 1)
        self.assertIn("imports implementation code from `llama`", trf009[0].message)

    def test_trf009_allows_same_model_import_in_modeling_file(self):
        source = """
from .configuration_foo import FooConfig
from transformers.models.foo.configuration_foo import FooConfig as FooConfigAlias
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF009})
        trf009 = [v for v in violations if v.rule_id == cms.TRF009]
        self.assertEqual(trf009, [])

    def test_trf009_ignores_modular_files(self):
        source = """
from transformers.models.llama.modeling_llama import LlamaAttention
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = cms.analyze_file(file_path, source, enabled_rules={cms.TRF009})
        trf009 = [v for v in violations if v.rule_id == cms.TRF009]
        self.assertEqual(trf009, [])

    # --- Utility tests ---

    @patch("check_modeling_structure.subprocess.run")
    def test_get_changed_modeling_files_filters_non_model_files(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout=(
                    "src/transformers/models/foo/modeling_foo.py\n"
                    "src/transformers/models/foo/modular_foo.py\n"
                    "src/transformers/models/foo/configuration_foo.py\n"
                    "docs/source/en/index.md\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff", "--cached"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "ls-files"], returncode=0, stdout="", stderr=""),
        ]
        changed_files = cms.get_changed_modeling_files("origin/main")
        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/foo/modeling_foo.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
            },
        )

    @patch("check_modeling_structure.subprocess.run")
    def test_get_changed_modeling_files_includes_uncommitted_worktree_changes(self, mock_run):
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=0,
                stdout="src/transformers/models/helium/modeling_helium.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "diff", "--cached"],
                returncode=0,
                stdout="src/transformers/models/foo/modular_foo.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["git", "ls-files"],
                returncode=0,
                stdout=("src/transformers/models/bar/modeling_bar.py\ndocs/source/en/index.md\n"),
                stderr="",
            ),
        ]

        changed_files = cms.get_changed_modeling_files("origin/main")

        self.assertEqual(
            changed_files,
            {
                Path("src/transformers/models/helium/modeling_helium.py"),
                Path("src/transformers/models/foo/modular_foo.py"),
                Path("src/transformers/models/bar/modeling_bar.py"),
            },
        )


if __name__ == "__main__":
    unittest.main()
