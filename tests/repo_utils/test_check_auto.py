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
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
utils_path = os.path.join(git_repo_path, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import check_auto  # noqa: E402


@contextmanager
def cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _write_config(root: Path, module_name: str, classes: list[tuple[str, str]]) -> None:
    """Create `src/transformers/models/<module>/configuration_<module>.py` with the given classes.

    `classes` is a list of (class_name, model_type) pairs, all subclassing PreTrainedConfig.
    """
    module_dir = root / "src" / "transformers" / "models" / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    body = "from transformers import PreTrainedConfig\n\n"
    for cls_name, model_type in classes:
        body += textwrap.dedent(
            f'''
            class {cls_name}(PreTrainedConfig):
                model_type = "{model_type}"
            '''
        )
    (module_dir / f"configuration_{module_name}.py").write_text(body, encoding="utf-8")


class BuildConfigMappingNamesTest(unittest.TestCase):
    """Tests for the natural-match tie-break in `check_auto.build_config_mapping_names`.

    A natural match is one where a config's `model_type` equals its module directory name
    (e.g. `DetrConfig` with `model_type = "detr"` inside `models/detr/`). When two classes
    share a `model_type`, the natural one must always win regardless of filesystem ordering.
    """

    def test_single_natural_match(self):
        """Baseline: one config in its eponymous module → no special mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, "detr", [("DetrConfig", "detr")])
            with cwd(root):
                model_type_map, special_mappings = check_auto.build_config_mapping_names()

        self.assertEqual(model_type_map, {"detr": "DetrConfig"})
        self.assertEqual(special_mappings, {})

    def test_natural_wins_when_encountered_first(self):
        """detr (natural) is alphabetically before maskformer (non-natural for model_type=detr).

        This is the order modern Linux filesystems produce. The natural match must be kept
        and the alias must not appear in the canonical mapping.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, "detr", [("DetrConfig", "detr")])
            _write_config(
                root,
                "maskformer",
                [("MaskFormerConfig", "maskformer"), ("MaskFormerDetrConfig", "detr")],
            )
            with cwd(root):
                model_type_map, special_mappings = check_auto.build_config_mapping_names()

        self.assertEqual(model_type_map["detr"], "DetrConfig")
        self.assertEqual(model_type_map["maskformer"], "MaskFormerConfig")
        self.assertNotIn("detr", special_mappings)

    def test_natural_wins_when_encountered_second(self):
        """The non-natural alias is alphabetically *before* the natural module.

        Without the prefer-natural logic this is the case that breaks: the alias would be
        recorded first and then never overwritten. The fix must still pick the natural class
        and clear the now-stale special mapping.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # `aaa_alias` sorts before `foo`, so its non-natural class is processed first.
            _write_config(root, "aaa_alias", [("AaaConfig", "aaa_alias"), ("FooAliasConfig", "foo")])
            _write_config(root, "foo", [("FooConfig", "foo")])
            with cwd(root):
                model_type_map, special_mappings = check_auto.build_config_mapping_names()

        self.assertEqual(model_type_map["foo"], "FooConfig")
        self.assertNotIn("foo", special_mappings, "stale alias entry must be cleared")
        # The alias module's own natural entry is still recorded.
        self.assertEqual(model_type_map["aaa_alias"], "AaaConfig")

    def test_non_natural_only_records_special_mapping(self):
        """If a model_type has no natural match, the alias is the canonical entry."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, "wrapper", [("WrapperConfig", "wrapper"), ("InnerConfig", "inner")])
            with cwd(root):
                model_type_map, special_mappings = check_auto.build_config_mapping_names()

        self.assertEqual(model_type_map["inner"], "InnerConfig")
        self.assertEqual(special_mappings["inner"], "wrapper")
