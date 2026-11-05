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
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
utils_path = os.path.join(git_repo_path, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import check_repo  # noqa: E402


class RecordingNamespace:
    """Record directory listings and attribute access for cache tests."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.dir_calls = 0
        self.getattr_calls = []

    def __dir__(self):
        self.dir_calls += 1
        return list(self._mapping.keys()) + ["__doc__"]

    def __getattr__(self, name):
        self.getattr_calls.append(name)
        try:
            return self._mapping[name]
        except KeyError as error:
            raise AttributeError(name) from error


@contextmanager
def patch_transformers_path(path: Path):
    """Temporarily point `check_repo` at a temporary transformers source tree."""
    old_path = check_repo.PATH_TO_TRANSFORMERS
    check_repo.PATH_TO_TRANSFORMERS = str(path)
    try:
        yield
    finally:
        check_repo.PATH_TO_TRANSFORMERS = old_path


class CheckRepoTest(unittest.TestCase):
    def setUp(self):
        """Reset the `get_model_modules` cache before each test."""
        check_repo.get_model_modules.cache_clear()
        self.addCleanup(check_repo.get_model_modules.cache_clear)

    def _write_modeling_file(self, root: Path, model_name: str, content: str) -> None:
        """Create a temporary `modeling_*.py` file used by `check_models_have_kwargs`."""
        model_dir = root / "src" / "transformers" / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"modeling_{model_name}.py").write_text(content, encoding="utf-8")

    def test_get_model_modules_is_cached(self):
        """Repeated calls should reuse the cached module list instead of traversing models twice."""
        alpha_modeling = object()
        alpha_module = RecordingNamespace({"modeling_alpha": alpha_modeling, "configuration_alpha": object()})
        fake_models = RecordingNamespace({"alpha": alpha_module, "deprecated_alpha": object()})
        fake_transformers = SimpleNamespace(models=fake_models)

        with patch.object(check_repo, "transformers", fake_transformers):
            first = check_repo.get_model_modules()
            second = check_repo.get_model_modules()

        self.assertIs(first, second)
        self.assertEqual(first, [alpha_modeling])
        self.assertEqual(fake_models.dir_calls, 1)
        self.assertEqual(fake_models.getattr_calls, ["alpha"])
        self.assertEqual(alpha_module.dir_calls, 1)
        self.assertEqual(alpha_module.getattr_calls, ["modeling_alpha"])

    def test_check_models_have_kwargs_ignores_nested_classes(self):
        """Nested helper classes should not trigger missing-`**kwargs` failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_modeling_file(
                root,
                "foo",
                """
class PreTrainedModel:
    pass


class FooModel(PreTrainedModel):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class HelperContainer:
    class NestedModel(PreTrainedModel):
        def forward(self, hidden_states):
            return hidden_states
""".strip()
                + "\n",
            )

            with patch_transformers_path(root / "src" / "transformers"):
                check_repo.check_models_have_kwargs()

    def test_check_models_have_kwargs_still_checks_top_level_models(self):
        """Top-level model classes should still fail when `forward()` omits `**kwargs`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_modeling_file(
                root,
                "foo",
                """
class PreTrainedModel:
    pass


class FooModel(PreTrainedModel):
    def forward(self, hidden_states):
        return hidden_states


def make_nested_model():
    class NestedModel(PreTrainedModel):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    return NestedModel
""".strip()
                + "\n",
            )

            with patch_transformers_path(root / "src" / "transformers"):
                with self.assertRaisesRegex(Exception, "FooModel"):
                    check_repo.check_models_have_kwargs()
