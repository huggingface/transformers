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

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[2] / "utils"))

import check_decorator_return_types as checker  # noqa: E402


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "PATH_TO_TRANSFORMERS", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "decorator",
    [
        "can_return_tuple",
        "utils.can_return_tuple",
        "capture_outputs()",
        "utils.capture_outputs(tie_last_hidden_states=False)",
    ],
)
def test_fixes_target_decorators(model_files, decorator):
    path = model_files / "modeling_test.py"
    path.write_text(f"@{decorator}\ndef forward() -> tuple | Output | None:\n    pass\n", encoding="utf-8")
    checker.check_decorator_return_types(overwrite=True)
    expected = f"@{decorator}\ndef forward() -> Output | None:\n    pass\n"
    assert path.read_text(encoding="utf-8") == expected
    checker.check_decorator_return_types()
    checker.check_decorator_return_types(overwrite=True)
    assert path.read_text(encoding="utf-8") == expected


def test_preserves_unrelated_functions_and_nested_tuples(model_files):
    source = "@other\ndef forward() -> tuple | Output:\n    pass\n\n@can_return_tuple\ndef encode() -> Output[tuple[int, ...]]:\n    pass\n"
    path = model_files / "modeling_test.py"
    path.write_text(source, encoding="utf-8")
    checker.check_decorator_return_types(overwrite=True)
    assert path.read_text(encoding="utf-8") == source


def test_fixes_unicode_offsets(model_files):
    path = model_files / "modeling_test.py"
    path.write_text('@can_return_tuple\ndef forward(label="é") -> tuple | Output:\n    pass\n', encoding="utf-8")
    checker.check_decorator_return_types(overwrite=True)
    assert path.read_text(encoding="utf-8") == '@can_return_tuple\ndef forward(label="é") -> Output:\n    pass\n'


@pytest.mark.parametrize("annotation", ["", " -> None"])
def test_reports_missing_annotations(model_files, annotation):
    path = model_files / "modeling_test.py"
    path.write_text(f"@can_return_tuple\ndef forward(){annotation}:\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have a non-None return annotation"):
        checker.check_decorator_return_types(overwrite=True)


def test_generated_files_are_checked_but_not_overwritten(model_files):
    path = model_files / "modeling_test.py"
    source = "# Header\n# This file was automatically generated from modular_test.py.\n@can_return_tuple\ndef forward() -> tuple | Output:\n    pass\n"
    path.write_text(source, encoding="utf-8")
    checker.check_decorator_return_types(overwrite=True)
    assert path.read_text(encoding="utf-8") == source
    with pytest.raises(ValueError, match="must not be annotated with a union that includes 'tuple'"):
        checker.check_decorator_return_types()


def test_modular_super_inherits_only_missing_annotations(model_files):
    path = model_files / "modular_test.py"
    path.write_text("@can_return_tuple\ndef forward():\n    return super().forward()\n", encoding="utf-8")
    checker.check_decorator_return_types()
    path.write_text(
        "@can_return_tuple\ndef forward() -> tuple | Output:\n    return super().forward()\n", encoding="utf-8"
    )
    checker.check_decorator_return_types(overwrite=True)
    assert "-> Output:" in path.read_text(encoding="utf-8")


def test_modeling_super_requires_annotation(model_files):
    path = model_files / "modeling_test.py"
    path.write_text("@can_return_tuple\ndef forward():\n    return super().forward()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have a non-None return annotation"):
        checker.check_decorator_return_types()
