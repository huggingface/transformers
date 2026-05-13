# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from pathlib import Path

import pytest

from transformers import dynamic_module_utils
from transformers.dynamic_module_utils import get_cached_module_file, get_imports


TOP_LEVEL_IMPORT = """
import os
"""

IMPORT_IN_FUNCTION = """
def foo():
    import os
    return False
"""

DEEPLY_NESTED_IMPORT = """
def foo():
    def bar():
        if True:
            import os
        return False
    return bar()
"""

TOP_LEVEL_TRY_IMPORT = """
import os

try:
    import bar
except ImportError:
    raise ValueError()
"""

TRY_IMPORT_IN_FUNCTION = """
import os

def foo():
    try:
        import bar
    except ImportError:
        raise ValueError()
"""

MULTIPLE_EXCEPTS_IMPORT = """
import os

try:
    import bar
except (ImportError, AttributeError):
    raise ValueError()
"""

EXCEPT_AS_IMPORT = """
import os

try:
    import bar
except ImportError as e:
    raise ValueError()
"""

GENERIC_EXCEPT_IMPORT = """
import os

try:
    import bar
except:
    raise ValueError()
"""

MULTILINE_TRY_IMPORT = """
import os

try:
    import bar
    import baz
except ImportError:
    raise ValueError()
"""

MULTILINE_BOTH_IMPORT = """
import os

try:
    import bar
    import baz
except ImportError:
    x = 1
    raise ValueError()
"""

CASES = [
    TOP_LEVEL_IMPORT,
    IMPORT_IN_FUNCTION,
    DEEPLY_NESTED_IMPORT,
    TOP_LEVEL_TRY_IMPORT,
    GENERIC_EXCEPT_IMPORT,
    MULTILINE_TRY_IMPORT,
    MULTILINE_BOTH_IMPORT,
    MULTIPLE_EXCEPTS_IMPORT,
    EXCEPT_AS_IMPORT,
    TRY_IMPORT_IN_FUNCTION,
]


@pytest.mark.parametrize("case", CASES)
def test_import_parsing(tmp_path, case):
    tmp_file_path = os.path.join(tmp_path, "test_file.py")
    with open(tmp_file_path, "w") as _tmp_file:
        _tmp_file.write(case)

    parsed_imports = get_imports(tmp_file_path)
    assert parsed_imports == ["os"]


def _create_local_module(module_dir: Path, module_code: str, helper_code: str | None = None):
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "custom_model.py").write_text(module_code, encoding="utf-8")
    if helper_code is not None:
        (module_dir / "helper.py").write_text(helper_code, encoding="utf-8")


def test_get_cached_module_file_local_cache_key_uses_basename_and_content_hash(monkeypatch, tmp_path):
    modules_cache = tmp_path / "hf_modules_cache"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache))

    model_dir_a = tmp_path / "pretrained_a" / "subdir"
    model_dir_b = tmp_path / "pretrained_b" / "subdir"
    model_dir_c = tmp_path / "pretrained_c" / "subdir"

    _create_local_module(model_dir_a, 'MAGIC = "A"\n')
    _create_local_module(model_dir_b, 'MAGIC = "B"\n')
    _create_local_module(model_dir_c, 'MAGIC = "A"\n')

    cached_module_a = get_cached_module_file(str(model_dir_a), "custom_model.py")
    cached_module_b = get_cached_module_file(str(model_dir_b), "custom_model.py")
    cached_module_c = get_cached_module_file(str(model_dir_c), "custom_model.py")

    cached_module_path_a = Path(cached_module_a)
    assert cached_module_path_a.parent.parent.name == "subdir"
    assert len(cached_module_path_a.parent.name) == 16
    assert cached_module_a != cached_module_b
    assert cached_module_a == cached_module_c


def test_get_cached_module_file_local_cache_key_includes_relative_import_sources(monkeypatch, tmp_path):
    modules_cache = tmp_path / "hf_modules_cache"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache))

    model_dir_a = tmp_path / "pretrained_a" / "subdir"
    model_dir_b = tmp_path / "pretrained_b" / "subdir"

    module_code = "from .helper import MAGIC\nVALUE = MAGIC\n"
    _create_local_module(model_dir_a, module_code, 'MAGIC = "A"\n')
    _create_local_module(model_dir_b, module_code, 'MAGIC = "B"\n')

    cached_module_a = get_cached_module_file(str(model_dir_a), "custom_model.py")
    cached_module_b = get_cached_module_file(str(model_dir_b), "custom_model.py")

    cached_helper_a = modules_cache / Path(cached_module_a).parent / "helper.py"
    cached_helper_b = modules_cache / Path(cached_module_b).parent / "helper.py"

    assert cached_module_a != cached_module_b
    assert cached_helper_a.read_text(encoding="utf-8") == 'MAGIC = "A"\n'
    assert cached_helper_b.read_text(encoding="utf-8") == 'MAGIC = "B"\n'


def test_get_cached_module_file_local_cache_key_keeps_hash_stable_with_different_basenames(monkeypatch, tmp_path):
    modules_cache = tmp_path / "hf_modules_cache"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(modules_cache))

    model_dir_a = tmp_path / "pretrained_a" / "alpha_subdir"
    model_dir_b = tmp_path / "pretrained_b" / "beta_subdir"

    _create_local_module(model_dir_a, 'MAGIC = "A"\n')
    _create_local_module(model_dir_b, 'MAGIC = "A"\n')

    cached_module_a = Path(get_cached_module_file(str(model_dir_a), "custom_model.py"))
    cached_module_b = Path(get_cached_module_file(str(model_dir_b), "custom_model.py"))

    assert cached_module_a.parent.parent.name == "alpha_subdir"
    assert cached_module_b.parent.parent.name == "beta_subdir"
    assert cached_module_a.parent.name == cached_module_b.parent.name
