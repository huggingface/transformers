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

import pytest

from transformers.dynamic_module_utils import get_imports


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
