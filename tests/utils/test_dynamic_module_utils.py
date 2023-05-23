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
]


@pytest.mark.parametrize("case", CASES)
def test_import_parsing(tmp_path, case):
    tmp_file_path = os.path.join(tmp_path, "test_file.py")
    with open(tmp_file_path, "w") as _tmp_file:
        _tmp_file.write(case)

    parsed_imports = get_imports(tmp_file_path)
    assert parsed_imports == ["os"]
