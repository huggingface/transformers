import textwrap


def test_conditional_import_ignored(tmp_path):
    code = textwrap.dedent(
        """
        COND = False
        if COND:
            import flash_attn
        import math
        """
    )
    p = tmp_path / "sample.py"
    p.write_text(code)

    from transformers.dynamic_module_utils import get_imports
    result = get_imports(str(p))

    assert "math" in result
    assert "flash_attn" not in result


def test_dynamic_condition_import_ignored(tmp_path):
    code = textwrap.dedent(
        """
        def cond():
            return True

        if cond():
            import os
        """
    )
    p = tmp_path / "sample2.py"
    p.write_text(code)

    from transformers.dynamic_module_utils import get_imports
    result = get_imports(str(p))

    assert "os" not in result


def test_if_true_includes_import(tmp_path):
    code = textwrap.dedent(
        """
        if True:
            import json
        """
    )
    p = tmp_path / "sample3.py"
    p.write_text(code)

    from transformers.dynamic_module_utils import get_imports
    result = get_imports(str(p))

    assert "json" in result
