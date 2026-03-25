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

import ast
import inspect
import os
import sys
import tempfile
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from check_docstrings import (  # noqa: E402
    _auto_docstring_cache,
    _build_ast_indexes,
    _find_typed_dict_classes,
    _get_auto_docstring_names,
    get_default_description,
    has_auto_docstring_decorator,
    replace_default_in_arg_description,
)


class CheckDostringsTested(unittest.TestCase):
    def test_replace_default_in_arg_description(self):
        # Standard docstring with default.
        desc_with_default = "`float`, *optional*, defaults to 2.0"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default, 2.0), "`float`, *optional*, defaults to 2.0"
        )
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default, 1.0), "`float`, *optional*, defaults to 1.0"
        )
        self.assertEqual(replace_default_in_arg_description(desc_with_default, inspect._empty), "`float`")

        # Standard docstring with default but optional is not using the stars.
        desc_with_default_typo = "`float`, `optional`, defaults to 2.0"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default_typo, 2.0), "`float`, *optional*, defaults to 2.0"
        )
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default_typo, 1.0), "`float`, *optional*, defaults to 1.0"
        )

        # If the default is None we do not erase the value in the docstring.
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default, None), "`float`, *optional*, defaults to 2.0"
        )
        # If the default is None (and set as such in the docstring), we do not include it.
        desc_with_default = "`float`, *optional*, defaults to None"
        self.assertEqual(replace_default_in_arg_description(desc_with_default, None), "`float`, *optional*")
        desc_with_default = "`float`, *optional*, defaults to `None`"
        self.assertEqual(replace_default_in_arg_description(desc_with_default, None), "`float`, *optional*")

        # Operations are not replaced, but put in backtiks.
        desc_with_default = "`float`, *optional*, defaults to 1/255"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default, 1 / 255), "`float`, *optional*, defaults to `1/255`"
        )
        desc_with_default = "`float`, *optional*, defaults to `1/255`"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_default, 1 / 255), "`float`, *optional*, defaults to `1/255`"
        )

        desc_with_optional = "`float`, *optional*"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_optional, 2.0), "`float`, *optional*, defaults to 2.0"
        )
        self.assertEqual(
            replace_default_in_arg_description(desc_with_optional, 1.0), "`float`, *optional*, defaults to 1.0"
        )
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, None), "`float`, *optional*")
        self.assertEqual(replace_default_in_arg_description(desc_with_optional, inspect._empty), "`float`")

        desc_with_no_optional = "`float`"
        self.assertEqual(
            replace_default_in_arg_description(desc_with_no_optional, 2.0), "`float`, *optional*, defaults to 2.0"
        )
        self.assertEqual(
            replace_default_in_arg_description(desc_with_no_optional, 1.0), "`float`, *optional*, defaults to 1.0"
        )
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, None), "`float`, *optional*")
        self.assertEqual(replace_default_in_arg_description(desc_with_no_optional, inspect._empty), "`float`")

    def test_get_default_description(self):
        # Fake function to have arguments to test.
        def _fake_function(a, b: int, c=1, d: float = 2.0, e: str = "blob"):
            pass

        params = inspect.signature(_fake_function).parameters
        assert get_default_description(params["a"]) == "`<fill_type>`"
        assert get_default_description(params["b"]) == "`int`"
        assert get_default_description(params["c"]) == "`<fill_type>`, *optional*, defaults to 1"
        assert get_default_description(params["d"]) == "`float`, *optional*, defaults to 2.0"
        assert get_default_description(params["e"]) == '`str`, *optional*, defaults to `"blob"`'


class TestGetAutoDocstringNames(unittest.TestCase):
    """Tests for _get_auto_docstring_names and has_auto_docstring_decorator."""

    def setUp(self):
        _auto_docstring_cache.clear()

    def tearDown(self):
        _auto_docstring_cache.clear()

    def _write_temp(self, source):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write(source)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_detects_simple_decorator(self):
        path = self._write_temp("from transformers import auto_docstring\n\n@auto_docstring\nclass Foo:\n    pass\n")
        names = _get_auto_docstring_names(path)
        self.assertEqual(names, {"Foo"})

    def test_detects_decorator_with_call(self):
        path = self._write_temp("@auto_docstring(custom_args='x')\nclass Bar:\n    pass\n")
        names = _get_auto_docstring_names(path)
        self.assertEqual(names, {"Bar"})

    def test_ignores_other_decorators(self):
        path = self._write_temp("@dataclass\nclass Baz:\n    pass\n")
        names = _get_auto_docstring_names(path)
        self.assertEqual(names, set())

    def test_multiple_classes(self):
        path = self._write_temp(
            "@auto_docstring\nclass A:\n    pass\n\nclass B:\n    pass\n\n@auto_docstring()\ndef func_c():\n    pass\n"
        )
        names = _get_auto_docstring_names(path)
        self.assertEqual(names, {"A", "func_c"})

    def test_caching(self):
        path = self._write_temp("@auto_docstring\nclass X:\n    pass\n")
        result1 = _get_auto_docstring_names(path)
        result2 = _get_auto_docstring_names(path)
        self.assertIs(result1, result2)

    def test_syntax_error_returns_empty(self):
        path = self._write_temp("def broken(\n")
        names = _get_auto_docstring_names(path)
        self.assertEqual(names, set())

    def test_has_auto_docstring_decorator_uses_cache(self):
        from unittest.mock import patch

        path = self._write_temp("@auto_docstring\nclass Cached:\n    pass\n")
        _auto_docstring_cache[path] = {"Cached"}

        # Create classes whose __name__ matches/doesn't match the cache
        Cached = type("Cached", (), {})
        Other = type("Other", (), {})

        with patch.object(inspect, "getfile", return_value=path):
            self.assertTrue(has_auto_docstring_decorator(Cached))
            self.assertFalse(has_auto_docstring_decorator(Other))


class TestBuildAstIndexes(unittest.TestCase):
    """Tests for _build_ast_indexes with pre-parsed tree."""

    def test_finds_decorated_items(self):
        source = (
            "@auto_docstring\n"
            "class MyModel:\n"
            "    def __init__(self, hidden_size=768):\n"
            "        self.hidden_size = hidden_size\n"
        )
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "MyModel")
        self.assertEqual(items[0].kind, "class")
        self.assertIn("hidden_size", items[0].args)

    def test_shared_tree(self):
        source = "@auto_docstring\nclass A:\n    pass\n"
        tree = ast.parse(source)
        items_with_tree = _build_ast_indexes(source, tree=tree)
        items_without = _build_ast_indexes(source)
        self.assertEqual(len(items_with_tree), len(items_without))
        self.assertEqual(items_with_tree[0].name, items_without[0].name)

    def test_no_decorated_items(self):
        source = "class Plain:\n    pass\n"
        items = _build_ast_indexes(source)
        self.assertEqual(items, [])

    def test_function_decorated(self):
        source = "@auto_docstring\ndef my_func(x, y=10):\n    pass\n"
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "my_func")
        self.assertEqual(items[0].kind, "function")
        self.assertIn("x", items[0].args)
        self.assertIn("y", items[0].args)

    def test_custom_args_from_variable(self):
        source = (
            'MY_ARGS = "custom param docs"\n'
            "\n"
            "@auto_docstring(custom_args=MY_ARGS)\n"
            "class WithCustom:\n"
            "    def __init__(self):\n"
            "        pass\n"
        )
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].custom_args_text, "custom param docs")


class TestFindTypedDictClasses(unittest.TestCase):
    """Tests for _find_typed_dict_classes with pre-parsed tree."""

    def test_finds_typed_dict(self):
        source = "from typing import TypedDict\n\nclass MyKwargs(TypedDict):\n    field_a: str\n    field_b: int\n"
        result = _find_typed_dict_classes(source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "MyKwargs")
        self.assertIn("field_a", result[0]["all_fields"])
        self.assertIn("field_b", result[0]["all_fields"])

    def test_shared_tree(self):
        source = "class MyKwargs(TypedDict):\n    x: int\n"
        tree = ast.parse(source)
        r1 = _find_typed_dict_classes(source, tree=tree)
        r2 = _find_typed_dict_classes(source)
        self.assertEqual(len(r1), len(r2))
        self.assertEqual(r1[0]["name"], r2[0]["name"])

    def test_skips_standard_kwargs(self):
        source = "class TextKwargs(TypedDict):\n    field: str\n"
        result = _find_typed_dict_classes(source)
        self.assertEqual(result, [])

    def test_no_typed_dicts(self):
        source = "class Regular:\n    pass\n"
        result = _find_typed_dict_classes(source)
        self.assertEqual(result, [])

    def test_skips_private_fields(self):
        source = "class MyKwargs(TypedDict):\n    public: int\n    _private: str\n"
        result = _find_typed_dict_classes(source)
        self.assertEqual(len(result), 1)
        self.assertIn("public", result[0]["all_fields"])
        self.assertNotIn("_private", result[0]["all_fields"])
