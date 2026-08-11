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
import textwrap
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from check_docstrings import (  # noqa: E402
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
        self.cache = {}

    def _write_temp(self, source):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_detects_simple_decorator(self):
        """Test that a class decorated with @auto_docstring is detected."""
        path = self._write_temp(
            textwrap.dedent("""\
            from transformers import auto_docstring

            @auto_docstring
            class Foo:
                pass
        """)
        )
        names = _get_auto_docstring_names(path, cache=self.cache)
        self.assertEqual(names, {"Foo"})

    def test_detects_decorator_with_call(self):
        """Test that a class decorated with @auto_docstring(args) (called form) is detected."""
        path = self._write_temp(
            textwrap.dedent("""\
            @auto_docstring(custom_args='x')
            class Bar:
                pass
        """)
        )
        names = _get_auto_docstring_names(path, cache=self.cache)
        self.assertEqual(names, {"Bar"})

    def test_ignores_other_decorators(self):
        """Test that classes with non-auto_docstring decorators are not detected."""
        path = self._write_temp(
            textwrap.dedent("""\
            @dataclass
            class Baz:
                pass
        """)
        )
        names = _get_auto_docstring_names(path, cache=self.cache)
        self.assertEqual(names, set())

    def test_multiple_classes(self):
        """Test that only decorated classes and functions are returned when multiple definitions exist."""
        path = self._write_temp(
            textwrap.dedent("""\
            @auto_docstring
            class A:
                pass

            class B:
                pass

            @auto_docstring()
            def func_c():
                pass
        """)
        )
        names = _get_auto_docstring_names(path, cache=self.cache)
        self.assertEqual(names, {"A", "func_c"})

    def test_caching(self):
        """Test that repeated calls for the same file return the cached (identical) result object."""
        path = self._write_temp(
            textwrap.dedent("""\
            @auto_docstring
            class X:
                pass
        """)
        )
        result1 = _get_auto_docstring_names(path, cache=self.cache)
        result2 = _get_auto_docstring_names(path, cache=self.cache)
        self.assertIs(result1, result2)

    def test_syntax_error_returns_empty(self):
        """Test that a file with a syntax error returns an empty set instead of raising."""
        path = self._write_temp("def broken(\n")
        names = _get_auto_docstring_names(path, cache=self.cache)
        self.assertEqual(names, set())

    def test_has_auto_docstring_decorator_uses_cache(self):
        """Test that has_auto_docstring_decorator looks up names from the pre-populated cache."""
        from unittest.mock import patch

        path = self._write_temp(
            textwrap.dedent("""\
            @auto_docstring
            class Cached:
                pass
        """)
        )
        self.cache[path] = {"Cached"}

        # Create classes whose __name__ matches/doesn't match the cache
        Cached = type("Cached", (), {})
        Other = type("Other", (), {})

        with patch.object(inspect, "getfile", return_value=path):
            self.assertTrue(has_auto_docstring_decorator(Cached, cache=self.cache))
            self.assertFalse(has_auto_docstring_decorator(Other, cache=self.cache))


class TestBuildAstIndexes(unittest.TestCase):
    """Tests for _build_ast_indexes with pre-parsed tree."""

    def test_finds_decorated_items(self):
        """Test that _build_ast_indexes finds a decorated class and extracts its __init__ args."""
        source = textwrap.dedent("""\
            @auto_docstring
            class MyModel:
                def __init__(self, hidden_size=768):
                    self.hidden_size = hidden_size
        """)
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "MyModel")
        self.assertEqual(items[0].kind, "class")
        self.assertIn("hidden_size", items[0].args)

    def test_shared_tree(self):
        """Test that passing a pre-parsed AST tree produces the same results as letting the function parse internally."""
        source = textwrap.dedent("""\
            @auto_docstring
            class A:
                pass
        """)
        tree = ast.parse(source)
        items_with_tree = _build_ast_indexes(source, tree=tree)
        items_without = _build_ast_indexes(source)
        self.assertEqual(len(items_with_tree), len(items_without))
        self.assertEqual(items_with_tree[0].name, items_without[0].name)

    def test_no_decorated_items(self):
        """Test that a class without the auto_docstring decorator is not indexed."""
        source = textwrap.dedent("""\
            class Plain:
                pass
        """)
        items = _build_ast_indexes(source)
        self.assertEqual(items, [])

    def test_function_decorated(self):
        """Test that a decorated function is indexed with its arguments."""
        source = textwrap.dedent("""\
            @auto_docstring
            def my_func(x, y=10):
                pass
        """)
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "my_func")
        self.assertEqual(items[0].kind, "function")
        self.assertIn("x", items[0].args)
        self.assertIn("y", items[0].args)

    def test_custom_args_from_variable(self):
        """Test that custom_args passed as a module-level variable are resolved to their string value."""
        source = textwrap.dedent("""\
            MY_ARGS = "custom param docs"

            @auto_docstring(custom_args=MY_ARGS)
            class WithCustom:
                def __init__(self):
                    pass
        """)
        items = _build_ast_indexes(source)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].custom_args_text, "custom param docs")


class TestFindTypedDictClasses(unittest.TestCase):
    """Tests for _find_typed_dict_classes with pre-parsed tree."""

    def test_finds_typed_dict(self):
        """Test that a TypedDict subclass is found and its public fields are extracted."""
        source = textwrap.dedent("""\
            from typing import TypedDict

            class MyKwargs(TypedDict):
                field_a: str
                field_b: int
        """)
        result = _find_typed_dict_classes(source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "MyKwargs")
        self.assertIn("field_a", result[0]["all_fields"])
        self.assertIn("field_b", result[0]["all_fields"])

    def test_shared_tree(self):
        """Test that passing a pre-parsed AST tree produces the same results as internal parsing."""
        source = textwrap.dedent("""\
            class MyKwargs(TypedDict):
                x: int
        """)
        tree = ast.parse(source)
        r1 = _find_typed_dict_classes(source, tree=tree)
        r2 = _find_typed_dict_classes(source)
        self.assertEqual(len(r1), len(r2))
        self.assertEqual(r1[0]["name"], r2[0]["name"])

    def test_skips_standard_kwargs(self):
        """Test that well-known kwargs TypedDicts (e.g. TextKwargs) are excluded from results."""
        source = textwrap.dedent("""\
            class TextKwargs(TypedDict):
                field: str
        """)
        result = _find_typed_dict_classes(source)
        self.assertEqual(result, [])

    def test_no_typed_dicts(self):
        """Test that source with no TypedDict subclasses returns an empty list."""
        source = textwrap.dedent("""\
            class Regular:
                pass
        """)
        result = _find_typed_dict_classes(source)
        self.assertEqual(result, [])

    def test_skips_private_fields(self):
        """Test that fields starting with an underscore are excluded from the extracted TypedDict fields."""
        source = textwrap.dedent("""\
            class MyKwargs(TypedDict):
                public: int
                _private: str
        """)
        result = _find_typed_dict_classes(source)
        self.assertEqual(len(result), 1)
        self.assertIn("public", result[0]["all_fields"])
        self.assertNotIn("_private", result[0]["all_fields"])
