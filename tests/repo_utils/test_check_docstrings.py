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

import inspect
import os
import sys
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from check_docstrings import get_default_description, replace_default_in_arg_description  # noqa: E402


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
