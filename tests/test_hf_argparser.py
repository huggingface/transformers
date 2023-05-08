# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
from collections import namedtuple
import dataclasses
import enum
import os.path
import sys
from unittest import mock

from parameterized import parameterized
from transformers import hf_argparser
from transformers.testing_utils import TestCasePlus
import yaml


class TestHfArgparserGlobals(TestCasePlus):
    @parameterized.expand(
        [
            ("bool_true", True, True),
            ("bool_false", False, False),
            ("bool_str_yes", "yes", True),
            ("bool_str_true", "trUE", True),
            ("bool_str_t", "t", True),
            ("bool_str_y", "y", True),
            ("bool_str_1", "1", True),
            ("bool_str_no", "no", False),
            ("bool_str_false", "faLSe", False),
            ("bool_str_f", "f", False),
            ("bool_str_n", "n", False),
            ("bool_str_0", "0", False),
        ]
    )
    def test_string_to_bool(self, name: str, value: bool | str, expected: bool):
        del name  # unused
        self.assertEqual(hf_argparser.string_to_bool(value), expected)

    def test_string_to_bool_raises_error(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "Truthy value expected"):
            hf_argparser.string_to_bool("what")

    def test_make_choice_type_function(self):
        choices = [1, "two", True]

        choice_func = hf_argparser.make_choice_type_function(choices)

        self.assertEqual(choice_func("1"), 1)
        self.assertEqual(choice_func("two"), "two")
        self.assertEqual(choice_func("True"), True)

    def test_HfArg(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: str = hf_argparser.HfArg(
                default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!"
            )

        test_instance = TestClass()
        self.assertEqual(test_instance.hf_arg, "Huggingface")
        fields = dataclasses.fields(test_instance)
        self.assertEqual(len(fields), 1)
        field = fields[0]
        self.assertEqual(field.metadata["aliases"], ["--example", "-e"])
        self.assertEqual(field.metadata["help"], "What a nice syntax!")


class EnumTest(enum.Enum):
    UNSET = 0
    A = 1
    B = 2


class TestHfArgumentParser(TestCasePlus):
    argparser: hf_argparser.HfArgumentParser
    tmp_dir: str
    args_file: str

    def setUp(self):
        super().setUp()

        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str", aliases=["--str", "-s"])
            hf_arg_int: int = hf_argparser.HfArg(default=2, aliases=["--int", "-i"])
            hf_arg_bool: bool = hf_argparser.HfArg(default=False, aliases=["-b"])
            hf_arg_opt_bool: bool | None = hf_argparser.HfArg(default=None, aliases=["-ob"])
            hf_arg_enum: EnumTest = hf_argparser.HfArg(default=EnumTest.UNSET, aliases=["-e"])

        self.argparser = hf_argparser.HfArgumentParser(TestClass)

        # Add a test args file.
        self.tmp_dir = self.get_auto_remove_tmp_dir(after=True)
        self.args_file = os.path.join(self.tmp_dir, "argsfile.args")
        with open(self.args_file, mode="w", encoding="utf-8") as f:
            f.write(
                """
                --str=file_str
                --int=5173
                """
            )

        # Mock out sys.argv so that it can be tested.

    TestTuple = namedtuple(
        "TestTuple",
        [
            "name",
            "args",
            "expected_str",
            "expected_int",
            # Optional:
            "return_remaining_strings",  # defaults to False.
            "expected_remaining_strings",  # defaults to None.
            "args_filename",  # defaults to False. If True, gets set to self.args_file.
            "args_file_flag",  # defaults to "". If set, adds {self.args_file} to args.
        ],
        defaults=[False, None, False, ""],
    )

    @parameterized.expand(
        [
            TestTuple("defaults", [], "default_str", 2),
            TestTuple("flags", ["--str", "arg_str", "--int", "11"], "arg_str", 11),
            TestTuple("aliases", ["-s", "arg_str", "-i", "11"], "arg_str", 11),
            TestTuple(
                "aliases",
                ["extra"],
                "default_str",
                2,
                return_remaining_strings=True,
                expected_remaining_strings=["extra"],
            ),
            TestTuple("args_filename", [], "file_str", 5173, args_filename=True),
            TestTuple("args_filename_overridden_by_args", ["-s", "override"], "override", 5173, args_filename=True),
            TestTuple("args_file_flag", ["--args"], "file_str", 5173, args_file_flag="--args"),
            TestTuple(
                "args_file_flag_uses_only_last_flag",
                ["--args=ignore", "--args"],
                "file_str",
                5173,
                args_file_flag="--args",
            ),
        ]
    )
    def test_parse_args_into_dataclasses(
        self,
        name: str,
        args: list[str],
        expected_str: str,
        expected_int: int,
        return_remaining_strings: bool,
        expected_remaining_strings: list[str] | None,
        args_filename: bool,
        args_file_flag: str,
    ):
        del name  # unused
        args_file = self.args_file if args_filename else ""
        if args_file_flag:
            args.append(self.args_file)

        instances = self.argparser.parse_args_into_dataclasses(
            args=args,
            return_remaining_strings=return_remaining_strings,
            args_filename=args_file,
            args_file_flag=args_file_flag,
        )

        self.assertEqual(instances[0].hf_arg_str, expected_str)
        self.assertEqual(instances[0].hf_arg_int, expected_int)
        self.assertEqual(instances[0].hf_arg_bool, False)
        self.assertEqual(instances[0].hf_arg_opt_bool, None)
        self.assertEqual(instances[0].hf_arg_enum, EnumTest.UNSET)
        if return_remaining_strings:
            self.assertEqual(instances[1], expected_remaining_strings)

    def test_parse_args_into_dataclasses_remaining_strings_raises_error(self):
        with self.assertRaisesRegex(ValueError, "arguments are not used"):
            self.argparser.parse_args_into_dataclasses(args=["extra"])

    @mock.patch.object(sys, "argv", [])
    def test_parse_args_into_dataclasses_look_for_args_file(self):
        prefix = os.path.join(os.path.dirname(self.args_file), "argsfile")
        sys.argv = [prefix]

        instances = self.argparser.parse_args_into_dataclasses()

        self.assertEqual(instances[0].hf_arg_str, "file_str")
        self.assertEqual(instances[0].hf_arg_int, 5173)

    def test_parse_args_into_dataclasses_bool_accepts_string(self):
        instances = self.argparser.parse_args_into_dataclasses(args=["-b=y"])

        self.assertEqual(instances[0].hf_arg_bool, True)

    def test_parse_args_into_dataclasses_opt_bool_accepts_string(self):
        instances = self.argparser.parse_args_into_dataclasses(args=["-ob=y"])

        self.assertEqual(instances[0].hf_arg_opt_bool, True)

    def test_parse_args_into_dataclasses_enum(self):
        # I don't know if this was the intent, but -e=B fails, and -e=2 does not
        # result in hf_arg_enum=EnumTest.B.
        instances = self.argparser.parse_args_into_dataclasses(args=["-e=2"])

        self.assertEqual(instances[0].hf_arg_enum, EnumTest.B.value)

    def test_parse_args_into_dataclasses_rejects_nonoptional_nonstr_union(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: bool | int = hf_argparser.HfArg()

        with self.assertRaisesRegex(ValueError, "Union"):
            hf_argparser.HfArgumentParser(TestClass)

    def test_parse_args_into_dataclasses_rejects_more_than_two_type_union(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: bool | int | None = hf_argparser.HfArg()

        with self.assertRaisesRegex(ValueError, "Union"):
            hf_argparser.HfArgumentParser(TestClass)

    def test_parse_args_into_dataclasses_accepts_optional(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: None | int = hf_argparser.HfArg(aliases=["-i"])

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_args_into_dataclasses(args=["-i=2"])

        self.assertEqual(instances[0].hf_arg, 2)

    def test_parse_args_into_dataclasses_filters_out_str_from_union(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: str | int = hf_argparser.HfArg(aliases=["-i"])

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_args_into_dataclasses(args=["-i=2"])

        self.assertEqual(instances[0].hf_arg, 2)

    def test_parse_args_into_dataclasses_accepts_list(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: list[int] = hf_argparser.HfArg(aliases=["-i"])

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_args_into_dataclasses(args=["-i", "2", "3"])

        self.assertEqual(instances[0].hf_arg, [2, 3])

    def test_parse_args_into_dataclasses_accepts_float(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg: float = hf_argparser.HfArg(aliases=["-f"])

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_args_into_dataclasses(args=["-f", "1.2"])

        self.assertEqual(instances[0].hf_arg, 1.2)

    def test_parse_dict(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_dict(
            args={"hf_arg_str": "arg_str", "hf_arg_int": 7, "extra": True},
            allow_extra_keys=True,
        )

        self.assertEqual(instances[0].hf_arg_str, "arg_str")
        self.assertEqual(instances[0].hf_arg_int, 7)

    def test_parse_dict_rejects_extra_keys(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        parser = hf_argparser.HfArgumentParser(TestClass)

        with self.assertRaisesRegex(ValueError, "keys are not used"):
            parser.parse_dict(args={"extra": True})

    def test_parse_json_file(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        args_file = os.path.join(self.tmp_dir, "argsfile.json")
        with open(args_file, mode="w", encoding="utf-8") as f:
            f.write(
                """
                {
                    "hf_arg_str": "json_str_ĝ",
                    "hf_arg_int":7509,
                    "extra":true
                }
                """)

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_json_file(args_file, allow_extra_keys=True)

        self.assertEqual(instances[0].hf_arg_str, "json_str_ĝ")
        self.assertEqual(instances[0].hf_arg_int, 7509)

    def test_parse_json_file_rejects_extra_keys(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        args_file = os.path.join(self.tmp_dir, "argsfile.json")
        with open(args_file, mode="w", encoding="utf-8") as f:
            f.write("""{"extra":true}""")

        parser = hf_argparser.HfArgumentParser(TestClass)

        with self.assertRaisesRegex(ValueError, "keys are not used"):
            parser.parse_json_file(args_file)

    def test_parse_yaml_file(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        args_file = os.path.join(self.tmp_dir, "argsfile.yaml")
        with open(args_file, mode="w", encoding="utf-8") as f:
            f.write(
                """
                hf_arg_str: json_str
                hf_arg_int: 7509
                extra: true
                """)

        parser = hf_argparser.HfArgumentParser(TestClass)

        instances = parser.parse_yaml_file(args_file, allow_extra_keys=True)

        self.assertEqual(instances[0].hf_arg_str, "json_str")
        self.assertEqual(instances[0].hf_arg_int, 7509)

    def test_parse_yaml_file_rejects_extra_keys(self):
        @dataclasses.dataclass
        class TestClass:
            hf_arg_str: str = hf_argparser.HfArg(default="default_str")
            hf_arg_int: int = hf_argparser.HfArg(default=2)

        args_file = os.path.join(self.tmp_dir, "argsfile.yaml")
        with open(args_file, mode="w", encoding="utf-8") as f:
            f.write(
            """
            extra: true
            """)

        parser = hf_argparser.HfArgumentParser(TestClass)

        with self.assertRaisesRegex(ValueError, "keys are not used"):
            parser.parse_yaml_file(args_file)
