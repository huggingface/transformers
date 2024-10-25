# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, get_args, get_origin

import yaml

from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import make_choice_type_function, string_to_bool
from transformers.testing_utils import require_torch
from transformers.training_args import _VALID_DICT_FIELDS


# Since Python 3.10, we can use the builtin `|` operator for Union types
# See PEP 604: https://peps.python.org/pep-0604
is_python_no_less_than_3_10 = sys.version_info >= (3, 10)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class BasicExample:
    foo: int
    bar: float
    baz: str
    flag: bool


@dataclass
class WithDefaultExample:
    foo: int = 42
    baz: str = field(default="toto", metadata={"help": "help message"})


@dataclass
class WithDefaultBoolExample:
    foo: bool = False
    baz: bool = True
    opt: Optional[bool] = None


class BasicEnum(Enum):
    titi = "titi"
    toto = "toto"


class MixedTypeEnum(Enum):
    titi = "titi"
    toto = "toto"
    fourtytwo = 42


@dataclass
class EnumExample:
    foo: BasicEnum = "toto"

    def __post_init__(self):
        self.foo = BasicEnum(self.foo)


@dataclass
class MixedTypeEnumExample:
    foo: MixedTypeEnum = "toto"

    def __post_init__(self):
        self.foo = MixedTypeEnum(self.foo)


@dataclass
class OptionalExample:
    foo: Optional[int] = None
    bar: Optional[float] = field(default=None, metadata={"help": "help message"})
    baz: Optional[str] = None
    ces: Optional[List[str]] = list_field(default=[])
    des: Optional[List[int]] = list_field(default=[])


@dataclass
class ListExample:
    foo_int: List[int] = list_field(default=[])
    bar_int: List[int] = list_field(default=[1, 2, 3])
    foo_str: List[str] = list_field(default=["Hallo", "Bonjour", "Hello"])
    foo_float: List[float] = list_field(default=[0.1, 0.2, 0.3])


@dataclass
class RequiredExample:
    required_list: List[int] = field()
    required_str: str = field()
    required_enum: BasicEnum = field()

    def __post_init__(self):
        self.required_enum = BasicEnum(self.required_enum)


@dataclass
class StringLiteralAnnotationExample:
    foo: int
    required_enum: "BasicEnum" = field()
    opt: "Optional[bool]" = None
    baz: "str" = field(default="toto", metadata={"help": "help message"})
    foo_str: "List[str]" = list_field(default=["Hallo", "Bonjour", "Hello"])


if is_python_no_less_than_3_10:

    @dataclass
    class WithDefaultBoolExamplePep604:
        foo: bool = False
        baz: bool = True
        opt: bool | None = None

    @dataclass
    class OptionalExamplePep604:
        foo: int | None = None
        bar: float | None = field(default=None, metadata={"help": "help message"})
        baz: str | None = None
        ces: list[str] | None = list_field(default=[])
        des: list[int] | None = list_field(default=[])


class HfArgumentParserTest(unittest.TestCase):
    def argparsersEqual(self, a: argparse.ArgumentParser, b: argparse.ArgumentParser):
        """
        Small helper to check pseudo-equality of parsed arguments on `ArgumentParser` instances.
        """
        self.assertEqual(len(a._actions), len(b._actions))
        for x, y in zip(a._actions, b._actions):
            xx = {k: v for k, v in vars(x).items() if k != "container"}
            yy = {k: v for k, v in vars(y).items() if k != "container"}

            # Choices with mixed type have custom function as "type"
            # So we need to compare results directly for equality
            if xx.get("choices", None) and yy.get("choices", None):
                for expected_choice in yy["choices"] + xx["choices"]:
                    self.assertEqual(xx["type"](expected_choice), yy["type"](expected_choice))
                del xx["type"], yy["type"]

            self.assertEqual(xx, yy)

    def test_basic(self):
        parser = HfArgumentParser(BasicExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", type=int, required=True)
        expected.add_argument("--bar", type=float, required=True)
        expected.add_argument("--baz", type=str, required=True)
        expected.add_argument("--flag", type=string_to_bool, default=False, const=True, nargs="?")
        self.argparsersEqual(parser, expected)

        args = ["--foo", "1", "--baz", "quux", "--bar", "0.5"]
        (example,) = parser.parse_args_into_dataclasses(args, look_for_args_file=False)
        self.assertFalse(example.flag)

    def test_with_default(self):
        parser = HfArgumentParser(WithDefaultExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=42, type=int)
        expected.add_argument("--baz", default="toto", type=str, help="help message")
        self.argparsersEqual(parser, expected)

    def test_with_default_bool(self):
        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", type=string_to_bool, default=False, const=True, nargs="?")
        expected.add_argument("--baz", type=string_to_bool, default=True, const=True, nargs="?")
        # A boolean no_* argument always has to come after its "default: True" regular counter-part
        # and its default must be set to False
        expected.add_argument("--no_baz", "--no-baz", action="store_false", default=False, dest="baz")
        expected.add_argument("--opt", type=string_to_bool, default=None)

        dataclass_types = [WithDefaultBoolExample]
        if is_python_no_less_than_3_10:
            dataclass_types.append(WithDefaultBoolExamplePep604)

        for dataclass_type in dataclass_types:
            parser = HfArgumentParser(dataclass_type)
            self.argparsersEqual(parser, expected)

            args = parser.parse_args([])
            self.assertEqual(args, Namespace(foo=False, baz=True, opt=None))

            args = parser.parse_args(["--foo", "--no_baz"])
            self.assertEqual(args, Namespace(foo=True, baz=False, opt=None))

            args = parser.parse_args(["--foo", "--no-baz"])
            self.assertEqual(args, Namespace(foo=True, baz=False, opt=None))

            args = parser.parse_args(["--foo", "--baz"])
            self.assertEqual(args, Namespace(foo=True, baz=True, opt=None))

            args = parser.parse_args(["--foo", "True", "--baz", "True", "--opt", "True"])
            self.assertEqual(args, Namespace(foo=True, baz=True, opt=True))

            args = parser.parse_args(["--foo", "False", "--baz", "False", "--opt", "False"])
            self.assertEqual(args, Namespace(foo=False, baz=False, opt=False))

    def test_with_enum(self):
        parser = HfArgumentParser(MixedTypeEnumExample)

        expected = argparse.ArgumentParser()
        expected.add_argument(
            "--foo",
            default="toto",
            choices=["titi", "toto", 42],
            type=make_choice_type_function(["titi", "toto", 42]),
        )
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args.foo, "toto")
        enum_ex = parser.parse_args_into_dataclasses([])[0]
        self.assertEqual(enum_ex.foo, MixedTypeEnum.toto)

        args = parser.parse_args(["--foo", "titi"])
        self.assertEqual(args.foo, "titi")
        enum_ex = parser.parse_args_into_dataclasses(["--foo", "titi"])[0]
        self.assertEqual(enum_ex.foo, MixedTypeEnum.titi)

        args = parser.parse_args(["--foo", "42"])
        self.assertEqual(args.foo, 42)
        enum_ex = parser.parse_args_into_dataclasses(["--foo", "42"])[0]
        self.assertEqual(enum_ex.foo, MixedTypeEnum.fourtytwo)

    def test_with_literal(self):
        @dataclass
        class LiteralExample:
            foo: Literal["titi", "toto", 42] = "toto"

        parser = HfArgumentParser(LiteralExample)

        expected = argparse.ArgumentParser()
        expected.add_argument(
            "--foo",
            default="toto",
            choices=("titi", "toto", 42),
            type=make_choice_type_function(["titi", "toto", 42]),
        )
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args.foo, "toto")

        args = parser.parse_args(["--foo", "titi"])
        self.assertEqual(args.foo, "titi")

        args = parser.parse_args(["--foo", "42"])
        self.assertEqual(args.foo, 42)

    def test_with_list(self):
        parser = HfArgumentParser(ListExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo_int", "--foo-int", nargs="+", default=[], type=int)
        expected.add_argument("--bar_int", "--bar-int", nargs="+", default=[1, 2, 3], type=int)
        expected.add_argument("--foo_str", "--foo-str", nargs="+", default=["Hallo", "Bonjour", "Hello"], type=str)
        expected.add_argument("--foo_float", "--foo-float", nargs="+", default=[0.1, 0.2, 0.3], type=float)

        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(
            args,
            Namespace(foo_int=[], bar_int=[1, 2, 3], foo_str=["Hallo", "Bonjour", "Hello"], foo_float=[0.1, 0.2, 0.3]),
        )

        args = parser.parse_args("--foo_int 1 --bar_int 2 3 --foo_str a b c --foo_float 0.1 0.7".split())
        self.assertEqual(args, Namespace(foo_int=[1], bar_int=[2, 3], foo_str=["a", "b", "c"], foo_float=[0.1, 0.7]))

        args = parser.parse_args("--foo-int 1 --bar-int 2 3 --foo-str a b c --foo-float 0.1 0.7".split())
        self.assertEqual(args, Namespace(foo_int=[1], bar_int=[2, 3], foo_str=["a", "b", "c"], foo_float=[0.1, 0.7]))

    def test_with_optional(self):
        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=None, type=int)
        expected.add_argument("--bar", default=None, type=float, help="help message")
        expected.add_argument("--baz", default=None, type=str)
        expected.add_argument("--ces", nargs="+", default=[], type=str)
        expected.add_argument("--des", nargs="+", default=[], type=int)

        dataclass_types = [OptionalExample]
        if is_python_no_less_than_3_10:
            dataclass_types.append(OptionalExamplePep604)

        for dataclass_type in dataclass_types:
            parser = HfArgumentParser(dataclass_type)

            self.argparsersEqual(parser, expected)

            args = parser.parse_args([])
            self.assertEqual(args, Namespace(foo=None, bar=None, baz=None, ces=[], des=[]))

            args = parser.parse_args("--foo 12 --bar 3.14 --baz 42 --ces a b c --des 1 2 3".split())
            self.assertEqual(args, Namespace(foo=12, bar=3.14, baz="42", ces=["a", "b", "c"], des=[1, 2, 3]))

    def test_with_required(self):
        parser = HfArgumentParser(RequiredExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--required_list", "--required-list", nargs="+", type=int, required=True)
        expected.add_argument("--required_str", "--required-str", type=str, required=True)
        expected.add_argument(
            "--required_enum",
            "--required-enum",
            type=make_choice_type_function(["titi", "toto"]),
            choices=["titi", "toto"],
            required=True,
        )
        self.argparsersEqual(parser, expected)

    def test_with_string_literal_annotation(self):
        parser = HfArgumentParser(StringLiteralAnnotationExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", type=int, required=True)
        expected.add_argument(
            "--required_enum",
            "--required-enum",
            type=make_choice_type_function(["titi", "toto"]),
            choices=["titi", "toto"],
            required=True,
        )
        expected.add_argument("--opt", type=string_to_bool, default=None)
        expected.add_argument("--baz", default="toto", type=str, help="help message")
        expected.add_argument("--foo_str", "--foo-str", nargs="+", default=["Hallo", "Bonjour", "Hello"], type=str)
        self.argparsersEqual(parser, expected)

    def test_parse_dict(self):
        parser = HfArgumentParser(BasicExample)

        args_dict = {
            "foo": 12,
            "bar": 3.14,
            "baz": "42",
            "flag": True,
        }

        parsed_args = parser.parse_dict(args_dict)[0]
        args = BasicExample(**args_dict)
        self.assertEqual(parsed_args, args)

    def test_parse_dict_extra_key(self):
        parser = HfArgumentParser(BasicExample)

        args_dict = {
            "foo": 12,
            "bar": 3.14,
            "baz": "42",
            "flag": True,
            "extra": 42,
        }

        self.assertRaises(ValueError, parser.parse_dict, args_dict, allow_extra_keys=False)

    def test_parse_json(self):
        parser = HfArgumentParser(BasicExample)

        args_dict_for_json = {
            "foo": 12,
            "bar": 3.14,
            "baz": "42",
            "flag": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_local_path = os.path.join(tmp_dir, "temp_json")
            os.mkdir(temp_local_path)
            with open(temp_local_path + ".json", "w+") as f:
                json.dump(args_dict_for_json, f)
            parsed_args = parser.parse_json_file(Path(temp_local_path + ".json"))[0]

        args = BasicExample(**args_dict_for_json)
        self.assertEqual(parsed_args, args)

    def test_parse_yaml(self):
        parser = HfArgumentParser(BasicExample)

        args_dict_for_yaml = {
            "foo": 12,
            "bar": 3.14,
            "baz": "42",
            "flag": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_local_path = os.path.join(tmp_dir, "temp_yaml")
            os.mkdir(temp_local_path)
            with open(temp_local_path + ".yaml", "w+") as f:
                yaml.dump(args_dict_for_yaml, f)
            parsed_args = parser.parse_yaml_file(Path(temp_local_path + ".yaml"))[0]
        args = BasicExample(**args_dict_for_yaml)
        self.assertEqual(parsed_args, args)

    def test_integration_training_args(self):
        parser = HfArgumentParser(TrainingArguments)
        self.assertIsNotNone(parser)

    def test_valid_dict_annotation(self):
        """
        Tests to make sure that `dict` based annotations
        are correctly made in the `TrainingArguments`.

        If this fails, a type annotation change is
        needed on a new input
        """
        base_list = _VALID_DICT_FIELDS.copy()
        args = TrainingArguments

        # First find any annotations that contain `dict`
        fields = args.__dataclass_fields__

        raw_dict_fields = []
        optional_dict_fields = []

        for field in fields.values():
            # First verify raw dict
            if field.type in (dict, Dict):
                raw_dict_fields.append(field)
            # Next check for `Union` or `Optional`
            elif get_origin(field.type) == Union:
                if any(arg in (dict, Dict) for arg in get_args(field.type)):
                    optional_dict_fields.append(field)

        # First check: anything in `raw_dict_fields` is very bad
        self.assertEqual(
            len(raw_dict_fields),
            0,
            "Found invalid raw `dict` types in the `TrainingArgument` typings. "
            "This leads to issues with the CLI. Please turn this into `typing.Optional[dict,str]`",
        )

        # Next check raw annotations
        for field in optional_dict_fields:
            args = get_args(field.type)
            # These should be returned as `dict`, `str`, ...
            # we only care about the first two
            self.assertIn(args[0], (Dict, dict))
            self.assertEqual(
                str(args[1]),
                "<class 'str'>",
                f"Expected field `{field.name}` to have a type signature of at least `typing.Union[dict,str,...]` for CLI compatibility, "
                "but `str` not found. Please fix this.",
            )

        # Second check: anything in `optional_dict_fields` is bad if it's not in `base_list`
        for field in optional_dict_fields:
            self.assertIn(
                field.name,
                base_list,
                f"Optional dict field `{field.name}` is not in the base list of valid fields. Please add it to `training_args._VALID_DICT_FIELDS`",
            )

    @require_torch
    def test_valid_dict_input_parsing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                accelerator_config='{"split_batches": "True", "gradient_accumulation_kwargs": {"num_steps": 2}}',
            )
            self.assertEqual(args.accelerator_config.split_batches, True)
            self.assertEqual(args.accelerator_config.gradient_accumulation_kwargs["num_steps"], 2)
