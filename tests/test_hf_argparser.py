import argparse
import unittest
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments


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


class BasicEnum(Enum):
    titi = "titi"
    toto = "toto"


@dataclass
class EnumExample:
    foo: BasicEnum = BasicEnum.toto


@dataclass
class OptionalExample:
    foo: Optional[int] = None
    bar: Optional[float] = field(default=None, metadata={"help": "help message"})
    baz: Optional[str] = None


class HfArgumentParserTest(unittest.TestCase):
    def argparsersEqual(self, a: argparse.ArgumentParser, b: argparse.ArgumentParser) -> bool:
        """
        Small helper to check pseudo-equality of parsed arguments on `ArgumentParser` instances.
        """
        self.assertEqual(len(a._actions), len(b._actions))
        for x, y in zip(a._actions, b._actions):
            xx = {k: v for k, v in vars(x).items() if k != "container"}
            yy = {k: v for k, v in vars(y).items() if k != "container"}
            self.assertEqual(xx, yy)

    def test_basic(self):
        parser = HfArgumentParser(BasicExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", type=int, required=True)
        expected.add_argument("--bar", type=float, required=True)
        expected.add_argument("--baz", type=str, required=True)
        expected.add_argument("--flag", action="store_true")
        self.argparsersEqual(parser, expected)

    def test_with_default(self):
        parser = HfArgumentParser(WithDefaultExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=42, type=int)
        expected.add_argument("--baz", default="toto", type=str, help="help message")
        self.argparsersEqual(parser, expected)

    def test_with_default_bool(self):
        parser = HfArgumentParser(WithDefaultBoolExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", action="store_true")
        expected.add_argument("--no-baz", action="store_false", dest="baz")
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args, Namespace(foo=False, baz=True))

        args = parser.parse_args(["--foo", "--no-baz"])
        self.assertEqual(args, Namespace(foo=True, baz=False))

    def test_with_enum(self):
        parser = HfArgumentParser(EnumExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=BasicEnum.toto, choices=list(BasicEnum), type=BasicEnum)
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args.foo, BasicEnum.toto)

        args = parser.parse_args(["--foo", "titi"])
        self.assertEqual(args.foo, BasicEnum.titi)

    def test_with_optional(self):
        parser = HfArgumentParser(OptionalExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=None, type=int)
        expected.add_argument("--bar", default=None, type=float, help="help message")
        expected.add_argument("--baz", default=None, type=str)
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args, Namespace(foo=None, bar=None, baz=None))

        args = parser.parse_args("--foo 12 --bar 3.14 --baz 42".split())
        self.assertEqual(args, Namespace(foo=12, bar=3.14, baz="42"))

    def test_integration_training_args(self):
        parser = HfArgumentParser(TrainingArguments)
        self.assertIsNotNone(parser)
