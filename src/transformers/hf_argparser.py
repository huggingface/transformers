import dataclasses
from argparse import ArgumentParser
from enum import Enum
from typing import Any, Iterable, NewType, Tuple, Union


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses
    to generate arguments.
    """

    def __init__(self, obj: Union[None, DataClass, DataClassType] = None):
        """
        Args:
            obj:
                (Optional) Can be either a dataclass instance, or a dataclass type.
                Both will work the exact same way.
        """
        super().__init__()
        if obj is not None:
            self.add_dataclass_arguments(obj)

    def add_dataclass_arguments(self, obj: Union[DataClass, DataClassType]):
        """
        Args:
            obj:
                Can be either a dataclass instance, or a dataclass type.
                Both will work the exact same way.
        """
        for field in dataclasses.fields(obj):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for x in (int, float, str):
                if typestring == f"typing.Union[{x.__name__}, NoneType]":
                    field.type = x
            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool:
                kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no-{field.name}"
                    kwargs["dest"] = field.name
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    @classmethod
    def parse_into_dataclasses(
        self, types: Iterable[DataClassType], args_to_parse=None, return_remaining=False
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types,
        relying on a shared ArgumentParser generated
        from the dataclasses' type hints.

        This relies on argparse's `ArgumentParser.parse_known_args`.
        See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            types:
                List of dataclass types for which we will "fill" instances with the parsed args.
            args_to_parse:
                List of strings to parse. The default is taken from sys.argv.
                (same as argparse.ArgumentParser)
            return_remaining:
                If true, also return a list of remaining argument strings.

        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as the input
                - The potential list of remaining argument strings.
                  (same as argparse.ArgumentParser.parse_known_args)
        """
        parser = HfArgumentParser()
        for dtype in types:
            parser.add_dataclass_arguments(dtype)
        # Now let's parse.
        namespace, remaining_args = parser.parse_known_args(args=args_to_parse)
        outputs = []
        for dtype in types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        if return_remaining:
            return (*outputs, remaining_args)
        else:
            return (*outputs,)
