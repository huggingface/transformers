import dataclasses
from argparse import ArgumentParser
from enum import Enum
from typing import Any, List, NewType, Optional


DataClass = NewType("DataClass", Any)


class HfArgparser:
    """
    This helper class uses type hints on dataclasses to
    generate `argparse` arguments.
    """

    parser: ArgumentParser

    def __init__(self, parser: Optional[ArgumentParser] = None):
        if parser is not None:
            self.parser = parser
        else:
            self.parser = ArgumentParser()

    def add_arguments(self, obj: DataClass) -> ArgumentParser:
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
            self.parser.add_argument(field_name, **kwargs)
        return self.parser

    @classmethod
    def pipeline(objs: List[DataClass]) -> None:
        """
        Pass command-line args into a sequence of `ArgumentParser`s generated
        from dataclasses, filling each of those dataclasses along the way.
        """
        parser = ArgumentParser()
        remaining_args = None
        # ^^ we start from the default parameter (i.e. stdin)
        for obj in objs:
            parser_step = HfArgparser(parser).add_arguments(obj)
            _, remaining_args = parser_step.parse_known_args(args=remaining_args, namespace=obj)
