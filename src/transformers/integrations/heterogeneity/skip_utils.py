# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from transformers.utils import is_torch_available


if TYPE_CHECKING:
    from torch import nn

if is_torch_available():
    import torch
    from torch import nn


@dataclass(frozen=True)
class ReturnEntry:
    arg_name: str
    transform: Callable


@dataclass(frozen=True)
class _ResolvedReturnEntry(ReturnEntry):
    position: int | None
    default: object

    def resolve(self, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        if self.arg_name in kwargs:
            return kwargs[self.arg_name]
        if self.position is not None and self.position < len(args):
            return args[self.position]
        if self.default is not inspect.Parameter.empty:
            return self.default
        raise TypeError(f"missing a required argument: '{self.arg_name}'")


def _resolve_return_entries(
    cls: type[nn.Module], return_entries: tuple[ReturnEntry | None, ...] | None
) -> tuple[_ResolvedReturnEntry | None, ...] | None:
    if return_entries is None:
        return None

    parameters = inspect.signature(cls.forward).parameters
    parameter_names = tuple(parameters)[1:]

    missing_names = [
        return_entry.arg_name
        for return_entry in return_entries
        if return_entry is not None and return_entry.arg_name not in parameter_names
    ]
    if missing_names:
        raise ValueError(
            f"In the skip replacement for {cls.__qualname__}, the following return entry arg names "
            f"are not arguments of {cls.__qualname__}.forward(): {missing_names}"
        )

    resolved_entries = []
    for return_entry in return_entries:
        if return_entry is None:
            resolved_entries.append(None)
            continue

        parameter = parameters[return_entry.arg_name]
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"Return entries cannot reference variadic argument '{return_entry.arg_name}' "
                f"in {cls.__qualname__}.forward()."
            )

        resolved_entries.append(
            _ResolvedReturnEntry(
                arg_name=return_entry.arg_name,
                transform=return_entry.transform,
                position=(
                    parameter_names.index(return_entry.arg_name)
                    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    else None
                ),
                default=parameter.default,
            )
        )

    return tuple(resolved_entries)


if is_torch_available():

    class _NoOpReplacement(nn.Module):
        def __init__(
            self,
            *,
            source_class_name: str,
            return_entries: tuple[_ResolvedReturnEntry | None, ...] | None = None,
            return_tuple: bool = False,
        ):
            super().__init__()
            self._source_class_name = source_class_name
            self._return_entries = return_entries
            self._return_tuple = return_tuple

            self.register_buffer("weight", torch.empty(0), persistent=False)

        def forward(self, *args, **kwargs):
            if self._return_entries is None:
                return None

            outputs = [None] * len(self._return_entries)
            for i, return_entry in enumerate(self._return_entries):
                if return_entry is None:
                    continue

                try:
                    arg_value = return_entry.resolve(args, kwargs)
                except TypeError:
                    raise TypeError(
                        f"In the skip replacement for {self._source_class_name}, "
                        f"required argument '{return_entry.arg_name}' was not provided"
                    ) from None

                try:
                    outputs[i] = return_entry.transform(arg_value)
                except Exception as e:
                    raise RuntimeError(
                        f"In the skip replacement for {self._source_class_name}, failed to apply transform "
                        f"{return_entry.transform!r} to argument '{return_entry.arg_name}' "
                        f"(value type: {type(arg_value).__name__}): {e}"
                    ) from e

            return tuple(outputs) if self._return_tuple else outputs[0]


def get_skip_replacement(
    cls: type[nn.Module],
    to_return: ReturnEntry | list[ReturnEntry | None] | None,
) -> Callable[[], nn.Module]:
    if to_return is None:
        return_entries = None
        return_tuple = False
    elif isinstance(to_return, ReturnEntry):
        return_entries = (to_return,)
        return_tuple = False
    else:
        return_entries = tuple(to_return)
        return_tuple = True

    return partial(
        _NoOpReplacement,
        source_class_name=cls.__qualname__,
        return_entries=_resolve_return_entries(cls, return_entries),
        return_tuple=return_tuple,
    )
