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
from typing import TYPE_CHECKING

from transformers.utils import is_torch_available


if TYPE_CHECKING:
    from torch import nn

if is_torch_available():
    import torch
    from torch import nn


@dataclass
class ReturnEntry:
    arg_name: str
    transform: Callable


if is_torch_available():

    class _NoOpReplacement(nn.Module):
        def __init__(
            self,
            *,
            source_class_name: str,
            signature: inspect.Signature,
            return_entries: tuple[ReturnEntry | None, ...] | None = None,
            return_tuple: bool = False,
        ):
            super().__init__()
            self._source_class_name = source_class_name
            self._signature = signature
            self._return_entries = return_entries
            self._return_tuple = return_tuple

            self.register_buffer("weight", torch.empty(0), persistent=False)

        def forward(self, *args, **kwargs):
            if self._return_entries is None:
                return None

            try:
                bound_arguments = self._signature.bind(self, *args, **kwargs)
            except TypeError as e:
                raise TypeError(f"{self._source_class_name}.forward() {e}") from None
            bound_arguments.apply_defaults()
            outputs = [None] * len(self._return_entries)
            for i, return_entry in enumerate(self._return_entries):
                if return_entry is None:
                    continue

                try:
                    outputs[i] = return_entry.transform(bound_arguments.arguments[return_entry.arg_name])
                except Exception as e:
                    arg_value = bound_arguments.arguments[return_entry.arg_name]
                    raise RuntimeError(
                        f"In the skip replacement for {self._source_class_name}, failed to apply transform "
                        f"{return_entry.transform!r} to argument '{return_entry.arg_name}' "
                        f"(value type: {type(arg_value).__name__}): {e}"
                    ) from e

            return tuple(outputs) if self._return_tuple else outputs[0]


    @dataclass(frozen=True)
    class _NoOpReplacementFactory:
        source_class_name: str
        signature: inspect.Signature
        return_entries: tuple[ReturnEntry | None, ...] | None
        return_tuple: bool

        def __call__(self) -> nn.Module:
            return _NoOpReplacement(
                source_class_name=self.source_class_name,
                signature=self.signature,
                return_entries=self.return_entries,
                return_tuple=self.return_tuple,
            )


def get_skip_replacement(
    cls: type[nn.Module],
    to_return: ReturnEntry | list[ReturnEntry | None] | None,
) -> Callable[[], nn.Module]:
    if to_return is None:
        return _NoOpReplacementFactory(
            source_class_name=cls.__qualname__,
            signature=inspect.signature(cls.forward),
            return_entries=None,
            return_tuple=False,
        )

    if isinstance(to_return, ReturnEntry):
        return_entries = (to_return,)
        return_tuple = False
    else:
        return_entries = tuple(to_return)
        return_tuple = True

    signature = inspect.signature(cls.forward)

    missing_names = [
        return_entry.arg_name
        for return_entry in return_entries
        if return_entry is not None and return_entry.arg_name not in signature.parameters
    ]
    if missing_names:
        raise ValueError(
            f"In the skip replacement for {cls.__qualname__}, the following return entry arg names "
            f"are not arguments of {cls.__qualname__}.forward(): {missing_names}"
        )

    return _NoOpReplacementFactory(
        source_class_name=cls.__qualname__,
        signature=signature,
        return_entries=return_entries,
        return_tuple=return_tuple,
    )
