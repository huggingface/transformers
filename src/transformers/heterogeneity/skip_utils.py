from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import nn


@dataclass
class ReturnEntry:
    arg_name: str
    transform: Callable


def get_skip_replacement(
    cls: type[nn.Module],
    to_return: ReturnEntry | list[ReturnEntry | None] | None,
) -> type[nn.Module]:
    import torch
    from torch import nn

    class NoOpReplacement(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty(0), persistent=False)

        def forward(self, *args, **kwargs):
            if to_return is None:
                return None

            if isinstance(to_return, ReturnEntry):
                return_entries = [to_return]
                return_tuple = False
            else:
                return_entries = to_return
                return_tuple = True

            sig = inspect.signature(cls.forward)
            try:
                bound_arguments = sig.bind(self, *args, **kwargs)
            except TypeError as e:
                raise TypeError(f"{cls.__qualname__}.forward() {e}") from None
            bound_arguments.apply_defaults()
            outputs = [None] * len(return_entries)
            missing_names = []
            for i, return_entry in enumerate(return_entries):
                if return_entry is None:
                    outputs[i] = None
                    continue

                if return_entry.arg_name not in bound_arguments.arguments:
                    missing_names.append(return_entry.arg_name)
                    continue

                try:
                    outputs[i] = return_entry.transform(bound_arguments.arguments[return_entry.arg_name])
                except Exception as e:
                    arg_value = bound_arguments.arguments[return_entry.arg_name]
                    raise type(e)(
                        f"In the skip replacement for {cls.__qualname__}, failed to apply transform "
                        f"{return_entry.transform!r} to argument '{return_entry.arg_name}' "
                        f"(value type: {type(arg_value).__name__}): {e}"
                    ) from e

            if missing_names:
                raise ValueError(
                    f"In the skip replacement for {cls.__qualname__}, the following return entry arg names "
                    f"are not arguments of {cls.__qualname__}.forward(): {missing_names}"
                )

            return tuple(outputs) if return_tuple else outputs[0]

    return NoOpReplacement
