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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


class LayerIdxResolver(ABC):
    """Base class for finding the index of a layer while the layer is being built.

    When a layer is built, the patched `__init__` first calls `resolve` to get the layer index. Then it calls the
    original `__init__` of `layer_cls` with the layer's config, `config.per_layer_config[layer_idx]`. Create a subclass
    if `LayerIdxFromArgument` and `LayerIdxFromModelInitStack` can't find the index.
    """

    @abstractmethod
    def resolve(
        self,
        *,
        layer_init: FunctionType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        model: PreTrainedModel,
    ) -> Any:
        """Returns the index of the layer that is being built.

        Args:
            layer_init (`FunctionType`):
                The original `__init__` of the spec's `layer_cls`, before patching.
            args (`tuple[Any, ...]`):
                The positional arguments of the `__init__` call. The first two are always the layer (`self`) and the
                model's config.
            kwargs (`dict[str, Any]`):
                The keyword arguments of the `__init__` call.
            model ([`PreTrainedModel`]):
                The model that builds the layer in its `__init__`. Its config is the config object that was passed to
                the layer's `__init__`.

        Returns:
            `int`: The layer index, from `0` to `config.num_hidden_layers - 1`. Other values raise an error.
        """


@dataclass(frozen=True)
class LayerIdxFromArgument(LayerIdxResolver):
    """Finds the layer index in an argument of the layer's `__init__`, such as `layer_idx`.

    The argument must be a named parameter of `__init__`, not part of `**kwargs`, and can be passed by position or by
    keyword.

    Args:
        variable_name (`str`):
            The name of the `__init__` argument that holds the layer index.
    """

    variable_name: str

    def resolve(
        self,
        *,
        layer_init: FunctionType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        model: PreTrainedModel,
    ) -> Any:
        signature = inspect.signature(layer_init)
        try:
            bound_arguments = signature.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"`LayerIdxFromArgument({self.variable_name!r})` could not bind the arguments passed to "
                f"`{layer_init.__qualname__}`: {e}"
            ) from None

        if self.variable_name not in bound_arguments.arguments:
            if self.variable_name in signature.parameters:
                reason = f"`{self.variable_name}` was not passed to `{layer_init.__qualname__}`"
            else:
                reason = f"`{self.variable_name}` is not an argument of `{layer_init.__qualname__}`"
            raise RuntimeError(
                f"`LayerIdxFromArgument({self.variable_name!r})` could not determine the layer index for "
                f"heterogeneous model initialization of `{model.__class__.__name__}` because {reason}."
            )
        return bound_arguments.arguments[self.variable_name]


@dataclass(frozen=True)
class LayerIdxFromModelInitStack(LayerIdxResolver):
    """Finds the layer index in a local variable of the code that builds the layer.

    Use it when the layer's `__init__` doesn't take the index as an argument, for example when the index is a loop
    variable in the model's `__init__`. The resolver searches the functions that called the layer's `__init__`, from
    the direct caller outward, and returns the first value that it finds. The search stops at the first `__init__` of
    the model.

    Args:
        variable_name (`str`):
            The name of the local variable that holds the layer index.
    """

    variable_name: str

    def resolve(
        self,
        *,
        layer_init: FunctionType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        model: PreTrainedModel,
    ) -> Any:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            raise RuntimeError(self._missing_variable_error(model))

        # Skip the frames of this resolver and of the patched layer `__init__`, so that their local variables can't
        # hide variables with the same name in the model's code.
        frame = frame.f_back.f_back

        while frame is not None:
            if self.variable_name in frame.f_locals:
                return frame.f_locals[self.variable_name]
            if frame.f_code.co_name == "__init__" and frame.f_locals.get("self") is model:
                break
            frame = frame.f_back

        raise RuntimeError(self._missing_variable_error(model))

    def _missing_variable_error(self, model: PreTrainedModel) -> str:
        return (
            f"`LayerIdxFromModelInitStack({self.variable_name!r})` could not find `{self.variable_name}` in the model "
            f"initialization stack up to and including `{model.__class__.__name__}.__init__`."
        )
