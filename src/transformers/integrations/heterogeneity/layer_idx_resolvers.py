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


@dataclass(frozen=True)
class LayerIdxResolver(ABC):
    """Resolve a decoder layer's index while its constructor is being called."""

    variable_name: str

    @abstractmethod
    def resolve(
        self,
        *,
        layer_init: FunctionType,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        model: PreTrainedModel,
    ) -> Any:
        """Resolve the layer index from the active layer-initialization call."""


@dataclass(frozen=True)
class LayerIdxFromArgument(LayerIdxResolver):
    """Resolve a layer index from an argument of the repeated layer's constructor."""

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

        bound_arguments.apply_defaults()
        if self.variable_name not in bound_arguments.arguments:
            raise RuntimeError(
                f"`LayerIdxFromArgument({self.variable_name!r})` could not determine the layer index for "
                f"heterogeneous model initialization of `{model.__class__.__name__}` because "
                f"`{self.variable_name}` is not an argument of `{layer_init.__qualname__}`."
            )
        return bound_arguments.arguments[self.variable_name]


@dataclass(frozen=True)
class LayerIdxFromModelInitStack(LayerIdxResolver):
    """Resolve a layer index from a local in the current model initialization stack."""

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

        # Skip this resolver and the patched layer initializer so their locals cannot shadow the model's.
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
