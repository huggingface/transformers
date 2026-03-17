# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

"""Dynamo exporter.

This module provides the `DynamoExporter` class and helper functions used to
export `PreTrainedModel` instances to `ExportedProgram` via `torch.export.export`.

Helper sections below the exporter class:

1. **Pytree registration** (`register_cache_pytrees_for_model`): flatten/unflatten
   for Cache objects and other custom types so `torch.export` can trace through them.
2. **Dynamic shapes** (`get_auto_dynamic_shapes`): automatic `Dim.AUTO` shape
   inference for all tensor and cache inputs.
"""

import copy
import importlib
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available
from .base import HfExporter
from .utils import prepare_for_export


if is_torch_available():
    import torch

    from ..cache_utils import Cache


if TYPE_CHECKING:
    from torch.export import ExportedProgram

    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    """Exporter that converts `PreTrainedModel` instances to `ExportedProgram`.

    Registers pytree nodes for custom types (Cache, etc.), infers dynamic
    shapes when enabled, and delegates to `torch.export.export`.
    """

    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ExportedProgram":
        model, sample_inputs = prepare_for_export(model, sample_inputs)

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)

        register_cache_pytrees_for_model(model)
        with patch_model(model, sample_inputs):
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
                kwargs=copy.deepcopy(sample_inputs),
                prefer_deferred_runtime_asserts_over_guards=self.export_config.prefer_deferred_runtime_asserts_over_guards,
            )

        return exported_program


# ── Pytree registration ─────────────────────────────────────────────────────
# torch.export needs pytree flatten/unflatten for Cache objects and other
# custom types. The generic flattener serialises any object to a JSON-native
# context (bools, ints, strings, dicts, lists) while collecting tensors into
# a flat list — the inverse reconstructs the original object.
#
# To register a new type: it should be handled automatically by the generic
# flattener. If not, add a branch in _flatten_to_context / _unflatten_from_context.


def _class_to_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _path_to_class(path: str) -> type:
    module_name, qualname = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _flatten_to_context(obj: Any, tensors: list) -> Any:
    """Single-pass: recursively build a JSON-native context while collecting tensors into `tensors`."""
    # --- Pure Python / JSON-native (exact type check — subclasses fall through to stateful objects) ---
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if type(obj) is list:
        return [_flatten_to_context(i, tensors) for i in obj]
    if type(obj) is dict:
        return {k: _flatten_to_context(v, tensors) for k, v in obj.items()}

    # --- Torch objects ---
    if isinstance(obj, torch.Tensor):
        idx = len(tensors)
        tensors.append(obj)
        return {"_t": "tensor", "i": idx}
    if isinstance(obj, torch.Size):
        return {"_t": "size", "v": list(obj)}
    if isinstance(obj, torch.device):
        return {"_t": "device", "s": str(obj)}
    if isinstance(obj, torch.dtype):
        return {"_t": "dtype", "n": str(obj).removeprefix("torch.")}
    if isinstance(obj, torch.layout):
        return {"_t": "layout", "n": str(obj).removeprefix("torch.")}
    if isinstance(obj, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        idx = len(tensors)
        tensors.append(obj)
        return {"_t": "sym", "i": idx}

    # --- Python types ---
    if isinstance(obj, type):
        return {"_t": "type", "p": _class_to_path(obj)}

    # --- Generic Python objects (by structural category) ---
    cls = type(obj)
    if isinstance(obj, dict):  # dict subclasses (OrderedDict, etc.)
        return {
            "_t": "map",
            "p": _class_to_path(cls),
            "v": {k: _flatten_to_context(v, tensors) for k, v in obj.items()},
        }
    if isinstance(obj, (tuple, list, set, frozenset)):  # sequences/sets incl. NamedTuple
        return {
            "_t": "seq",
            "p": _class_to_path(cls),
            "v": [_flatten_to_context(i, tensors) for i in obj],
        }
    if hasattr(obj, "__dict__"):
        return {
            "_t": "obj",
            "p": _class_to_path(cls),
            "s": {k: _flatten_to_context(v, tensors) for k, v in vars(obj).items()},
        }

    raise TypeError(f"Cannot flatten {type(obj).__name__} for pytree context")


def _unflatten_from_context(ctx: Any, tensors: list) -> Any:
    """Reconstruct an object from its JSON-native context, substituting tensor index markers."""
    # --- Pure Python / JSON-native ---
    if ctx is None or type(ctx) in (bool, int, float, str):
        return ctx
    if type(ctx) is list:
        return [_unflatten_from_context(i, tensors) for i in ctx]
    if type(ctx) is dict and "_t" not in ctx:
        return {k: _unflatten_from_context(v, tensors) for k, v in ctx.items()}

    # --- Torch objects ---
    t = ctx["_t"]
    if t == "tensor":
        return tensors[ctx["i"]]
    if t == "layout":
        return getattr(torch, ctx["n"])
    if t == "dtype":
        return getattr(torch, ctx["n"])
    if t == "device":
        return torch.device(ctx["s"])
    if t == "size":
        return torch.Size(ctx["v"])
    if t == "sym":
        return tensors[ctx["i"]]

    # --- Python types ---
    if t == "type":
        return _path_to_class(ctx["p"])

    # --- Generic Python objects ---
    if t == "map":
        cls = _path_to_class(ctx["p"])
        return cls({k: _unflatten_from_context(v, tensors) for k, v in ctx["v"].items()})
    if t == "seq":
        cls = _path_to_class(ctx["p"])
        items = [_unflatten_from_context(i, tensors) for i in ctx["v"]]
        try:
            return cls(items)  # tuple, list subclass, set, frozenset, etc.
        except TypeError:
            return cls(*items)  # NamedTuple (requires positional args)
    if t == "obj":
        cls = _path_to_class(ctx["p"])
        state = {k: _unflatten_from_context(v, tensors) for k, v in ctx["s"].items()}
        instance = cls.__new__(cls)
        instance.__dict__.update(state)
        return instance

    raise TypeError(f"Unknown tag {t!r} in pytree context")


def _pytree_flatten(obj: Any) -> tuple[list, Any]:
    tensors: list = []
    context = _flatten_to_context(obj, tensors)
    return tensors, context


def _pytree_flatten_with_keys(obj: Any):
    leaves, context = _pytree_flatten(obj)
    return [(torch.utils._pytree.SequenceKey(i), leaf) for i, leaf in enumerate(leaves)], context


def _pytree_unflatten(values, context: Any) -> Any:
    return _unflatten_from_context(context, list(values))


def _register_pytree_node(object_cls: type):
    try:
        torch.utils._pytree.register_pytree_node(
            object_cls,
            _pytree_flatten,
            _pytree_unflatten,
            serialized_type_name=_class_to_path(object_cls),
            flatten_with_keys_fn=_pytree_flatten_with_keys,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def _iter_subclasses(cls: type):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_subclasses(subclass)


def register_cache_pytrees_for_model(model: "PreTrainedModel"):
    """Register all relevant cache types as pytree nodes for torch.export."""
    # All transformers Cache subclasses
    for cache_type in _iter_subclasses(Cache):
        _register_pytree_node(cache_type)
    # Model-specific cache classes not inheriting from Cache (e.g. custom per-model caches)
    for _, obj in inspect.getmembers(inspect.getmodule(model)):
        if (
            inspect.isclass(obj)
            and obj.__module__ == model.__class__.__module__
            and obj.__name__.endswith("Cache")
            and not issubclass(obj, Cache)
        ):
            _register_pytree_node(obj)


# ── Dynamic shapes ──────────────────────────────────────────────────────────
# Automatic Dim.AUTO inference for all tensor and cache inputs when
# DynamoConfig.dynamic is True and no explicit dynamic_shapes are provided.


@contextmanager
def patch_model(model: "PreTrainedModel", inputs: dict[str, Any]):
    """Temporarily replace ``model.forward`` with a flat explicit signature derived from ``inputs``.

    Prevents torch.export from expanding ``**kwargs: Unpack[TransformersKwargs]`` into
    ``combined_args``, which would cause an arity mismatch with ``dynamic_shapes``.
    """
    original_forward = model.forward

    def _flat_forward(**kwargs):
        return original_forward(**kwargs)

    _flat_forward.__signature__ = inspect.Signature(
        [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None) for k in inputs]
    )

    try:
        model.forward = _flat_forward
        yield
    finally:
        model.forward = original_forward


def _auto_dynamic_shape(tensor: torch.Tensor) -> dict[int, torch.export.Dim]:
    """Generate a dynamic shape with all dimensions set to Dim.AUTO for a given tensor."""
    return dict.fromkeys(range(tensor.dim()), torch.export.Dim.AUTO)


def get_auto_dynamic_shapes(inputs: Any) -> Any:
    """Recursively build dynamic shapes for any input value.

    - Tensors → per-dimension Dim.AUTO spec.
    - Scalars / None → None (no dynamic dims).
    - Objects with ``__dict__`` (ModelOutput, Cache, …) → flat list of leaf specs,
      matching the ``TreeSpec(list, …)`` that torch.export produces for these types.
    - Lists / tuples → same container type, recursed element-wise.
    - Plain dicts → recursed dict of specs.
    - Everything else → None.
    """
    if isinstance(inputs, torch.Tensor):
        return _auto_dynamic_shape(inputs)
    if inputs is None or isinstance(inputs, (int, float, bool, str)):
        return None
    if hasattr(inputs, "__dict__"):
        leaves, _ = _pytree_flatten(inputs)
        return get_auto_dynamic_shapes(leaves)
    if type(inputs) in (list, tuple, set, frozenset):
        return type(inputs)(get_auto_dynamic_shapes(v) for v in inputs)
    if type(inputs) is dict:
        return {k: get_auto_dynamic_shapes(v) for k, v in inputs.items()}
    return None
