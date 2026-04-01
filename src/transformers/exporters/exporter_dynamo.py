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

Helper sections in this file:

1. **Model patches** (`patch_untraceable_patterns`): reversible patches applied during
   `torch.export` tracing to replace non-exportable model patterns (data-dependent
   loops, in-place ops, mask checks) with export-safe equivalents.
2. **Pytree registration** (`register_cache_pytrees_for_model`): flatten/unflatten
   for Cache objects and other custom types so `torch.export` can trace through them.
3. **Model signature patch** (`patch_forward_signature`): replaces `model.forward`
   with a flat explicit signature so `torch.export` does not choke on `**kwargs`.
4. **Dynamic shapes** (`get_auto_dynamic_shapes`): automatic `Dim.AUTO` inference
   for all tensor and cache inputs when `DynamoConfig.dynamic=True`.
"""

import copy
import functools
import importlib
import inspect
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, torch_compilable_check
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
    """Exporter that converts a [`PreTrainedModel`] to a `torch.export.ExportedProgram`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

    >>> exporter = DynamoExporter(export_config=DynamoConfig(dynamic=True))
    >>> exported = exporter.export(model, inputs)
    >>> outputs = exported.module()(**inputs)
    ```
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

        # Explicit dynamic_shapes often trigger model-internal guards (e.g. RoPE constraints)
        # that the symbolic solver can't prove at trace time. Defer them to runtime automatically.
        prefer_deferred = (
            self.export_config.prefer_deferred_runtime_asserts_over_guards
            or self.export_config.dynamic_shapes is not None
        )

        register_cache_pytrees_for_model(model)
        with patch_untraceable_patterns(model), patch_forward_signature(model, sample_inputs):
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
                kwargs=copy.deepcopy(sample_inputs),
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred,
            )

        return exported_program


# ── Untraceable pattern patches ────────────────────────────────────────────
# Reversible patches applied by `patch_untraceable_patterns` during
# `torch.export` tracing. Each replaces a non-exportable model pattern
# (data-dependent control flow, in-place ops on views, etc.) with an
# export-safe equivalent.
#
# Each patcher takes a module and returns `(attr, replacement)` if the
# module matches, or `None` otherwise. Originals are saved by the context
# manager and restored when tracing completes.
#
# To add a new patch: define a `_patch_*` function and append to _MODEL_PATCHERS.


def _exportable_update_mask(attention_mask, past_key_values_or_cache_position=None, *args, **kwargs):
    """Export-safe mamba/linear-attn mask: keeps only `has_previous_state` (a Python bool)."""
    has_previous_state = getattr(past_key_values_or_cache_position, "has_previous_state", False)
    if callable(has_previous_state):
        has_previous_state = has_previous_state()
    if has_previous_state:
        return None
    return attention_mask


def _patch_mamba_mask(module):
    """Replace data-dependent `torch.all(mask == 1)` in mamba mask update."""
    if hasattr(module, "_update_mamba_mask"):
        return ("_update_mamba_mask", _exportable_update_mask)


def _patch_linear_attn_mask(module):
    """Replace data-dependent mask check in linear attention (falcon_h1)."""
    if hasattr(module, "_update_linear_attn_mask"):
        return ("_update_linear_attn_mask", _exportable_update_mask)


def _patch_classifier_cast(module):
    """Disable classifier dtype cast in nllb-moe (not traceable)."""
    if hasattr(module, "_cast_classifier"):
        return ("_cast_classifier", lambda *args, **kwargs: None)


def _patch_chunked_vision_attention(module):
    """Replace split → loop → cat with reshaped batch SDPA for vision attention."""
    has_attention = hasattr(module, "qkv") or (
        hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")
    )
    has_chunked_attention = has_attention and "zip(*splits)" in inspect.getsource(module.forward)
    if has_chunked_attention:
        return ("forward", functools.partial(_reshaped_vision_attention_forward, module))


def _reshaped_vision_attention_forward(
    self,
    hidden_states: "torch.Tensor",
    cu_seqlens: "torch.Tensor",
    rotary_pos_emb: "torch.Tensor | None" = None,
    position_embeddings: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    **kwargs,
) -> "torch.Tensor":
    """Export-safe vision attention: reshape segments into batch dim, single SDPA call."""

    seq_length = hidden_states.shape[0]
    num_segments = cu_seqlens.shape[0] - 1
    torch_compilable_check(
        seq_length % num_segments == 0,
        "Chunked vision attention requires uniform segment lengths during export. "
        "Ensure all images have the same resolution (use do_resize=True in the processor) "
        "or pad inputs to a common size.",
    )

    if hasattr(self, "qkv"):
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
    else:
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        model_module = sys.modules[type(self).__module__]
        apply_rotary_pos_emb_vision = getattr(model_module, "apply_rotary_pos_emb_vision")
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    seg_len = seq_length // num_segments

    # (seq, heads, dim) → (n_seg, seg_len, heads, dim) → (n_seg, heads, seg_len, dim)
    def _to_batched(t):
        return t.unflatten(0, (num_segments, seg_len)).transpose(1, 2)

    query_states = _to_batched(query_states)
    key_states = _to_batched(key_states)
    value_states = _to_batched(value_states)

    torch._check(query_states.shape[0] != 0)
    torch._check(query_states.shape[2] != 0)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=False,
        scale=self.scaling,
        dropout_p=0.0 if not self.training else self.attention_dropout,
    )

    # (n_seg, heads, seg_len, dim) → (n_seg, seg_len, heads, dim) → (seq, heads*dim)
    attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
    out_proj = self.proj if hasattr(self, "proj") else self.out_proj
    return out_proj(attn_output)


_MODEL_PATCHERS = [
    _patch_mamba_mask,
    _patch_classifier_cast,
    _patch_linear_attn_mask,
    _patch_chunked_vision_attention,
]


@contextmanager
def patch_untraceable_patterns(model: "PreTrainedModel"):
    """Temporarily replace untraceable model patterns with export-safe equivalents.

    Iterates every module, applies matching patchers from `_MODEL_PATCHERS`,
    and reverts all changes when the context exits.
    """
    saved = []
    for module in model.modules():
        for patcher in _MODEL_PATCHERS:
            result = patcher(module)
            if result is not None:
                attr, replacement = result
                saved.append((module, attr, getattr(module, attr)))
                setattr(module, attr, replacement)
    try:
        yield
    finally:
        for module, attr, original in reversed(saved):
            setattr(module, attr, original)


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
def patch_forward_signature(model: "PreTrainedModel", inputs: dict[str, Any]):
    """Temporarily replace `model.forward` with a flat explicit signature derived from `inputs`.

    `torch.export` infers the exported function signature from `model.forward.__signature__`.
    Most transformers models use `**kwargs: Unpack[TransformersKwargs]`, which causes
    `torch.export` to expand the signature into a large `combined_args` bundle that
    mismatches the `dynamic_shapes` dict. This patch replaces the forward with a
    minimal signature containing only the keys present in `inputs`.
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
