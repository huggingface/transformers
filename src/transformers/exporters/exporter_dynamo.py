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

Wraps `torch.export.export(strict=False)` with helpers that make Transformers
models exportable. The export pipeline uses five sections, in execution order:

1. **Model signature patch** (`patch_forward_signature`): replaces `model.forward`
   with a flat explicit signature derived from `sample_inputs` so `torch.export` does
   not expand `**kwargs` into a `combined_args` bundle that mismatches `dynamic_shapes`.
   This is the entry contract `torch.export` reads before tracing.
2. **Model patches** (`_PATCHES["dynamo"]` via `apply_patches("dynamo")`): reversible
   class-attribute swaps applied during tracing to replace non-exportable model patterns
   (data-dependent loops, in-place ops, mask checks) with export-safe equivalents.
   Modeling code itself is not updated because these patches are too model-specific.
3. **Pytree registration** (`register_cache_pytrees_for_model`): flatten/unflatten
   hooks (via `torch.utils._pytree.register_pytree_node`) for Cache subclasses and
   custom containers so `torch.export` can trace through them.
4. **Dynamic shapes** (`get_auto_dynamic_shapes`): automatic `Dim.AUTO` inference
   for all tensor and cache inputs when `DynamoConfig.dynamic=True`.
5. **Model state cleanup** (`reset_model_state`): non-Cache stateful module attributes
   (`_STATEFUL_CACHE_ATTRS`) are saved on entry, set to `None` during the trace, and
   restored on exit — so a previous eager forward doesn't leak into the trace and any
   FakeTensors the trace planted are discarded before the next eager forward.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_detectron2_available, is_torch_available, torch_compilable_check
from .base import HfExporter
from .configs import DynamoConfig
from .utils import apply_patches, patch_attributes, prepare_for_export, register_patch


if is_torch_available():
    import torch
    from torch.export import ExportedProgram

    from ..cache_utils import Cache
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    """Exporter that converts a [`PreTrainedModel`] to an `ExportedProgram`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

    >>> exporter = DynamoExporter()
    >>> exported = exporter.export(model, inputs, config=DynamoConfig(dynamic=True))
    >>> outputs = exported.module()(**inputs)
    ```
    """

    required_packages = ["torch"]
    tested_versions = {"torch": "2.12.0"}

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: DynamoConfig | dict[str, Any],
    ) -> ExportedProgram:
        if isinstance(config, dict):
            config = DynamoConfig(**config)
        elif not isinstance(config, DynamoConfig):
            raise TypeError(f"Expected config to be a DynamoConfig or dict, got {type(config)}")

        model, sample_inputs, output_flags = prepare_for_export(model, sample_inputs)

        dynamic_shapes = config.dynamic_shapes
        if config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)

        register_cache_pytrees_for_model(model)

        with (
            apply_patches("dynamo"),
            reset_model_state(model),
            patch_model_config(model, output_flags),
            patch_forward_signature(model, sample_inputs),
        ):
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                kwargs=copy.deepcopy(dict(sample_inputs)),
                strict=config.strict,
                dynamic_shapes=dynamic_shapes,
                prefer_deferred_runtime_asserts_over_guards=config.prefer_deferred_runtime_asserts_over_guards,
            )

        return exported_program


# ── Stage 1: Model signature patch ──────────────────────────────────────────
# Replaces `model.forward` with a flat explicit signature derived from the
# inputs dict so `torch.export` does not expand `**kwargs` into a large bundle.
# `patch_model_config` lives here too — it strips output flags from the inputs
# and applies them onto `model.config` for the duration of the trace.


# Output flags stripped from inputs and applied onto `model.config` for the trace.
@contextmanager
def patch_model_config(model: PreTrainedModel, output_flags: dict[str, Any]):
    """Reversibly tweak `model.config` for the trace:

    - Applies `output_flags` (popped from inputs by `prepare_for_export`) onto
      `model.config.<flag>` so the model picks them up via its usual `<flag> if <flag> is
      not None else self.config.<flag>` fallback.
    - Disables `use_mamba_kernels` on every submodel's config that declares it (mamba/jamba
      kernels are not exportable).

    Originals are restored on exit. Flags whose value is `None`, or that the config doesn't
    declare, are silently skipped — useful for submodels that don't accept every parent flag.
    """
    config_patches = []
    for flag, value in output_flags.items():
        if value is None or not hasattr(model, "config") or not hasattr(model.config, flag):
            continue
        config_patches.append((model.config, flag, lambda _original, v=value: v))
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_mamba_kernels"):
            config_patches.append((module.config, "use_mamba_kernels", lambda _original: False))
    with patch_attributes(config_patches):
        yield


@contextmanager
def patch_forward_signature(model: PreTrainedModel, inputs: dict[str, Any]):
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


# ── Stage 2: Model patches ────────────────────────────────────────────────────
# Reversible class-attribute swaps applied during `torch.export` tracing via
# `apply_patches("dynamo")`. Each replaces a non-exportable model pattern
# (data-dependent control flow, in-place ops on views, etc.) with an
# export-safe equivalent on the owning class — every live instance sees the
# replacement until the context exits. Modeling code itself is not updated
# because these patches are too model-specific; we do strive to keep modeling
# code compliant where reasonable.
#
# Each `@register_patch("dynamo", *dotted_paths)` decorator targets one or
# more `Class.method` paths and wraps a `factory(original) -> replacement`.
# Multiple paths share the same factory when the same method shape needs to be
# swapped on several classes (e.g. `_update_mamba_mask` on Jamba/Bamba/…).


@register_patch("dynamo", "transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier")
def _patch_classifier_cast(_original):
    """Disable classifier dtype cast in nllb-moe (not traceable)."""
    return lambda self, *args, **kwargs: None


# --- Chunked vision/audio attention ─────────────────────────────────────────
# Sub-encoders that pack multiple variable-length sequences into one flat tensor
# with `cu_seqlens` markers fall back to `split → per-segment SDPA → cat` in the
# unpatched forward, which is a Python loop that `torch.export` can't trace.
# `_reshaped_vision_attention_forward` replaces that loop with a reshape into a
# per-segment batch followed by a single SDPA call. It handles the layout
# differences across encoders (combined `qkv` vs separate `q/k/v` vs separate
# `q_proj/k_proj/v_proj`, asymmetric `q_dim/kv_dim` split, `(cos, sin)` vs single
# rotary tensor vs none, `.proj` vs `.out_proj`, NaViT `(1, T, D)` packing,
# tuple vs single return). The `returns_tuple` flag is bound once per class at
# install time by inspecting the original `forward`'s source.
#
# NOTE: this whole stack of patches becomes unnecessary once transformers adopts a
# proper varlen-attention op (e.g. PyTorch's `torch._nested.scaled_dot_product_attention`
# or a Flex-Attention varlen kernel) — the modeling forwards can then express the
# segmented attention directly with `cu_seqlens` and trace through `torch.export`
# without this reshape-into-batch workaround. Drop this section when that lands.


def _reshaped_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    returns_tuple: bool = False,
    **kwargs,
):
    """Export-safe chunked vision/audio attention: reshape segments into a batch dim,
    apply rotary if provided, run one SDPA call, project, and re-emit in the original layout."""

    # Normalise NaViT-style `(1, T, D)` packing (minicpmv4_6) to the flat `(T, D)` layout
    # the rest of this wrapper assumes. The leading dim is always 1 — multi-image batches
    # are packed along the sequence dim.
    needs_batch_restore = hidden_states.ndim == 3
    if needs_batch_restore:
        hidden_states = hidden_states.squeeze(0)

    seq_length = hidden_states.shape[0]
    num_segments = cu_seqlens.shape[0] - 1
    torch_compilable_check(
        seq_length % num_segments == 0,
        "Chunked vision attention requires uniform segment lengths during export. "
        "Ensure all images have the same resolution (use do_resize=True in the processor) "
        "or pad inputs to a common size.",
    )

    if hasattr(self, "qkv"):
        # Grouped-query attention (q_dim != kv_dim, e.g. Exaone4.5) splits asymmetrically;
        # uniform reshape into (seq, 3, num_heads, -1) only works when Q, K, V share the head count.
        if hasattr(self, "q_dim") and hasattr(self, "kv_dim") and self.q_dim != self.kv_dim:
            query_states, key_states, value_states = self.qkv(hidden_states).split(
                [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
            )
            query_states = query_states.view(seq_length, self.num_heads, self.head_dim)
            key_states = key_states.view(seq_length, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(seq_length, self.num_key_value_heads, self.head_dim)
        else:
            query_states, key_states, value_states = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).transpose(0, 1).unbind(0)
            )
    else:
        q_proj = getattr(self, "q_proj", getattr(self, "q", None))
        k_proj = getattr(self, "k_proj", getattr(self, "k", None))
        v_proj = getattr(self, "v_proj", getattr(self, "v", None))
        query_states = q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

    if position_embeddings is not None:
        apply_rotary_pos_emb_vision = sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
        if isinstance(position_embeddings, (tuple, list)):
            # (cos, sin) tuple convention — most VL encoders.
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        else:
            # Single `rotary_pos_emb` tensor convention — Qwen2.5/3 Omni vision applies rotary per-states.
            query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), position_embeddings).squeeze(0)
            key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), position_embeddings).squeeze(0)

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
        enable_gqa=getattr(self, "num_key_value_heads", self.num_heads) != self.num_heads,
    )

    # (n_seg, heads, seg_len, dim) → (n_seg, seg_len, heads, dim) → (seq, heads*dim)
    attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
    out_proj = self.proj if hasattr(self, "proj") else self.out_proj
    attn_output = out_proj(attn_output)

    if needs_batch_restore:
        attn_output = attn_output.unsqueeze(0)

    return (attn_output, None) if returns_tuple else attn_output


@register_patch(
    "dynamo",
    # Combined `qkv` + `(cos, sin)` rotary + `.proj`
    "transformers.models.qwen2_vl.modeling_qwen2_vl.VisionAttention.forward",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward",
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionAttention.forward",
    "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeVisionAttention.forward",
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionAttention.forward",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeVisionAttention.forward",
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeVisionAttention.forward",
    "transformers.models.glm4v.modeling_glm4v.Glm4vVisionAttention.forward",
    "transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeVisionAttention.forward",
    "transformers.models.glm_ocr.modeling_glm_ocr.GlmOcrVisionAttention.forward",
    "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe.Ernie4_5_VLMoeVisionAttention.forward",
    # Asymmetric `qkv` split + `(cos, sin)` rotary + `.proj`
    "transformers.models.exaone4_5.modeling_exaone4_5.Exaone4_5_VisionAttention.forward",
    # Combined `qkv` + no in-attention rotary + `.proj`
    "transformers.models.glm_image.modeling_glm_image.GlmImageVisionAttention.forward",
    # Separate `.q` / `.k` / `.v` + single rotary tensor + `.proj`
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniVisionAttention.forward",
    # Separate `_proj` + `(cos, sin)` rotary + `.out_proj` (tuple return)
    "transformers.models.video_llama_3.modeling_video_llama_3.VideoLlama3VisionAttention.forward",
    "transformers.models.paddleocr_vl.modeling_paddleocr_vl.PaddleOCRVisionAttention.forward",
    # NaViT (1, T, D) + separate `_proj` + `.out_proj` (tuple return)
    "transformers.models.minicpmv4_6.modeling_minicpmv4_6.MiniCPMV4_6VisionAttention.forward",
    # Audio attention: separate `_proj` + `.out_proj`, no rotary
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniAudioAttention.forward",
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeAudioAttention.forward",
)
def _patch_chunked_vision_attention(original):
    """Bind `returns_tuple` once per class by inspecting the original forward's source."""
    src = inspect.getsource(original)
    returns_tuple = "return attn_output, attn_weight" in src or "return attn_output, None" in src

    def forward(self, *args, **kwargs):
        return _reshaped_vision_attention_forward(self, *args, returns_tuple=returns_tuple, **kwargs)

    return forward


# ── Stage 3: Pytree registration ─────────────────────────────────────────────
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


def register_cache_pytrees_for_model(model: PreTrainedModel):
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

    # detectron2 ImageList (used by layoutlmv2)
    if is_detectron2_available() and isinstance(model, PreTrainedModel) and model.config.model_type == "layoutlmv2":
        from detectron2.structures.image_list import ImageList

        _register_pytree_node(ImageList)


# ── Stage 4: Dynamic shapes ─────────────────────────────────────────────────
# Automatic `Dim.AUTO` inference for all tensor and cache inputs when
# `DynamoConfig.dynamic` is True and no explicit `dynamic_shapes` are provided.


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


# ── Stage 5: Model state cleanup ────────────────────────────────────────────
# `torch.export` traces forward with FakeTensors, which can leave non-Cache stateful
# tensor attributes as FakeTensors after tracing — a follow-up eager forward then
# hits shape/dtype mismatches when it reuses the stale state. We also want stale
# eager-mode state cleared on entry so it doesn't leak into the trace.
# `reset_model_state` brackets the `torch.export.export` call: it saves every
# attribute in `_STATEFUL_CACHE_ATTRS` on every submodule, sets them to `None` for
# the trace, and restores the originals on exit (finally semantics).
#
# To register a new stateful attribute: append its name to `_STATEFUL_CACHE_ATTRS`.

_STATEFUL_CACHE_ATTRS = (
    "_cached_decode_position_ids",  # glm_image (m-rope decode position ids)
    "_prefill_len",  # glm_image (m-rope prefill length)
    "cached_rotary_positional_embedding",  # wav2vec2_bert, seamless_m4t, clvp
    "cached_sequence_length",  # wav2vec2_bert, seamless_m4t, clvp
)


@contextmanager
def reset_model_state(model: torch.nn.Module):
    """Save each `_STATEFUL_CACHE_ATTRS` value, null it for the trace, restore on exit.

    FakeTensors that `torch.export` plants into these attributes during the trace are
    discarded by the restore.
    """
    originals = [
        (module, attr, getattr(module, attr))
        for module in model.modules()
        for attr in _STATEFUL_CACHE_ATTRS
        if hasattr(module, attr)
    ]
    for module, attr, _ in originals:
        setattr(module, attr, None)
    try:
        yield
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)
