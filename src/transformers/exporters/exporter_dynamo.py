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

"""Dynamo exporter: `torch.export.export(strict=False)` plus what makes Transformers models traceable.

In execution order:

1. **Signature patch** (`patch_forward_signature`): a flat explicit `forward` signature built from the sample
   inputs, so `**kwargs` does not become a `combined_args` bundle that mismatches `dynamic_shapes`.
2. **Model patches** (`apply_patches("dynamo")`): reversible swaps of non-exportable modeling patterns
   (data-dependent loops, in-place ops, mask checks), including the varlen vision attention.
3. **Pytree registration** (`register_cache_pytrees_for_model`): flatten/unflatten for `Cache` subclasses and
   custom containers.
4. **Dynamic shapes** (`get_auto_dynamic_shapes`): `Dim.AUTO` for every input axis the architecture does not fix,
   when `DynamoConfig.dynamic=True`.
5. **State cleanup** (`reset_model_state`): stateful module attributes cleared for the trace and restored after.
6. **Unused-weight removal** (`drop_unused_weights`): parameters the graph never reads are dropped, since a
   component is traced from a wrapper holding the whole model.
"""

from __future__ import annotations

import copy
import inspect
import sys
import types
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_detectron2_available, is_torch_available, torch_compilable_check
from .base import HfExporter
from .configs import DynamoConfig, ExportFormat
from .metadata import (
    build_export_metadata,
)
from .utils import (
    _class_to_path,
    _path_to_class,
    apply_patches,
    patch_attributes,
    prepare_for_export,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.export.graph_signature import ExportGraphSignature

    from ..cache_utils import Cache
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


class DynamoExporter(HfExporter):
    """Exporter that converts a [`PreTrainedModel`] to an `ExportedProgram`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_dynamo import DynamoExporter, DynamoConfig

    >>> exporter = DynamoExporter()
    >>> exported_artifacts = exporter.export(model, inputs, config=DynamoConfig(dynamic=True))
    >>> outputs = exported_artifacts.module()(**inputs)
    ```
    """

    export_format = ExportFormat.DYNAMO
    config_class = DynamoConfig
    artifact_suffix = ".pt2"

    required_packages = ["torch"]
    min_versions = {"torch": "2.11.0"}
    tested_versions = {"torch": "2.13.0"}

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: DynamoConfig | dict[str, Any],
    ) -> ExportedProgram:
        config = self._as_config(config)

        model, sample_inputs, output_flags = prepare_for_export(model, sample_inputs)

        dynamic_shapes = config.dynamic_shapes
        if config.dynamic and dynamic_shapes is None:
            logger.warning_once(
                "`dynamic=True` with no explicit `dynamic_shapes` marks every input axis `Dim.AUTO`, so "
                "torch.export resolves symbolic shapes for all of them — including axes that are actually "
                "fixed (batch, a size-1 decode step, num_heads/head_dim). Passing explicit `dynamic_shapes` "
                "that mark only the axes which vary bypasses that symbolic-shape resolution and exports "
                "significantly faster."
            )
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

        exported_program = drop_unused_weights(exported_program)

        metadata = build_export_metadata(model, sample_inputs, exported_program, self.required_packages)
        return exported_program, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        torch.export.save(artifact, str(path))


# ── Stage 1: Model signature patch ──────────────────────────────────────────


@contextmanager
def patch_model_config(model: PreTrainedModel, output_flags: dict[str, Any]):
    """Reversibly apply `output_flags` (popped by `prepare_for_export`) onto `model.config` for the trace.

    Flags that are `None` or that the config doesn't declare are skipped.
    """
    config_patches = []
    for flag, value in output_flags.items():
        if value is None or not hasattr(model, "config") or not hasattr(model.config, flag):
            continue
        config_patches.append((model.config, flag, lambda _original, v=value: v))
    with patch_attributes(config_patches):
        yield


@contextmanager
def patch_forward_signature(model: PreTrainedModel, inputs: dict[str, Any]):
    """Temporarily replace `model.forward` with a flat explicit signature derived from `inputs`.

    With `**kwargs` in the signature, `torch.export` builds a `combined_args` bundle that mismatches the
    `dynamic_shapes` dict.
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
# Reversible class-attribute swaps applied during tracing via `apply_patches("dynamo")`, for patterns too
# model-specific to fix in modeling code.


@register_patch(
    "dynamo",
    "transformers.cache_utils.DynamicSlidingWindowLayer.get_mask_sizes",
    "transformers.cache_utils.DynamicSlidingWindowLayer.get_seq_length",
)
def _patch_sliding_window_length(original):
    """Read a growing sliding layer's length off its keys tensor instead of its own counter.

    The python-int counter is baked as a constant, pinning the graph to the traced step (a tree-spec
    `cumulative_length` mismatch). Below the window `keys.shape[-2]` is the same quantity, and symbolic.
    """

    def patch(self, *args, **kwargs):
        keys = getattr(self, "keys", None)
        cached = keys.shape[-2] if keys is not None else 0
        with patch_attributes([(self, "cumulative_length", lambda _original: cached)]):
            return original(self, *args, **kwargs)

    return patch


@register_patch("dynamo", "transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier")
def _patch_classifier_cast(_original):
    """Disable classifier dtype cast in nllb-moe (not traceable)."""
    return lambda self, *args, **kwargs: None


@register_patch("dynamo", "torch.nn.functional.scaled_dot_product_attention")
def _patch_sdpa(original):
    """Route SDPA through the MATH backend on CPU during tracing.

    CPU flash/efficient paths guard on ``Eq(batch, 1)`` (https://github.com/pytorch/pytorch/issues/180202),
    a ``GuardOnDataDependentSymNode`` when the batch comes from e.g. ``pixel_values[bool_mask]``.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def patch(query, *args, **kwargs):
        if query.device.type == "cpu":
            with sdpa_kernel(SDPBackend.MATH):
                return original(query, *args, **kwargs)
        return original(query, *args, **kwargs)

    return patch


@register_patch(
    "dynamo",
    "transformers.utils.import_utils.is_kernels_available",
    "transformers.utils.is_kernels_available",
    # Modules that rebind the name locally need their own override.
    "transformers.modeling_utils.is_kernels_available",
    "transformers.models.sam3_video.modeling_sam3_video.is_kernels_available",
    "transformers.models.mra.modeling_mra.is_kernels_available",
    "transformers.models.rwkv.modeling_rwkv.is_kernels_available",
    "transformers.models.yoso.modeling_yoso.is_kernels_available",
)
def _patch_is_kernels_available(_original):
    """Disable the ``kernels`` library during export; its native kernels are not traceable."""
    return lambda *args, **kwargs: False


# --- Chunked vision/audio attention ─────────────────────────────────────────
# Packed `cu_seqlens` encoders loop `split -> per-segment SDPA -> cat`, which can't be traced; replaced by
# one `_varlen_attn` op, which ONNX and ExecuTorch lower to a `cu_seqlens`-built masked SDPA.


def _varlen_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    returns_tuple: bool = False,
    **kwargs,
):
    """Export-safe chunked vision/audio attention: one varlen attention over the `cu_seqlens` segments."""

    # NaViT-style `(1, T, D)` packing (minicpmv4_6) to flat `(T, D)`.
    needs_batch_restore = hidden_states.ndim == 3
    if needs_batch_restore:
        hidden_states = hidden_states.squeeze(0)

    seq_length = hidden_states.shape[0]
    torch_compilable_check(
        seq_length != 0,
        "Chunked vision attention received an empty input.",
    )

    if hasattr(self, "qkv"):
        # GQA (e.g. Exaone4.5) splits the packed qkv asymmetrically.
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
        # Each encoder's own ``apply_rotary_pos_emb_vision``; signatures differ per model.
        apply_rotary_pos_emb_vision = sys.modules[type(self).__module__].apply_rotary_pos_emb_vision
        if isinstance(position_embeddings, (tuple, list)):
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        else:
            # Single rotary tensor (Qwen2.5/3 Omni).
            query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), position_embeddings).squeeze(0)
            key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), position_embeddings).squeeze(0)

    # `seq_length` bounds `max_q`/`max_k` without a data-dependent `.max()`.
    # The flash kernel needs head_dim % 8 == 0; otherwise emit the masked SDPA directly.
    enable_gqa = getattr(self, "num_key_value_heads", self.num_heads) != self.num_heads
    if query_states.shape[-1] % 8 == 0:
        from torch.nn.attention.varlen import varlen_attn

        cu = cu_seqlens.to(torch.int32)
        attn_output = varlen_attn(
            query_states,
            key_states,
            value_states,
            cu,
            cu,
            seq_length,
            seq_length,
            scale=self.scaling,
            enable_gqa=enable_gqa,
        )
    else:
        attn_output = varlen_attn_masked_sdpa(
            query_states, key_states, value_states, cu_seqlens, scale=self.scaling, enable_gqa=enable_gqa
        )
    attn_output = attn_output.reshape(seq_length, -1)
    out_proj = self.proj if hasattr(self, "proj") else self.out_proj
    attn_output = out_proj(attn_output)

    if needs_batch_restore:
        attn_output = attn_output.unsqueeze(0)

    return (attn_output, None) if returns_tuple else attn_output


# Named so `needs_half_precision_export` can tell which models hit the (bf16-only) varlen flash path.
_VARLEN_ATTENTION_PATHS = (
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
    "transformers.models.cohere_compass.modeling_cohere_compass.CohereCompassVisionAttention.forward",
    "transformers.models.exaone4_5.modeling_exaone4_5.Exaone4_5_VisionAttention.forward",
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniVisionAttention.forward",
    "transformers.models.kimi_k25.modeling_kimi_k25.Kimi_K25VisionAttention.forward",
    "transformers.models.muse_glimmer.modeling_muse_glimmer.MuseGlimmerVisionAttention.forward",
    "transformers.models.video_llama_3.modeling_video_llama_3.VideoLlama3VisionAttention.forward",
    "transformers.models.paddleocr_vl.modeling_paddleocr_vl.PaddleOCRVisionAttention.forward",
    "transformers.models.minicpmv4_6.modeling_minicpmv4_6.MiniCPMV4_6VisionAttention.forward",
    "transformers.models.minicpmv4_7.modeling_minicpmv4_7.MiniCPMV4_7VisionAttention.forward",
    # Audio attention
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniAudioAttention.forward",
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeAudioAttention.forward",
    "transformers.models.qwen3_asr.modeling_qwen3_asr.Qwen3ASRAudioAttention.forward",
)


@register_patch("dynamo", *_VARLEN_ATTENTION_PATHS)
def _patch_chunked_vision_attention(original):
    """Bind `returns_tuple` once per class by inspecting the original forward's source."""
    src = inspect.getsource(original)
    returns_tuple = "return attn_output, attn_weight" in src or "return attn_output, None" in src

    def forward(self, *args, **kwargs):
        return _varlen_vision_attention_forward(self, *args, returns_tuple=returns_tuple, **kwargs)

    return forward


def varlen_attn_masked_sdpa(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k=None,
    max_q=None,
    max_k=None,
    is_causal=False,
    scale=None,
    window_size=None,
    enable_gqa=False,
    seqused_k=None,
    block_table=None,
    num_splits=None,
):
    """Block-diagonal masked SDPA over the packed `(total, heads, dim)` sequence; the exportable equivalent
    of `torch_attn::_varlen_attn`.

    Registered as the op's CPU kernel, so unsupported arguments are refused rather than silently ignored.
    """
    # `(-1, -1)` is "no window", `(-1, 0)` the causal mask `is_causal` already carries
    if window_size is not None and tuple(window_size) in ((-1, -1), (-1, 0)):
        window_size = None
    unsupported = {"window_size": window_size, "seqused_k": seqused_k, "block_table": block_table}
    if named := [name for name, value in unsupported.items() if value is not None]:
        raise NotImplementedError(
            f"`varlen_attn_masked_sdpa` has no implementation for {named}; it masks whole segments only. "
            "Run this attention on a device with the flash kernel, or extend the mask built below."
        )
    positions = torch.arange(query.shape[0], device=query.device)
    segment_id = (positions[:, None] >= cu_seq_q[1:][None, :]).sum(-1)
    block_mask = (segment_id[:, None] == segment_id[None, :])[None, None]
    if is_causal:
        # Within the block mask, a global lower-triangular mask is per-segment causality.
        block_mask = block_mask & (positions[:, None] >= positions[None, :])[None, None]
    q, k, v = (tensor.transpose(0, 1)[None] for tensor in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=block_mask, scale=scale, enable_gqa=enable_gqa
    )
    return out[0].transpose(0, 1)


# torch ships `torch_attn::_varlen_attn` with a CUDA flash kernel only: add a CPU kernel, or on older torch
# define the op with the masked SDPA as `CompositeImplicitAutograd`.
if is_torch_available():

    def _varlen_attn_op_kernel(*args, **kwargs):
        # Schema returns `(output, softmax_lse, rng_state)`; the aux outputs are empty stubs.
        out = varlen_attn_masked_sdpa(*args, **kwargs)
        return out, out.new_empty(0), out.new_empty(0)

    try:
        import torch.nn.attention.varlen  # noqa: F401  # registers `torch_attn::_varlen_attn`

        torch.library.register_kernel("torch_attn::_varlen_attn", "cpu", _varlen_attn_op_kernel)
    except ImportError:
        torch.library.define(
            "torch_attn::_varlen_attn",
            "(Tensor query, Tensor key, Tensor value, Tensor cu_seq_q, Tensor? cu_seq_k, SymInt max_q, "
            "SymInt max_k, bool is_causal=False, float? scale=None, SymInt[]? window_size=None, "
            "bool enable_gqa=False, Tensor? seqused_k=None, Tensor? block_table=None, "
            "SymInt? num_splits=None) -> (Tensor, Tensor, Tensor)",
        )
        torch.library.register_kernel("torch_attn::_varlen_attn", "CompositeImplicitAutograd", _varlen_attn_op_kernel)


# ── Stage 3: Pytree registration ─────────────────────────────────────────────
# The generic flattener serialises any object to a JSON-native context while collecting its tensors.


def _maybe_sym_constant(sym: Any) -> Any:
    """The concrete value a ``Sym*`` has already specialized to, or ``None`` if it's still dynamic."""
    node = sym.node
    if isinstance(sym, torch.SymBool):
        return node.maybe_as_bool()
    if isinstance(sym, torch.SymFloat):
        return node.maybe_as_float()
    return node.maybe_as_int()


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
        # A specialized Sym* is baked as a scalar: as a leaf it can be a SymInt at one trace and a python int
        # at the retrace, and the flipping leaf count breaks `treespec.unflatten` (deepseek_v4).
        const = _maybe_sym_constant(obj)
        if const is not None:
            return const
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
    if isinstance(obj, types.MethodType):
        # e.g. recurrent_gemma binds methods onto its `DynamicCache`; not exportable.
        raise TypeError("Cannot flatten a bound method for pytree context")
    if hasattr(obj, "__dict__"):
        attributes = dict(vars(obj))
        # Sliding-layer step counters would pin the graph to the traced step (see
        # `_patch_sliding_window_length`); normalised before the walk so no SymInt leaf is orphaned.
        if "sliding_window" in attributes and not isinstance(attributes.get("cumulative_length", 0), torch.Tensor):
            attributes["cumulative_length"] = 0
        if "cumulative_length_int" in attributes:
            attributes["cumulative_length_int"] = 0
        # `generate`'s mark on a cache the caller passed in (an assistant's, in assisted decoding), not structure
        attributes.pop("_is_user_defined", None)
        return {
            "_t": "obj",
            "p": _class_to_path(cls),
            "s": {k: _flatten_to_context(v, tensors) for k, v in attributes.items()},
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
        instance = cls.__new__(cls)
        for k, v in ctx["s"].items():
            instance.__dict__[k] = _unflatten_from_context(v, tensors)
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


def register_pytree_node(object_cls: type):
    """Register a class (e.g. `StaticCache`) as a torch.export pytree node."""
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


def is_cache_class(cls: type) -> bool:
    """Whether ``cls`` is a cache type — a [`Cache`] subclass, or a model-specific class following
    the ``*Cache`` naming convention (e.g. ``xLSTMCache``, ``MimiConv1dPaddingCache``)."""
    return issubclass(cls, Cache) or cls.__name__.endswith("Cache")


def is_cache_object(value: Any) -> bool:
    """Whether ``value`` is a cache, by the rule [`register_cache_pytrees_for_model`] uses."""
    return is_cache_class(type(value))


def register_cache_pytrees_for_model(model: PreTrainedModel):
    """Register all relevant cache types as pytree nodes for torch.export."""
    for cache_type in _iter_subclasses(Cache):
        register_pytree_node(cache_type)

    # Per-model caches not inheriting from Cache
    for _, obj in inspect.getmembers(inspect.getmodule(model)):
        if inspect.isclass(obj) and obj.__module__ == model.__class__.__module__ and is_cache_class(obj):
            register_pytree_node(obj)

    # detectron2 ImageList (used by layoutlmv2)
    if is_detectron2_available() and isinstance(model, PreTrainedModel) and model.config.model_type == "layoutlmv2":
        from detectron2.structures.image_list import ImageList

        register_pytree_node(ImageList)


# ── Stage 4: Dynamic shapes ─────────────────────────────────────────────────


def architecture_axes(name: str, tensor: torch.Tensor) -> tuple[int, ...]:
    """The axes of one input the architecture fixes, kept out of `Dim.AUTO`.

    The embedding feature axis, and m-rope `position_ids`' section axis, which models branch on in Python
    (glm4v's `position_ids.shape[0] == 4`).
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.dim():
        return ()
    if name in ("inputs_embeds", "decoder_inputs_embeds"):
        return (tensor.dim() - 1,)
    if name in ("position_ids", "decoder_position_ids") and tensor.dim() == 3:
        return (0,)
    return ()


def _auto_dynamic_shape(
    tensor: torch.Tensor, is_cache_tensor: bool = False, static_axes: tuple[int, ...] = ()
) -> dict[int, torch.export.Dim]:
    """Generate a dynamic shape with all dimensions set to Dim.AUTO, except `static_axes`.

    A rank-4 KV cache tensor keeps heads and head_dim static, so the runtime can read the cache geometry
    back off the graph.
    """
    static_dims = (1, 3) if is_cache_tensor and tensor.dim() == 4 else ()
    static_dims = (*static_dims, *static_axes)
    return {dim: torch.export.Dim.AUTO for dim in range(tensor.dim()) if dim not in static_dims}


def get_auto_dynamic_shapes(inputs: Any, is_cache_tensor: bool = False, static_axes: tuple[int, ...] = ()) -> Any:
    """Recursively build dynamic shapes for any input value, mirroring its pytree structure.

    Registered pytree nodes yield one spec per child of their flatten; recursing through a ``Cache`` marks the
    tensors below it as cache state.
    """
    if isinstance(inputs, torch.Tensor):
        return _auto_dynamic_shape(inputs, is_cache_tensor, static_axes)
    if inputs is None or isinstance(inputs, (int, float, bool, str)):
        return None
    if type(inputs) in (list, tuple, set, frozenset):
        return type(inputs)(get_auto_dynamic_shapes(v, is_cache_tensor) for v in inputs)
    if type(inputs) is dict:
        return {k: get_auto_dynamic_shapes(v, is_cache_tensor, architecture_axes(k, v)) for k, v in inputs.items()}
    if (node := torch.utils._pytree.SUPPORTED_NODES.get(type(inputs))) is not None:
        # One level only: a `ModelOutput` must keep one child per field, not a flat tensor list.
        children, _ = node.flatten_fn(inputs)
        return [get_auto_dynamic_shapes(child, is_cache_tensor or isinstance(inputs, Cache)) for child in children]
    if hasattr(inputs, "__dict__"):
        leaves, _ = _pytree_flatten(inputs)
        return get_auto_dynamic_shapes(leaves, is_cache_tensor or isinstance(inputs, Cache))
    return None


# ── Stage 5: Model state cleanup ────────────────────────────────────────────
# Tracing can leave FakeTensors in non-Cache stateful attributes, and stale eager state can leak into the trace.

_STATEFUL_CACHE_ATTRS = (
    "cached_rotary_positional_embedding",  # wav2vec2_bert, seamless_m4t, clvp
    "cached_sequence_length",  # wav2vec2_bert, seamless_m4t, clvp
)


@contextmanager
def reset_model_state(model: torch.nn.Module):
    """Save each `_STATEFUL_CACHE_ATTRS` value, null it for the trace, restore on exit."""
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


# ── Stage 6: Unused-weight removal ──────────────────────────────────────────


def drop_unused_weights(exported_program: ExportedProgram) -> ExportedProgram:
    """Drop the parameters and buffers the graph never reads.

    A decomposed component is traced from a wrapper holding the whole model, so `torch.export` lifts every
    weight (a four-component video_llava export came to 3.2x the model's weights).
    """
    signature = exported_program.graph_signature
    lifted = {**signature.inputs_to_parameters, **signature.inputs_to_buffers}
    # Mutated buffers stay even when the body never reads them.
    mutated = {spec.arg.name for spec in signature.output_specs if hasattr(spec.arg, "name")}
    graph_module = copy.deepcopy(exported_program.graph_module)
    unused = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and node.name in lifted and not node.users and node.name not in mutated
    ]
    if not unused:
        return exported_program

    dropped = {node.name for node in unused}
    for node in unused:
        graph_module.graph.erase_node(node)
    graph_module.recompile()
    input_specs = [spec for spec in signature.input_specs if getattr(spec.arg, "name", None) not in dropped]
    weights = {lifted[name] for name in dropped}
    state_dict = {name: value for name, value in exported_program.state_dict.items() if name not in weights}
    return exported_program._update(
        graph_module, ExportGraphSignature(input_specs, signature.output_specs), state_dict=state_dict
    )
