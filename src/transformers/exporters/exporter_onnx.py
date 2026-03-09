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
import copy
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available
from .exporter_dynamo import DynamoExporter
from .utils import dedup_output_tensors, get_inputs_outputs_names, get_leaf_tensors, prepare_for_export


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.onnx import ONNXProgram

    from .. import masking_utils as _masking_utils_mod

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__file__)


ONNX_UNSUPPORTED_MODEL_TYPES: set[str] = {
    # --- Missing ONNX ops ---
    "splinter",  # aten.bincount has no ONNX decomposition
    # --- ONNX Runtime runtime / graph errors (model-specific, needs per-model investigation) ---
    "fine_acoustics",  # BarkFineModel: attention_mask exported as rank-3 but ORT expects rank-2
}

# The following are models that can be exported but their outputs
# are extremely inaccurate compared to the original model.
ONNX_EXTREMELY_INACCURATE_MODEL_TYPES: set[str] = {
    "blt",  # 94.3% mismatch in last_hidden_state
    "flaubert",  # 40% mismatch in end_top_index (top-k beam search non-determinism)
    "parakeet_ctc",  # 100% NaN in logits
    "parakeet_encoder",  # 100% NaN in last_hidden_state
    "patchtst",  # NaN loss output
    "pp_doclayout_v3",  # 68.3% mismatch in enc_topk_bboxes
    "d_fine",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "mm-grounding-dino",  # 25% mismatch in encoder_pred_boxes (non-deterministic top-k selection)
    "rt_detr",  # 43.3% mismatch in enc_topk_bboxes
    "rt_detr_v2",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "siglip2",  # 100% mismatch in logits
    "siglip2_vision_model",  # 73.8% mismatch in last_hidden_state
    "vit_mae",  # 99.3% mismatch in ids_restore (random masking)
    "xlm",  # 6.2% mismatch in end_top_index
}


class OnnxExporter(DynamoExporter):
    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ONNXProgram":
        """Exports a model to ONNX format using TorchDynamo.
        Args:
            model (`PreTrainedModel`):
                The model to export.
            sample_inputs (`Dict[str, Any]`):
                The sample inputs to use for the export.
        Returns:
            `ONNXProgram`: The exported model.
        """
        if model.config.model_type in ONNX_UNSUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} is not supported for model type '{model.config.model_type}'."
            )

        if model.config.model_type in ONNX_EXTREMELY_INACCURATE_MODEL_TYPES:
            raise NotImplementedError(
                f"Exporting a model of type '{model.config.model_type}' results in an ONNX model with extremely inaccurate outputs."
            )

        # we use a copy to avoid side effects
        inputs = copy.deepcopy(sample_inputs)
        model, inputs, outputs = prepare_for_export(model, inputs)
        inputs_names, outputs_names = get_inputs_outputs_names(inputs, outputs)

        with patch_for_onnx_export(model):
            exported_program: ExportedProgram = super().export(model, inputs)
            _sanitize_exported_graph_for_onnx(exported_program.graph_module)
            onnx_program: ONNXProgram = torch.onnx.export(
                exported_program,
                args=(),
                kwargs=inputs,
                f=self.export_config.f,
                input_names=inputs_names,
                output_names=outputs_names,
                opset_version=self.export_config.opset_version,
                external_data=self.export_config.external_data,
                export_params=self.export_config.export_params,
                optimize=self.export_config.optimize,
            )
            # ORT's CUDA EP rejects TopK nodes that don't have sorted=1 explicitly
            # set (e.g. nodes emitted from aten.sort.default which onnxscript translates
            # to TopK without setting the attribute).  Force sorted=1 on every TopK node
            # in the IR; this is always correct since sort/topk callers using sorted=False
            # only care about the selected set, not ordering.
            import onnxscript.ir as onnx_ir

            for ir_node in onnx_program.model.graph.all_nodes():
                if ir_node.op_type == "TopK":
                    ir_node.attributes["sorted"] = onnx_ir.Attr("sorted", onnx_ir.AttributeType.INT, 1)

        return onnx_program


# ── per-node ONNX sanitisation fixers ─────────────────────────────────────────
# Each fixer has signature  (gm: GraphModule, node: Node) -> bool.
# Return True  → node was consumed (replaced/erased); no further fixers run.
# Return False → node is still in the graph; next fixer gets a chance.
# To add a new fix: write an _onnx_fix_* function and append it to
# _ONNX_FX_NODE_FIXES below.


def _onnx_fix_alias(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace alias(x) → x to break the alias → detach_ → index_put_ chain.

    ``tensor[bool_mask] = value`` is traced by FX as ``alias → detach_ → index_put_``.
    The alias creates an aliasing relationship that ``functionalize()`` in step 2
    uses to regenerate ``detach_`` nodes even after they are removed.  Replacing the
    alias with a direct passthrough to its input breaks the chain.
    """
    if node.target is not torch.ops.aten.alias.default:
        return False
    node.replace_all_uses_with(node.args[0])
    gm.graph.erase_node(node)
    return True


def _onnx_fix_detach_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace ``aten.detach_`` (in-place) with ``aten.detach`` (out-of-place).

    The in-place ``detach_`` causes ``assert_functional_graph`` to fail when
    ``run_decompositions`` re-runs AOT autograd in step 2.
    """
    if node.target is not torch.ops.aten.detach_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.detach.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


def _onnx_fix_index_put_bool_mask(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Rewrite ``index_put(self, [bool_mask], values)`` → cumsum-gather-where.

    ``tensor[bool_mask] = values`` lowers ``aten.index.Tensor`` to a ``values``
    tensor of shape ``(n_true, D)``.  The standard ONNX translation emits
    ``Where(mask, values, self)`` which ORT rejects because ``(n_true, D)``
    cannot broadcast against ``(B, S, D)``.  The cumsum approach maps every
    sequence position to its offset in ``values`` without changing the result.
    Handles both the in-place (``index_put_``) and out-of-place (``index_put``)
    variants; the non-boolean in-place case falls through to
    ``_onnx_fix_index_put_inplace``.
    """
    if node.target not in (torch.ops.aten.index_put_.default, torch.ops.aten.index_put.default):
        return False
    self_arg, indices_list, values_arg = node.args[0], node.args[1], node.args[2]
    accumulate = node.args[3] if len(node.args) > 3 else False
    if (
        accumulate
        or len(indices_list) != 1
        or not isinstance(indices_list[0], torch.fx.Node)
        or indices_list[0].meta.get("val") is None
        or indices_list[0].meta["val"].dtype != torch.bool
    ):
        return False
    mask_node = indices_list[0]
    mask_shape = list(mask_node.meta["val"].shape)
    self_val = self_arg.meta.get("val")
    values_val = values_arg.meta.get("val") if isinstance(values_arg, torch.fx.Node) else None
    if self_val is not None:
        extra_dims = len(self_val.shape) - len(mask_shape)
    elif values_val is not None:
        extra_dims = max(0, values_val.ndim - 1)
    else:
        extra_dims = 0
    with gm.graph.inserting_before(node):
        expanded_mask = mask_node
        for _ in range(extra_dims):
            expanded_mask = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(expanded_mask, -1))
        flat = gm.graph.call_function(torch.ops.aten.flatten.using_ints, args=(mask_node,))
        flat_int = gm.graph.call_function(
            torch.ops.aten._to_copy.default, args=(flat,), kwargs={"dtype": torch.int64}
        )
        cs = gm.graph.call_function(torch.ops.aten.cumsum.default, args=(flat_int, 0))
        cs_m1 = gm.graph.call_function(torch.ops.aten.add.Scalar, args=(cs, -1))
        positions_flat = gm.graph.call_function(torch.ops.aten.clamp.default, args=(cs_m1,), kwargs={"min": 0})
        # Build the shape from size.int nodes so step-2 re-tracing can track each
        # dimension — passing raw SymInts from meta["val"].shape causes a "not tracked
        # with proxy" error when the batch/seq dimension is symbolic.
        dyn_shape = [
            gm.graph.call_function(torch.ops.aten.size.int, args=(mask_node, i))
            for i in range(len(mask_shape))
        ]
        positions = gm.graph.call_function(torch.ops.aten.view.default, args=(positions_flat, dyn_shape))
        if values_val is not None and values_val.ndim == 0:
            # 0-D scalar: unsqueeze to [1] so the standard gather path can be used.
            # zeros_like(positions) ensures every slot indexes element 0 of the
            # 1-element tensor (no out-of-bounds regardless of n_true).
            # The constant scalar is lifted on CPU; move to self's device so the
            # downstream index.Tensor and where nodes see uniform devices.
            effective_values = gm.graph.call_function(
                torch.ops.aten.unsqueeze.default, args=(values_arg, 0)
            )
            if self_val is not None and values_val.device != self_val.device:
                effective_values = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(effective_values,),
                    kwargs={"device": self_val.device},
                )
            positions_for_index = gm.graph.call_function(
                torch.ops.aten.zeros_like.default, args=(positions,)
            )
            effective_ndim = 1
        else:
            effective_values = values_arg
            positions_for_index = positions
            effective_ndim = values_val.ndim if values_val is not None else 1
        gathered = gm.graph.call_function(
            torch.ops.aten.index.Tensor, args=(effective_values, [positions_for_index])
        )
        # If values has fewer dims than self (e.g. 1-D scalar-per-true-slot with
        # a multi-dim self), the gather produces shape mask_shape instead of
        # self_shape.  Unsqueeze trailing dims so Where can broadcast correctly.
        gather_extra = extra_dims - max(0, effective_ndim - 1)
        for _ in range(gather_extra):
            gathered = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(gathered, -1))
        result = gm.graph.call_function(torch.ops.aten.where.self, args=(expanded_mask, gathered, self_arg))
    node.replace_all_uses_with(result)
    gm.graph.erase_node(node)
    return True


def _onnx_fix_index_put_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Make non-boolean ``aten.index_put_`` out-of-place.

    Runs after ``_onnx_fix_index_put_bool_mask``, so any surviving
    ``index_put_`` node is non-boolean.  The in-place form causes AOT autograd
    to regenerate ``alias → detach_`` during step 2 re-tracing.
    """
    if node.target is not torch.ops.aten.index_put_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.index_put.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


def _onnx_fix_topk_sorted(_gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Force ``sorted=True`` in ``aten.topk`` so ORT's CUDA EP accepts the node.

    ``aten.topk`` translates directly to ONNX ``TopK``.  ORT's CUDA execution
    provider rejects ``TopK(sorted=False)`` (provider type not set), while
    ``TopK(sorted=True)`` is accepted by both CPU and CUDA EP.  Forcing
    ``sorted=True`` is safe: the top-k *set* is unchanged; only the tie-breaking
    order of equal elements may differ, which callers using ``sorted=False`` do
    not rely on by definition.
    """
    if node.target is not torch.ops.aten.topk.default:
        return False
    # aten.topk positional arg 4 is `sorted` (self, k, dim, largest, sorted).
    if len(node.args) >= 5 and node.args[4] is False:
        node.update_arg(4, True)
    return False  # node stays; ONNX translation handles it


def _onnx_fix_sort_stable(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace ``aten.sort.stable`` with ``aten.sort.default``.

    Any ``torch.sort``/``torch.argsort`` call with an explicit ``stable=``
    keyword (even ``stable=False``) dispatches to the ``aten.sort.stable``
    overload, which has no registered ONNX translation and causes step 2 to
    fail with a ConversionError.  ``aten.sort.default`` translates to ONNX
    ``TopK`` with ``sorted=True``, which ORT's CUDA EP accepts.

    Dropping ``stable=`` is safe for export: ONNX ``TopK`` makes no guarantee
    about stable ordering of equal elements, so callers that requested stability
    cannot rely on it in the exported model regardless.
    """
    if node.target is not torch.ops.aten.sort.stable:
        return False
    # aten.sort.stable args: (self, stable, dim=-1, descending=False)
    # aten.sort.default args: (self, dim=-1, descending=False)
    self_arg = node.args[0]
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    descending = node.args[3] if len(node.args) > 3 else node.kwargs.get("descending", False)
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.sort.default, args=(self_arg, dim, descending))
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


# Ordered list of per-node fixers.  The first fixer that returns True consumes
# the node and stops further processing.  Append new entries to extend the
# sanitisation pipeline.
_ONNX_FX_NODE_FIXES = [
    _onnx_fix_alias,
    _onnx_fix_detach_inplace,
    _onnx_fix_index_put_bool_mask,
    _onnx_fix_index_put_inplace,
    _onnx_fix_topk_sorted,
    _onnx_fix_sort_stable,
]


def _sanitize_exported_graph_for_onnx(graph_module: "torch.fx.GraphModule") -> None:
    """Sanitize the exported FX graph before ONNX step 2 (run_decompositions).

    Walks every sub-GraphModule once and applies ``_ONNX_FX_NODE_FIXES`` to each
    ``call_function`` node (the first fixer that returns ``True`` consumes the node
    and stops the chain).  Shape-assertion nodes are erased before the fixer chain
    since they have no outputs to preserve.

    See the individual ``_onnx_fix_*`` functions above for documentation on each
    fix.  To add a new fix, implement an ``_onnx_fix_*`` function and append it to
    ``_ONNX_FX_NODE_FIXES``.
    """
    _assertion_ops = frozenset(
        {
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_tensor_metadata.default,
        }
    )
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target in _assertion_ops:
                gm.graph.erase_node(node)
                continue
            for fix in _ONNX_FX_NODE_FIXES:
                if fix(gm, node):
                    break
        gm.graph.eliminate_dead_code()
        gm.recompile()


@contextmanager
def patch_for_onnx_export(model):
    # ONNX export patcher context
    # This context manager monkey-patches PyTorch ops that are unsupported or buggy in ONNX export.
    # The following ops are patched with fallback implementations or workarounds:
    #   - torch.where / torch.Tensor.where: handles dtype mismatches and scalar operands
    #   - torch.unsqueeze: supports complex tensors
    #   - torch.nn.RMSNorm.forward: bypasses aten._fused_rms_norm when elementwise_affine=False
    #   - torch.nn.functional.scaled_dot_product_attention: supports 5D blocked attention; disables GQA for standard MHA
    #   - torch.randperm: replaced with argsort(rand(n)) using only ONNX-supported ops
    #   - torch.bucketize: replaced with vectorized (boundaries <= x).sum(-1) to avoid scalar constant tensors
    #     that functionalize() wraps with alias+detach_ during step-2 re-tracing, failing assert_functional_graph
    #   - torch.cummax / torch.cummin: decomposed via upper-triangular masked max/min (no native ONNX op)
    #   - torch.Tensor.masked_scatter: replaced with cumsum-gather-where (aten.masked_scatter has no ONNX op and
    #     its flattened source layout produces Where shape mismatches identical to index_put with boolean masks)
    #   - masking_utils._vmap_expansion_sdpa: replaced with broadcast-based expansion (vmap higher-order ops fail in step-2 AOT re-tracing)
    #   - model.forward: clones any tensor that appears more than once in the output structure (deduplication)
    # These patches are only active during export and are reverted afterwards.

    original_forward = model.forward

    original_cummax = torch.cummax
    original_cummin = torch.cummin
    original_torch_where = torch.where
    original_randperm = torch.randperm
    original_bucketize = torch.bucketize
    original_tensor_where = torch.Tensor.where
    original_torch_unsqueeze = torch.unsqueeze
    original_tensor_cummax = torch.Tensor.cummax
    original_tensor_cummin = torch.Tensor.cummin
    original_tensor_unsqueeze = torch.Tensor.unsqueeze
    original_rms_norm_forward = torch.nn.RMSNorm.forward
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    original_masked_scatter = torch.Tensor.masked_scatter

    original_vmap_expansion_sdpa = _masking_utils_mod._vmap_expansion_sdpa

    def _torch_where(condition, x=None, y=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
            y = y.to(x.dtype)
        elif isinstance(x, torch.Tensor) and isinstance(y, (int, float, bool)):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(y, torch.Tensor) and isinstance(x, (int, float, bool)):
            x = torch.tensor(x, dtype=y.dtype, device=y.device)
        if x is None and y is None:
            return original_torch_where(condition)
        elif y is None:
            return original_torch_where(condition, x)
        else:
            return original_torch_where(condition, x, y)

    def _tensor_where(self, condition, other):
        return _torch_where(condition, self, other)

    def _unsqueeze(self_or_input, dim):
        if torch.is_complex(self_or_input):
            real = original_torch_unsqueeze(self_or_input.real, dim)
            imag = original_torch_unsqueeze(self_or_input.imag, dim)
            return torch.complex(real, imag)
        else:
            return original_torch_unsqueeze(self_or_input, dim)

    def _rms_norm_forward(self, x):
        if not self.elementwise_affine:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return original_rms_norm_forward(self, x)

    def _scaled_dot_product_attention(query, key, *args, enable_gqa: bool = False, **kwargs):
        # When enable_gqa=True but q_num_heads == kv_num_heads, it is standard MHA, not GQA.
        # The upstream ONNX SDPA function incorrectly asserts q_num_heads > kv_num_heads when
        # enable_gqa=True, which fails for MHA models. Treat equal heads as MHA (enable_gqa=False).
        if enable_gqa and query.shape[1] == key.shape[1]:
            enable_gqa = False

        # ONNX only supports 4D SDPA [B, H, S, D]. For 5D blocked attention [B, G, H, S, D]
        # (e.g. GraniteSpeech local attention with num_blocks dim), flatten G into B, then unflatten.
        if query.dim() == 5:
            B, G = query.shape[0], query.shape[1]
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = args[0].flatten(0, 1)
            args = (value,) + args[1:]
            if kwargs.get("attn_mask") is not None and kwargs["attn_mask"].dim() == 5:
                kwargs["attn_mask"] = kwargs["attn_mask"].flatten(0, 1)
            out = original_scaled_dot_product_attention(query, key, *args, enable_gqa=enable_gqa, **kwargs)
            return out.unflatten(0, (B, G))

        return original_scaled_dot_product_attention(query, key, *args, enable_gqa=enable_gqa, **kwargs)

    def _randperm(n, *, dtype=torch.int64, layout=torch.strided, device=None, pin_memory=False, generator=None):
        # aten.randperm has no ONNX translation. Replace with argsort(rand(n)) which uses
        # only ONNX-supported ops (RandomUniform + TopK/Sort).
        return torch.argsort(torch.rand(n, device=device)).to(dtype)

    def _cummax_or_cummin(input, dim, *, mode):
        # aten.cummax / aten.cummin have no ONNX translations. Decompose via upper-triangular
        # masked reduction: for each output position i, include only input positions j <= i.
        # This uses only ONNX-supported ops (expand, where, max/min) at O(n²) memory cost.
        n = input.shape[dim]
        x = input.movedim(dim, -1)  # (..., n)
        x_grid = x.unsqueeze(-2).expand(*x.shape[:-1], n, n)  # (..., n, n)
        # Lower-triangular mask: include[i, j] = True when j <= i
        include = torch.ones(n, n, dtype=torch.bool, device=input.device).tril()
        # Fill excluded positions with the identity element for the reduction
        if input.dtype == torch.bool:
            fill_val = mode != "max"
        elif input.is_floating_point():
            fill_val = torch.finfo(input.dtype).min if mode == "max" else torch.finfo(input.dtype).max
        else:
            fill_val = torch.iinfo(input.dtype).min if mode == "max" else torch.iinfo(input.dtype).max
        fill = torch.full((), fill_val, dtype=input.dtype, device=input.device)
        masked = torch.where(include, x_grid, fill)
        out = masked.max(dim=-1) if mode == "max" else masked.min(dim=-1)
        return out.values.movedim(-1, dim), out.indices.movedim(-1, dim)

    def _bucketize(input, boundaries, *, out_int32=False, right=False):
        # Vectorized replacement for torch.bucketize. The original decomposes via
        # torch.where(cond, mid, 0) in step 2, creating scalar-constant tensors
        # (_tensor_constant0/1) that functionalize() wraps with alias+detach_,
        # causing assert_functional_graph to fail.
        # This implementation avoids any constant tensors.
        # Edge case: empty boundaries → all inputs fall in bucket 0.
        # Without this guard, mask has shape (*input.shape, 0) and mask.sum(-1)
        # translates to ReduceSum over a 0-dim tensor, which ORT CUDA rejects.
        if boundaries.numel() == 0:
            result = torch.zeros_like(input, dtype=torch.int64)
            return result.to(torch.int32) if out_int32 else result
        if right:
            mask = boundaries <= input.unsqueeze(-1)
        else:
            mask = boundaries < input.unsqueeze(-1)
        result = mask.sum(-1)
        return result.to(torch.int32) if out_int32 else result

    def _masked_scatter(self, mask, source):
        # aten.masked_scatter has no ONNX op, and its flattened-source layout causes the same
        # Where shape mismatch as boolean-mask index_put. Replace with cumsum-gather-where:
        # broadcast mask to self's shape, use a flat cumsum to map each True position to its
        # sequential index in source, gather unconditionally, then select with where.
        mask = mask.expand_as(self)
        flat_mask = mask.reshape(-1)
        positions = (flat_mask.to(torch.int64).cumsum(0) - 1).clamp(min=0)
        gathered = source.reshape(-1)[positions]
        return torch.where(flat_mask, gathered, self.reshape(-1)).reshape(self.shape)

    def _cummax(input, dim):
        return _cummax_or_cummin(input, dim, mode="max")

    def _cummin(input, dim):
        return _cummax_or_cummin(input, dim, mode="min")

    def _broadcast_mask_expansion(mask_function):
        # Replace torch.vmap-based mask expansion with broadcast-based expansion.
        # torch.vmap higher-order ops in the step-1 FX graph cause AOT autograd in step-2
        # to fail: vmapped tensors (_add_batch_dim_*) appear as direct operands in le/ge/sub
        # nodes, triggering "tensor escaped from vmapped function".
        # The broadcast-based path (_non_vmap_expansion_sdpa) is semantically equivalent for
        # all index-based mask functions and works correctly under AOT autograd re-tracing.
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            result = mask_function(
                *_masking_utils_mod._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            )
            return result.expand(batch_arange.shape[0], 1, q_arange.shape[0], kv_arange.shape[0])

        return _expanded

    @wraps(original_forward)
    def _forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        # Clone duplicate tensors so ONNX optimizer doesn't deduplicate/rename outputs
        deduped = dedup_output_tensors(outputs)
        # Flatten nested output structures and convert non-tensor leaves to tensors, since ONNX export only tracks tensor outputs.
        return get_leaf_tensors(deduped)

    # Patch model.forward
    model.forward = _forward

    # Patch torch
    torch.cummax = _cummax
    torch.cummin = _cummin
    torch.randperm = _randperm
    torch.bucketize = _bucketize
    torch.where = _torch_where
    torch.unsqueeze = _unsqueeze
    torch.Tensor.cummax = _cummax
    torch.Tensor.cummin = _cummin
    torch.Tensor.where = _tensor_where
    torch.Tensor.unsqueeze = _unsqueeze
    torch.Tensor.masked_scatter = _masked_scatter
    torch.nn.RMSNorm.forward = _rms_norm_forward
    torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention

    # Patch masking_utils
    _masking_utils_mod._vmap_expansion_sdpa = _broadcast_mask_expansion

    try:
        yield
    finally:
        model.forward = original_forward

        torch.cummax = original_cummax
        torch.cummin = original_cummin
        torch.randperm = original_randperm
        torch.bucketize = original_bucketize
        torch.where = original_torch_where
        torch.Tensor.where = original_tensor_where
        torch.unsqueeze = original_torch_unsqueeze
        torch.Tensor.cummax = original_tensor_cummax
        torch.Tensor.cummin = original_tensor_cummin
        torch.Tensor.unsqueeze = original_tensor_unsqueeze
        torch.Tensor.masked_scatter = original_masked_scatter
        torch.nn.RMSNorm.forward = original_rms_norm_forward
        torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention

        _masking_utils_mod._vmap_expansion_sdpa = original_vmap_expansion_sdpa
