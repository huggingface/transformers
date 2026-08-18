# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from types import MethodType
from typing import TYPE_CHECKING, Any

from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if TYPE_CHECKING:
    from ..quantizers.quantizer_mxfp4 import Mxfp4Config, Mxfp4HfQuantizer

from ..core_model_loading import ConversionOps, _IdentityOp
from ..distributed.utils import _is_torch_distributed_initialized
from ..quantizers.quantizers_utils import get_module_from_name, on_device, should_convert_module


logger = logging.get_logger(__name__)

triton_kernels_hub = None

FP4_VALUES = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


class Mxfp4Quantize(ConversionOps):
    """Quantizes one dense expert projection to mxfp4 at load time and attaches it in a kernel-ready format.

    This runs after the model's own merge operations, so `weight` is the fused projection in the module's layout:
    either `(num_experts, in_dim, out_dim)` (gpt-oss) or `(num_experts, out_dim, in_dim)` (all other models). The triton
    kernels reads the weights as `(num_experts, in_dim, out_dim)`, so we align with that always.
    """

    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        missing_keys: set[str] | None = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        module, proj = get_module_from_name(model, full_layer_name)
        hub = get_triton_kernels_hub()
        # Exctract the weight tensor
        _, weight = tuple(input_dict.items())[0]
        weight = weight[0] if isinstance(weight, list) else weight
        # Quantize the weight
        with torch.device(weight.device):
            weight = weight if getattr(module, "is_transposed", False) else weight.transpose(-1, -2)
            triton_weight_tensor, weight_scale = quantize_to_mxfp4(weight, hub)
            triton_weight_tensor, weight_scale = swizzle_mxfp4(triton_weight_tensor, weight_scale, hub)
            _register_packed_proj(module, proj, triton_weight_tensor, weight_scale)
            # Replace the module's weight with the quantized one
            if missing_keys is not None:
                missing_keys.discard(f"{full_layer_name}")
            module._is_hf_initialized = True
        return {}


class Mxfp4Dequantize(ConversionOps):
    """Expand one checkpoint `(blocks, scales)` expert projection back to a dense parameter.

    The dequantized weight comes out as `(num_experts, in_dim, out_dim)` (gpt-oss) or `(num_experts, out_dim, in_dim)`
    (all other models).
    """

    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        module, proj = get_module_from_name(model, full_layer_name)
        blocks, scales = (
            x[0] if isinstance(x, list) else x for x in (input_dict[f"{proj}_blocks"], input_dict[f"{proj}_scales"])
        )
        dequantized = convert_moe_packed_tensors(blocks, scales)
        if not getattr(module, "is_transposed", False):
            dequantized = dequantized.transpose(-1, -2).contiguous()
        return {full_layer_name: torch.nn.Parameter(dequantized)}

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


class Mxfp4Deserialize(ConversionOps):
    """Load one checkpoint `(blocks, scales)` expert projection, swizzled for the kernels in place."""

    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Eagerly set tensors on the module and perform swizzle
        module, proj = get_module_from_name(model, full_layer_name)
        blocks, scales = (
            x[0] if isinstance(x, list) else x for x in (input_dict[f"{proj}_blocks"], input_dict[f"{proj}_scales"])
        )
        swizzle_mxfp4_convertops(blocks, scales, module, proj, blocks.device, get_triton_kernels_hub())
        if missing_keys is not None:
            missing_keys.discard(f"{full_layer_name}")
        module._is_hf_initialized = True
        # We return an empty mapping since the module was updated in-place. This prevents
        # the loader from trying to materialize the original meta-parameter names again.
        # We don't use set_param_for_module since it expects mainly a torch.nn.Parameter or a safetensors pointer
        return {}

    @property
    def reverse_op(self) -> ConversionOps:
        return Mxfp4ReverseDeserialize(self.hf_quantizer)


def unswizzle_mxfp4_proj(module: "nn.Module", proj: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Un-swizzles the `(blocks, scales)` to match the serialized layout of those parameters. Returns the blocks as a
    uint8 tensor with shape [num_experts, out_dim, in_dim // 32, 16] and scales as a float32 tensor with shape
    [num_experts, out_dim, in_dim // 32].
    """
    weight, precision_config = make_packed_mxfp4_proj(module, proj)
    num_experts, in_dim, out_dim = weight.shape

    blocks = weight.storage.layout.unswizzle_data(weight.storage.data)
    blocks = blocks[..., : in_dim // 2, :out_dim].transpose(-1, -2) 
    scales = precision_config.weight_scale.storage.layout.unswizzle_data(precision_config.weight_scale.storage.data)
    scales = scales[..., : in_dim // 32, :out_dim].transpose(-1, -2)
    return blocks.reshape(num_experts, out_dim, in_dim // 32, 16).contiguous(), scales.contiguous()


class LoadPackedMxfp4Experts(ConversionOps):
    """Load per-expert `weight_blocks` / `weight_scales` MoE projections without dequantizing them.

    Saving a generic mxfp4-quantized MoE runs the model's reverse conversions on the packed tensors, so the checkpoint
    holds one `(blocks, scales)` pair per expert projection (the same per-expert layout compressed-tensors checkpoints
    use). Loading replays the model's own expert-merging operations on the blocks and the scales separately, and
    attaches the merged stacks to the experts module swizzled for the kernels.
    """

    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer", operations: list[ConversionOps]):
        self.hf_quantizer = hf_quantizer
        self.operations = operations  # the model's conversion ops which will be used on blocks and scales separately

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        model=None,
        full_layer_name: str | None = None,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        merged = {}
        for component in ("weight_blocks", "weight_scales"):
            component_sources = [p for p in source_patterns if p.rstrip("$").endswith(component)]
            component_dict = {p: input_dict[p] for p in component_sources if p in input_dict}
            if not component_dict:
                continue
            for operation in self.operations:
                component_dict = operation.convert(
                    component_dict,
                    source_patterns=component_sources,
                    target_patterns=target_patterns,
                    full_layer_name=full_layer_name,
                    model=model,
                    **kwargs,
                )
            # there should be only one item left in the dictionary
            merged[component] = component_dict.popitem()[1]

        module, proj = get_module_from_name(model, full_layer_name)
        blocks, scales = merged["weight_blocks"], merged["weight_scales"]
        swizzle_mxfp4_convertops(blocks, scales, module, proj, blocks.device, get_triton_kernels_hub())

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)
        module._is_hf_initialized = True
        return {}

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


class DequantizeMxfp4Experts(ConversionOps):
    """Expand per-expert `weight_blocks` / `weight_scales` MoE projections to dense weights at load time.

    Runs before the model's own expert-merging operations, handing them one dense
    `(num_experts, out_dim, in_dim)` stack per projection, as if the checkpoint had never been quantized.
    """

    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        processed_out = {}
        for key, blocks in input_dict.items():
            if not key.rstrip("$").endswith("weight_blocks"):
                continue
            scales = input_dict[key.replace("weight_blocks", "weight_scales")]
            blocks = torch.stack(blocks) if isinstance(blocks, list) else blocks
            scales = torch.stack(scales) if isinstance(scales, list) else scales
            # `(num_experts, in_dim, out_dim)` comes out; the merge expects the dense module layout.
            processed_out[key] = convert_moe_packed_tensors(blocks, scales).transpose(-1, -2).contiguous()
        return processed_out

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


class Mxfp4ReverseDeserialize(ConversionOps):
    def __init__(self, hf_quantizer: "Mxfp4HfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        full_layer_name: str,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        name = full_layer_name.rsplit("_", 1)[0]
        module, _ = get_module_from_name(model, full_layer_name)
        if "bias" in full_layer_name:
            bias_name = full_layer_name.replace("_blocks", "")
            proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
            return {bias_name: getattr(module, proj + "_bias")}

        proj = name.rsplit(".", 1)[-1]
        if make_packed_mxfp4_proj(module, proj)[1] is None:
            return {}
        blocks, scales = unswizzle_mxfp4_proj(module, proj)
        return {f"{name}_blocks": blocks, f"{name}_scales": scales}


# Copied from GPT_OSS repo and vllm
def quantize_to_mxfp4(w, triton_kernels_hub):
    downcast_to_mxfp_torch = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp_torch
    w, w_scale = downcast_to_mxfp_torch(w.to(torch.bfloat16), torch.uint8, axis=1)
    return w, w_scale


def get_triton_kernels_hub() -> Any:
    """Fetch the hub package holding the mxfp4 triton kernels, caching it in the module global."""
    global triton_kernels_hub
    if triton_kernels_hub is None:
        from .hub_kernels import get_kernel

        triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
    return triton_kernels_hub


def _register_packed_proj(module: "nn.Module", proj: str, weight: Any, weight_scale: Any) -> None:
    """Registers the projection swizzled weight and scale as plain parameters. The mxfp4 kernel cannot use them directly
    so at forward time, they will be wrapped in a special type, which will share the same storage as the parameter. The
    layout objects and logical shapes are also stored in the module, to help constructing the triton wrapper."""
    module._parameters.pop(proj, None)
    module.register_parameter(f"{proj}_swizzled", torch.nn.Parameter(weight.storage.data, requires_grad=False))
    module.register_parameter(
        f"{proj}_swizzled_scales", torch.nn.Parameter(weight_scale.storage.data, requires_grad=False)
    )
    metadata = (weight.storage.layout, tuple(weight.shape), weight_scale.storage.layout, tuple(weight_scale.shape))
    setattr(module, f"{proj}_kernel_meta", metadata)


def make_packed_mxfp4_proj(module: "nn.Module", proj: str) -> tuple[Any, Any]:
    """Returns the pair of parameters `(weight, precision_config)` used by the mxfp4 kernels. These are not regular
    tensors, but rather wrappers around the parameters' storage. These are rebuilt on every call to avoid keeping stale
    references around, for a minor cost (~6us).
    """
    # There can be no metadata if this is not called on an expert module or before the metadata is set
    meta = module.__dict__.get(f"{proj}_kernel_meta")  # avoids the nn.Module.getattr looking in parameters and buffers
    if meta is None:
        weight = getattr(module, proj, None)
        return weight, getattr(module, f"{proj}_precision_config", None)

    # If the metadata is present, we can use it to rebuild the weight and scale triton tensors
    hub = get_triton_kernels_hub()
    PrecisionConfig, FlexCtx, InFlexData = (
        hub.matmul_ogs.PrecisionConfig,
        hub.matmul_ogs.FlexCtx,
        hub.matmul_ogs.InFlexData,
    )
    weight_layout, weight_shape, scale_layout, scale_shape = meta
    weight = hub.tensor.Tensor(
        hub.tensor.Storage(getattr(module, f"{proj}_swizzled"), weight_layout),
        dtype=hub.tensor.FP4,
        shape=list(weight_shape),
    )
    scale = hub.tensor.Tensor(
        hub.tensor.Storage(getattr(module, f"{proj}_swizzled_scales"), scale_layout), shape=list(scale_shape)
    )
    return weight, PrecisionConfig(weight_scale=scale, flex_ctx=FlexCtx(rhs_data=InFlexData()))


def swizzle_mxfp4(w, w_scale, triton_kernels_hub):
    """
    Changes the layout of the tensors depending on the hardware
    """
    FP4, convert_layout, wrap_torch_tensor = (
        triton_kernels_hub.tensor.FP4,
        triton_kernels_hub.tensor.convert_layout,
        triton_kernels_hub.tensor.wrap_torch_tensor,
    )
    layout = triton_kernels_hub.tensor_details.layout
    StridedLayout = triton_kernels_hub.tensor_details.layout.StridedLayout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    if triton_kernels_hub.target_info.is_cuda() and triton_kernels_hub.target_info.cuda_capability_geq(10):
        # We turn off the persistent path because it mistake the strided scales we prepared for fp4 data. If we wanted
        # to use that path, we would need to prepare scales differently. Since the path is buggy, this is ok.
        triton_kernels_hub.matmul_ogs_details.opt_flags.update_opt_flags_constraints({"is_persistent": False})
    return w, w_scale


# Mostly copied from GPT_OSS repo: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
def _convert_moe_packed_tensors(
    blocks: Any,
    scales: Any,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    import math

    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        sub = out[r0:r1]

        # With device_map="auto", tensors sitting on a non-current accelerator device are not
        # ordered after their async H2D copy, so the compute below may read garbage and emit
        # out-of-bounds `lut` indices (illegal memory access on CUDA, indexing abort on XPU).
        # Aligning the active device with the tensor's device orders it correctly (no-op on CPU).
        with on_device(blk.device):
            # This vector is only used to index into `lut`, but is huge in GPU memory so we delete it immediately
            idx_lo = (blk & 0x0F).to(torch.int)
            sub[:, 0::2] = lut[idx_lo]
            del idx_lo

            # This vector is only used to index into `lut`, but is huge in GPU memory so we delete it immediately
            idx_hi = (blk >> 4).to(torch.int)
            sub[:, 1::2] = lut[idx_hi]
            del idx_hi

            # Perform op
            torch.ldexp(sub, exp, out=sub)
        del blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    return out.transpose(1, 2).contiguous()


def convert_moe_packed_tensors(
    blocks: Any,
    scales: Any,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # limits the peak memory during dequantization to ~5 GiB
) -> torch.Tensor:
    """Dequantize the mxfp4 experts weights. Tries to do it on the same device as the blocks and scales at first, and if
    it OOMs, re-tries on the CPU."""
    try:
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)
    # In the case of OOM, dequantize on the CPU and return: accelerate will take care of the dispatch once VRAM is freed
    except torch.OutOfMemoryError:
        blocks = blocks.to("cpu")
        scales = scales.to("cpu")
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)


class Mxfp4GptOssExperts(nn.Module):
    # Weights are stored `(num_experts, in_dim, out_dim)`, like the `GptOssExperts` this module replaces.
    is_transposed = True

    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )

        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
        )

        self.down_proj = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
            requires_grad=False,
        )

        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
        )
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None
        self.limit = getattr(config, "swiglu_limit", 7.0)

    def forward(
        self, hidden_states: torch.Tensor, routing_data: Any, gather_idx: Any, scatter_idx: Any
    ) -> torch.Tensor:
        hub = get_triton_kernels_hub()
        FnSpecs, FusedActivation, matmul_ogs = (
            hub.matmul_ogs.FnSpecs,
            hub.matmul_ogs.FusedActivation,
            hub.matmul_ogs.matmul_ogs,
        )
        swiglu_fn = hub.swiglu.swiglu_fn

        gate_up_proj, gate_up_precision_config = make_packed_mxfp4_proj(self, "gate_up_proj")
        down_proj, down_precision_config = make_packed_mxfp4_proj(self, "down_proj")

        with on_device(hidden_states.device):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)

            intermediate_cache1 = matmul_ogs(
                hidden_states,
                gate_up_proj,
                self.gate_up_proj_bias.to(torch.float32),
                routing_data,
                gather_indx=gather_idx,
                precision_config=gate_up_precision_config,
                gammas=None,
                fused_activation=act,
            )

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                down_proj,
                self.down_proj_bias.to(torch.float32),
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=down_precision_config,
                gammas=routing_data.gate_scal,
            )
        return intermediate_cache3


# Adapted from GPT_OSS repo
# TODO: Add absolute link when the repo is public
def routing_torch_dist(
    logits,
    n_expts_act,
):
    import os

    GatherIndx, RoutingData, ScatterIndx, compute_expt_data_torch = (
        triton_kernels_hub.routing.GatherIndx,
        triton_kernels_hub.routing.RoutingData,
        triton_kernels_hub.routing.ScatterIndx,
        triton_kernels_hub.routing.compute_expt_data_torch,
    )

    with on_device(logits.device):
        world_size = torch.distributed.get_world_size()
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        replace_value = -1

        n_tokens = logits.shape[0]
        n_expts_tot = logits.shape[1]

        n_local_experts = n_expts_tot // world_size
        local_expert_start = rank * n_local_experts
        local_expert_end = (rank + 1) * n_local_experts

        n_gates_pad = n_tokens * n_expts_act

        def topk(vals, k):
            tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
            tk_indx = tk_indx.long()
            tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
            return tk_val, tk_indx.int()

        expt_scal, expt_indx = topk(logits, n_expts_act)
        expt_scal = torch.softmax(expt_scal, dim=-1)
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)

        # Flatten and mask for local experts
        expt_scal = expt_scal.reshape(-1)

        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)[local_expert_start:local_expert_end]

        expt_indx = expt_indx.view(-1).to(torch.int32)

        # we use a large value to replace the indices that are not in the local expert range
        var = 1000
        expt_indx = torch.where(expt_indx < local_expert_start, var, expt_indx)
        topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        expt_indx = torch.where(expt_indx < local_expert_end, expt_indx, replace_value)
        expt_indx = torch.where(local_expert_start <= expt_indx, expt_indx, replace_value)

        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_scal = expt_scal[topk_indx]

        topk_indx = torch.where(gate_indx[topk_indx] == replace_value, replace_value, topk_indx)

        # # Routing metadata for local expert computation
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())

        expt_data = compute_expt_data_torch(hist, n_local_experts, n_gates_pad)

        hit_experts = n_expts_act
    return RoutingData(gate_scal, hist, n_local_experts, hit_experts, expt_data), gather_indx, scatter_indx


def mlp_forward(self, hidden_states):
    if _is_torch_distributed_initialized() and hasattr(self, "_is_hooked"):
        routing = routing_torch_dist
    else:
        routing = triton_kernels_hub.routing.routing

    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
    router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)

    with on_device(router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)

    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx=scatter_idx)
    routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)
    return routed_out, router_logits


def swizzle_mxfp4_convertops(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    module: "nn.Module",
    proj: str,
    target_device: "torch.device | str",
    triton_kernels_hub: Any,
) -> None:
    """Swizzle and attach the `(blocks, scales)` packed expert projections to the module. Args:
        - blocks (torch.Tensor): weights packed in blocks, shaped as [local_experts, out_dim, in_dim // 32, 16] in uint8
        with two e2m1 values per byte, 16 bytes per group of 32
        - scales (torch.Tensor): the e8m0 exponents [local_experts, out_dim, in_dim // 32] for each block
        - module (nn.Module): the module to attach the packed expert projections to
        - proj (str): the name of the projection
        - target_device (torch.device | str): the device to swizzle on
        - triton_kernels_hub (Any): hub kernel object
    """
    # If the target device is a CPU and an accelerator is present, use the accelerator instead
    is_cpu = getattr(target_device, "type", target_device) == "cpu"
    if is_cpu and hasattr(torch, "accelerator"):
        accelerator = torch.accelerator.current_accelerator()
        target_device = accelerator.type if accelerator is not None else target_device
    # Ensure device, layout and shape are correct
    local_experts, out_dim = blocks.shape[:2]
    blocks = blocks.to(target_device).contiguous().reshape(local_experts, out_dim, -1)
    scales = scales.to(target_device).contiguous()
    in_dim = blocks.shape[-1] * 2
    # Actual swizzling operation
    with on_device(target_device):
        blocks = blocks.transpose(-1, -2)
        scales = scales.transpose(-1, -2)
        triton_weight_tensor, weight_scale = swizzle_mxfp4(blocks, scales, triton_kernels_hub)
    # The swizzled storage is an opaque triton tensor, so we set its shape explicitly
    triton_weight_tensor.shape = torch.Size([local_experts, in_dim, out_dim])
    _register_packed_proj(module, proj, triton_weight_tensor, weight_scale)


def attach_packed_mxfp4_proj(module: "nn.Module", proj: str, packed: torch.Tensor, scales: torch.Tensor) -> None:
    """Attach one mxfp4-packed expert projection to `module` in the layout the triton kernels expect.

    `packed` is a stack of `weight_packed` tensors, `(num_experts, out_dim, in_dim // 2)` uint8, and `scales` the
    matching e8m0 stack, `(num_experts, out_dim, in_dim // 32)`. Both keep the two-e2m1-values-per-byte encoding they
    have on disk; only their layout changes. The kernels read the weights column-major as
    `(num_experts, in_dim, out_dim)`, hence the transpose before the swizzle.
    """
    hub = get_triton_kernels_hub()

    device = packed.device
    if device.type == "cpu" and hasattr(torch, "accelerator") and torch.accelerator.current_accelerator() is not None:
        device = torch.device(torch.accelerator.current_accelerator().type)
    packed = packed.to(device).contiguous()
    scales = scales.to(device).contiguous()

    num_experts, out_dim, packed_in_dim = packed.shape
    with on_device(device):
        weight, weight_scale = swizzle_mxfp4(packed.transpose(-2, -1), scales.transpose(-2, -1), hub)
    # The swizzled storage is opaque, so the logical shape the kernels index with is set explicitly.
    weight.shape = torch.Size([num_experts, packed_in_dim * 2, out_dim])
    _register_packed_proj(module, proj, weight, weight_scale)


def _routing_data_from_top_k(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
) -> tuple[Any, Any, Any]:
    """Build the triton-kernels routing structures from a router's already-computed top-k selection.

    `triton_kernels.routing.routing` derives the selection from raw logits with its own top-k and softmax,
    which would discard whatever the model's router does (sigmoid scoring, normalisation, correction bias).
    This sorts the `(token, expert)` pairs by expert id so each expert's rows are contiguous for the matmul,
    and keeps the routing weights the model computed.
    """
    routing = get_triton_kernels_hub().routing

    num_gates = top_k_index.numel()
    expert_index = top_k_index.reshape(-1).to(torch.int32)
    gate_scale = top_k_weights.reshape(-1)

    sorted_index = torch.argsort(expert_index, stable=True).to(torch.int32)
    gate_index = torch.argsort(sorted_index, stable=True).to(torch.int32)
    # `torch.histc` takes integers on CUDA but only floats on CPU/MPS.
    histc_input = expert_index.float() if expert_index.device.type in ("cpu", "mps") else expert_index
    histogram = torch.histc(histc_input, bins=num_experts, min=0, max=num_experts - 1).int()

    routing_data = routing.RoutingData(
        gate_scale[sorted_index],
        histogram,
        num_experts,
        top_k_index.shape[-1],
        routing.compute_expt_data(histogram, num_experts, num_gates),
    )
    gather_index = routing.GatherIndx(src_indx=sorted_index, dst_indx=gate_index)
    scatter_index = routing.ScatterIndx(src_indx=gate_index, dst_indx=sorted_index)
    return routing_data, gather_index, scatter_index


def mxfp4_experts_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run fused MoE experts which weights stayed packed in mxfp4 using the triton `matmul_ogs` kernels."""
    matmul_ogs = get_triton_kernels_hub().matmul_ogs.matmul_ogs

    has_gate = getattr(module, "has_gate", True)
    proj = "gate_up_proj" if has_gate else "up_proj"
    weight, precision_config = make_packed_mxfp4_proj(module, proj)
    down_weight, down_precision_config = make_packed_mxfp4_proj(module, "down_proj")
    if precision_config is None:
        raise ValueError(
            f"`experts_implementation='mxfp4'` needs mxfp4-packed expert weights, which {type(module).__name__} "
            "does not hold. It is selected automatically when loading an mxfp4-packed checkpoint and cannot be "
            "requested for a model that was not loaded from one."
        )

    routing_data, gather_index, scatter_index = _routing_data_from_top_k(
        top_k_index, top_k_weights, module.num_experts
    )
    proj_bias = getattr(module, f"{proj}_bias", None)
    down_bias = getattr(module, "down_proj_bias", None)

    with on_device(hidden_states.device):
        proj_out = matmul_ogs(
            hidden_states,
            weight,
            None if proj_bias is None else proj_bias.to(torch.float32),
            routing_data,
            gather_indx=gather_index,
            precision_config=precision_config,
        )
        proj_out = module._apply_gate(proj_out) if has_gate else module.act_fn(proj_out)
        out = matmul_ogs(
            proj_out,
            down_weight,
            None if down_bias is None else down_bias.to(torch.float32),
            routing_data,
            scatter_indx=scatter_index,
            precision_config=down_precision_config,
            gammas=routing_data.gate_scal,
        )
    return out.to(hidden_states.dtype)


def replace_with_mxfp4_linear(
    model: torch.nn.Module,
    quantization_config: "Mxfp4Config",
    modules_to_not_convert: list[str] | None = None,
):
    """
    Public method that replaces gpt-oss expert layers of the given model with mxfp4 quantized layers. Other models keep
    their own experts module and are routed through `ExpertsInterface` by the quantizer instead.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`Mxfp4Config`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
    """
    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert

    # Exit early if the model is dequantized: no need to convert the modules
    if quantization_config.dequantize:
        return model

    # Ensures the global `triton_kernels_hub` is initialized
    get_triton_kernels_hub()

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        if module.__class__.__name__ == "GptOssExperts":
            with torch.device("meta"):
                model.set_submodule(module_name, Mxfp4GptOssExperts(model.config))
        if module.__class__.__name__ == "GptOssMLP":
            module.forward = MethodType(mlp_forward, module)

    return model


# ----------------------------------------------- DEPRECATED FUNCTIONS ----------------------------------------------- #


def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, triton_kernels_hub, **kwargs):
    """
    This transforms the weights obtained using `convert_gpt_oss.py` to load them into `Mxfp4GptOssExperts`.
    Deprecated in 5.16.
    """
    logger.warning_once("load_and_swizzle_mxfp4 is deprecated in 5.16. It will be removed in 5.21.")
    PrecisionConfig, FlexCtx, InFlexData = (
        triton_kernels_hub.matmul_ogs.PrecisionConfig,
        triton_kernels_hub.matmul_ogs.FlexCtx,
        triton_kernels_hub.matmul_ogs.InFlexData,
    )
    from ..integrations.tensor_parallel import shard_and_distribute_module

    model = kwargs.get("model")
    empty_param = kwargs.get("empty_param")
    casting_dtype = kwargs.get("casting_dtype")
    to_contiguous = kwargs.get("to_contiguous")
    rank = kwargs.get("rank")
    device_mesh = kwargs.get("device_mesh")
    if "blocks" in param_name:
        proj = param_name.split(".")[-1].split("_blocks")[0]
    if "scales" in param_name:
        proj = param_name.split(".")[-1].split("_scales")[0]
    if device_mesh is not None:
        shard_and_distribute_module(
            model, param_value, empty_param, param_name, casting_dtype, to_contiguous, rank, device_mesh
        )
    else:
        setattr(module, param_name.rsplit(".", 1)[1], torch.nn.Parameter(param_value, requires_grad=False))
    blocks_attr = f"{proj}_blocks"
    scales_attr = f"{proj}_scales"
    blocks = getattr(module, blocks_attr)  # at this point values were loaded from ckpt
    scales = getattr(module, scales_attr)
    # Check if both blocks and scales both not on meta device
    if blocks.device.type != "meta" and scales.device.type != "meta":
        local_experts = blocks.size(0)
        if proj == "gate_up_proj":
            blocks = blocks.reshape(local_experts, module.intermediate_size * 2, -1)
        else:
            blocks = blocks.reshape(local_experts, -1, module.intermediate_size // 2)
        if (
            getattr(target_device, "type", target_device) == "cpu"
            and hasattr(torch, "accelerator")
            and torch.accelerator.current_accelerator() is not None
        ):
            target_device = torch.accelerator.current_accelerator().type
        blocks = blocks.to(target_device).contiguous()
        scales = scales.to(target_device).contiguous()
        with on_device(target_device):
            triton_weight_tensor, weight_scale = swizzle_mxfp4(
                blocks.transpose(-2, -1), scales.transpose(-2, -1), triton_kernels_hub
            )

        # need to overwrite the shapes for the kernels
        if proj == "gate_up_proj":
            triton_weight_tensor.shape = torch.Size([local_experts, module.hidden_size, module.intermediate_size * 2])
        else:
            triton_weight_tensor.shape = torch.Size([local_experts, module.intermediate_size, module.hidden_size])

        # triton_weight_tensor is what needs to be passed in oai kernels. It stores the data, the shapes and any more objects. It is like a subtensor
        setattr(module, proj, triton_weight_tensor)
        setattr(
            module,
            f"{proj}_precision_config",
            PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
        )

        # delete blocks and scales
        delattr(module, scales_attr)
        delattr(module, blocks_attr)
        del blocks


def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
    logger.warning_once("dequantize is deprecated in 5.16. It will be removed in 5.21.")
    from ..integrations.tensor_parallel import shard_and_distribute_module

    model = kwargs.get("model")
    empty_param = kwargs.get("empty_param")
    casting_dtype = kwargs.get("casting_dtype")
    to_contiguous = kwargs.get("to_contiguous")
    rank = kwargs.get("rank")
    device_mesh = kwargs.get("device_mesh")

    for proj in ["gate_up_proj", "down_proj"]:
        if proj in param_name:
            if device_mesh is not None:
                param_value = shard_and_distribute_module(
                    model,
                    param_value,
                    empty_param,
                    dq_param_name,
                    casting_dtype,
                    to_contiguous,
                    rank,
                    device_mesh,
                )
            blocks_attr = f"{proj}_blocks"
            scales_attr = f"{proj}_scales"
            setattr(module, param_name.rsplit(".", 1)[1], param_value)
            if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                setattr(module, proj, torch.nn.Parameter(dequantized.to(target_device)))
                delattr(module, blocks_attr)
                delattr(module, scales_attr)
