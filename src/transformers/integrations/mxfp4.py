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

from functools import partial

from ..utils import get_current_accelerator, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps, _IdentityOp
from ..distributed.utils import _is_torch_distributed_initialized
from ..quantizers.quantizers_utils import get_module_from_name, on_device, should_convert_module
from .deepgemm import (
    _assert_sm100_requirements,
    _coerce_sf_for_kernel,
    _combine_routed_output,
    _dispatch_routed_input,
    _pad_for_deepgemm,
    deepgemm_allowed,
    deepgemm_fp8_fp4_megamoe_experts_forward,
    load_deepgemm_kernel,
)
from .finegrained_fp8 import load_finegrained_fp8_kernel
from .hub_kernels import use_kernel_forward_from_hub
from .moe import ExpertsInterface


logger = logging.get_logger(__name__)

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


class Mxfp4Quantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        missing_keys: list[str] | None = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        module, _ = get_module_from_name(model, full_layer_name)

        with torch.device(value.device):
            if isinstance(module, Mxfp4GptOssExperts):
                triton_weight_tensor, weight_scale = quantize_to_mxfp4(value.transpose(-1, -2), triton_kernels_hub)
                PrecisionConfig, FlexCtx, InFlexData = (
                    triton_kernels_hub.matmul_ogs.PrecisionConfig,
                    triton_kernels_hub.matmul_ogs.FlexCtx,
                    triton_kernels_hub.matmul_ogs.InFlexData,
                )
                triton_weight_tensor, weight_scale = swizzle_mxfp4(
                    triton_weight_tensor, weight_scale, triton_kernels_hub
                )

                proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
                swizzled(module, proj, missing_keys, full_layer_name)
                setattr(module, proj, triton_weight_tensor)
                setattr(
                    module,
                    f"{proj}_precision_config",
                    PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
                )

                module._is_hf_initialized = True

                return {}


class Mxfp4Dequantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if not any(key.endswith("_blocks") for key in input_dict):
            # `Mxfp4Experts`: the `weight` and `weight_scale_inv` of each expert, before the model's converters merge them
            dtype = getattr(*get_module_from_name(model, full_layer_name)).dtype
            return {
                key: [dequantize_mxfp4(w, s, dtype) for w, s in zip(value, input_dict[f"{key}_scale_inv"])]
                for key, value in input_dict.items()
                if not key.endswith("_scale_inv")
            }
        param_data = {}
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
        if f"{proj}_blocks" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_blocks"], list):
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"][0]
            else:
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"]
        if f"{proj}_scales" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_scales"], list):
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"][0]
            else:
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"]

        # Here we are dequantizing the weights, to the dtype of the parameter they load into
        module, tensor_name = get_module_from_name(model, full_layer_name)
        dtype = getattr(module, tensor_name).dtype
        dequantized = dequantize_convertops(param_data[f"{proj}_blocks"], param_data[f"{proj}_scales"], dtype)
        return {full_layer_name: dequantized}

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


class Mxfp4Deserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        param_data = {}
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
        if f"{proj}_blocks" not in input_dict:  # not GPT-OSS's layout (e.g. merged DeepSeek-V4 experts): as is
            return input_dict

        if f"{proj}_blocks" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_blocks"], list):
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"][0]
            else:
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"]
        if f"{proj}_scales" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_scales"], list):
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"][0]
            else:
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"]

        # Eagerly set tensors on the module and perform swizzle
        module, _ = get_module_from_name(model, full_layer_name)
        swizzle_mxfp4_convertops(
            param_data[f"{proj}_blocks"],
            param_data[f"{proj}_scales"],
            module,
            proj,
            param_data[f"{proj}_blocks"].device,
            triton_kernels_hub,
        )
        swizzled(module, proj, missing_keys, full_layer_name)
        module._is_hf_initialized = True
        # We return an empty mapping since the module was updated in-place. This prevents
        # the loader from trying to materialize the original meta-parameter names again.
        # We don't use set_param_for_module since it expects mainly a torch.nn.Parameter or a safetensors pointer
        return {}

    @property
    def reverse_op(self) -> ConversionOps:
        return Mxfp4ReverseDeserialize(self.hf_quantizer)


class Mxfp4ReverseDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        module, _ = get_module_from_name(model, full_layer_name)
        name = full_layer_name.removesuffix("_blocks")
        proj = name.rsplit(".", 1)[-1]
        weight, scales = getattr(module, proj), getattr(module, f"{proj}_precision_config").weight_scale
        scales = scales.storage.layout.unswizzle_data(scales.storage.data).transpose(-1, -2)
        blocks = weight.storage.layout.unswizzle_data(weight.storage.data).transpose(-1, -2)
        return {f"{name}_blocks": blocks.reshape(*scales.shape, -1), f"{name}_scales": scales}


# Copied from GPT_OSS repo and vllm
def quantize_to_mxfp4(w, triton_kernels_hub):
    downcast_to_mxfp_torch = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp_torch
    w, w_scale = downcast_to_mxfp_torch(w.to(torch.bfloat16), torch.uint8, axis=1)
    return w, w_scale


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
    return w, w_scale


# Mostly copied from GPT_OSS repo
# TODO: Add absolute link when the repo is public
def _convert_moe_packed_tensors(
    blocks,
    scales,
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
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    # Since the intermediate ops require A LOT of memory, in very constrained device_map="auto" settings
    # it may OOM, hence this wrapper and move back to cpu if needed
    # torch statistics are not accurate enough to estimate if we will have enough memory due to fragmentation and
    # in-place operation on non-contiguous tensors (may sometimes require more temporary copies)
    try:
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)
    # In the case of OOM due to very tight device_map, we convert and return on cpu - it will then be put back on correct
    # device with the accelerate dispatch (doing it right away may still lead to OOM, but more memory is available later)
    except torch.OutOfMemoryError:
        blocks = blocks.to("cpu")
        scales = scales.to("cpu")
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)


class Mxfp4GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        # As GPT-OSS saves them: swizzled into `gate_up_proj` and its `gate_up_proj_precision_config` once loaded
        self.gate_up_proj_blocks = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, dtype=torch.uint8),
            requires_grad=False,
        )

        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
        )

        self.down_proj_blocks = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32), dtype=torch.uint8),
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

    def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
        FnSpecs, FusedActivation, matmul_ogs = (
            triton_kernels_hub.matmul_ogs.FnSpecs,
            triton_kernels_hub.matmul_ogs.FusedActivation,
            triton_kernels_hub.matmul_ogs.matmul_ogs,
        )
        swiglu_fn = triton_kernels_hub.swiglu.swiglu_fn

        with on_device(hidden_states.device):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)

            intermediate_cache1 = matmul_ogs(
                hidden_states,
                self.gate_up_proj,
                self.gate_up_proj_bias.to(torch.float32),
                routing_data,
                gather_indx=gather_idx,
                precision_config=self.gate_up_proj_precision_config,
                gammas=None,
                fused_activation=act,
            )

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                self.down_proj_bias.to(torch.float32),
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
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


@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class Mxfp4GptOssMLP(nn.Module):
    def __init__(self, router, experts):
        super().__init__()
        self.router = router
        self.experts = experts

    def forward(self, hidden_states):
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


# The finegrained-fp8 kernels' block size for MXFP4 weights: a scale per 32 values of a row
MXFP4_BLOCK_SIZE = (1, 32)


class Mxfp4Experts(nn.Module):
    """
    MXFP4 experts (e.g. DeepSeek-V4's): e2m1 pairs packed in `int8`, a ue8m0 scale per 32 values. The default forward
    dequantizes each routed expert; the `ALL_FP4_EXPERTS_FUNCTIONS` run them W4A8 on the GPU.
    """

    # `_experts_implementation` -> {default TP/EP plan style: impl-specific style}, applied by the quantizer
    _impl_tp_layer_overrides: dict[str, dict[str, str]] = {
        "deepgemm_megamoe": {
            "ep_dispatch_experts": "megamoe_experts",
            "moe_tp_experts": "megamoe_experts",
            "ep_router": "megamoe_router",
        },
    }

    def __init__(self, config, has_bias: bool = False, has_gate: bool = True):
        super().__init__()
        if has_bias:
            raise ValueError("Mxfp4Experts does not support bias")
        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        self.hidden_dim = config.hidden_size
        self.num_experts = getattr(config, "num_local_experts", None) or config.num_experts
        self.intermediate_dim = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = getattr(config, "swiglu_limit", None)

        up_proj = "gate_up_proj" if has_gate else "up_proj"
        up_dim = 2 * self.intermediate_dim if has_gate else self.intermediate_dim
        self._alloc_proj(up_proj, up_dim, self.hidden_dim)
        self._alloc_proj("down_proj", self.hidden_dim, self.intermediate_dim)

    def _alloc_proj(self, proj: str, out_dim: int, in_dim: int):
        "Two e2m1 values per `int8`, a ue8m0 scale per 32 of them"
        weight = torch.empty(self.num_experts, out_dim, in_dim // 2, dtype=torch.int8)
        scale = torch.empty(self.num_experts, out_dim, in_dim // 32, dtype=torch.float8_e8m0fnu)
        self.register_parameter(proj, nn.Parameter(weight, requires_grad=False))
        self.register_parameter(f"{proj}_scale_inv", nn.Parameter(scale, requires_grad=False))
        self.register_parameter(f"{proj}_bias", None)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def dequantized(self, proj: str, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
        weight, scale = getattr(self, proj)[expert_idx], getattr(self, f"{proj}_scale_inv")[expert_idx]
        return dequantize_mxfp4(weight, scale, dtype)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # Default: dequantize each routed expert to the activations' dtype, accumulate in float32
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        up_proj = "gate_up_proj" if self.has_gate else "up_proj"

        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            proj_out = nn.functional.linear(current_state, self.dequantized(up_proj, expert_idx, current_state.dtype))
            proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            proj_out = nn.functional.linear(proj_out, self.dequantized("down_proj", expert_idx, proj_out.dtype))
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted_out.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)


def _w4a8_experts(self, hidden_states, matmul):
    "up (-> gate) -> down on the routed rows, each projection through `matmul(input, weight, scale)`"
    up_proj = "gate_up_proj" if self.has_gate else "up_proj"
    proj_out = matmul(hidden_states, getattr(self, up_proj), getattr(self, f"{up_proj}_scale_inv"))
    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
    return matmul(proj_out, self.down_proj, self.down_proj_scale_inv)


def fp4_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    "W4A8 on the finegrained-fp8 Triton kernel, one row per (token, expert) pair"
    kernel = load_finegrained_fp8_kernel()
    num_tokens, num_top_k = top_k_index.shape
    expert_ids = top_k_index.reshape(-1)
    matmul = partial(kernel.batched_matmul, block_size=MXFP4_BLOCK_SIZE, expert_ids=expert_ids)
    proj_out = _w4a8_experts(self, hidden_states.repeat_interleave(num_top_k, dim=0), matmul)
    weighted_out = proj_out * top_k_weights.reshape(-1, 1).to(proj_out.dtype)
    if self._is_expert_parallel:  # the kernel leaves the rows of other ranks' experts uninitialized
        weighted_out.masked_fill_((expert_ids >= self.num_experts).unsqueeze(-1), 0.0)
    # reshape + sum: deterministic, unlike `index_add_` on CUDA
    return weighted_out.view(num_tokens, num_top_k, -1).sum(dim=1).to(hidden_states.dtype)


def fp4_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    "W4A8 on the finegrained-fp8 Triton kernel, the (token, expert) pairs grouped by expert"
    kernel = load_finegrained_fp8_kernel()
    num_tokens, num_top_k = top_k_index.shape
    expert_ids, perm = torch.sort(top_k_index.reshape(-1))
    # histc, not bincount (cuda graphs): CPU wants a float input, CUDA an int one (deterministic mode)
    histc_input = expert_ids.float() if hidden_states.device.type == "cpu" else expert_ids.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    matmul = partial(
        kernel.grouped_matmul, offsets=offsets, tokens_per_expert=tokens_per_expert, block_size=MXFP4_BLOCK_SIZE
    )
    proj_out = _w4a8_experts(self, hidden_states[perm // num_top_k], matmul)
    weighted_out = proj_out * top_k_weights.reshape(-1)[perm, None].to(proj_out.dtype)
    if self._is_expert_parallel:  # other ranks' experts sort last, past `offsets[-1]`: uninitialized
        weighted_out.masked_fill_((expert_ids >= self.num_experts).unsqueeze(-1), 0.0)
    unsorted = torch.empty_like(weighted_out)
    unsorted[perm] = weighted_out
    return unsorted.view(num_tokens, num_top_k, -1).sum(dim=1).to(hidden_states.dtype)


def deepgemm_fp4_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    "W4A8 on DeepGEMM's M-grouped FP8 x FP4 GEMM (Blackwell only)"
    if not deepgemm_allowed(hidden_states.device):
        raise RuntimeError(
            "DeepGEMM experts selected on a model spanning multiple CUDA devices in one process; "
            "its kernels are bound to a single CUDA context and corrupt across devices. Use "
            "`experts_implementation='grouped_mm'`, or run one device per process (TP/EP)."
        )
    # Before the kernel's load: FP4 has no Hopper kernel
    _assert_sm100_requirements(self.down_proj, self.down_proj_scale_inv)
    deepgemm = load_deepgemm_kernel()
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"DeepGEMM experts path requires bfloat16 hidden states, got {hidden_states.dtype}")

    num_tokens, num_top_k = top_k_index.shape
    sorted_hidden, sorted_weights, _, sentinel_mask, perm, sorted_to_padded, grouped_layout, padded_rows = (
        _dispatch_routed_input(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.num_experts,
            deepgemm.m_alignment,
            True,  # SM100's psum layout
            is_expert_parallel=self._is_expert_parallel,
        )
    )

    def matmul(input, weight, scale):
        # FP8 activations with a packed ue8m0 scale per 32, like the weights
        act, act_scale = deepgemm.per_token_cast_to_fp8(input, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        out = torch.empty(padded_rows, weight.size(1), device=input.device, dtype=torch.bfloat16)
        deepgemm.grouped_fp8_fp4_matmul_nt(
            (act, _coerce_sf_for_kernel(act_scale, True, expected_mn=padded_rows)),
            (weight, _coerce_sf_for_kernel(scale, True, expected_mn=weight.size(-2))),
            out,
            grouped_layout,
            recipe=(1, 1, 32),
            use_psum_layout=True,
        )
        return out

    out = _w4a8_experts(self, _pad_for_deepgemm(sorted_hidden, sorted_to_padded, padded_rows), matmul)
    hidden_dim = hidden_states.size(-1)
    return _combine_routed_output(
        out,
        sorted_weights,
        sentinel_mask,
        perm,
        sorted_to_padded,
        num_tokens,
        num_top_k,
        hidden_dim,
        hidden_states.dtype,
    )


class FP4ExpertsInterface(ExpertsInterface):
    """Interface for registering custom FP4 experts forward functions."""

    _global_mapping = {
        "batched_mm": fp4_batched_mm_experts_forward,
        "grouped_mm": fp4_grouped_mm_experts_forward,
        "deepgemm": deepgemm_fp4_experts_forward,
        "deepgemm_megamoe": deepgemm_fp8_fp4_megamoe_experts_forward,
    }


ALL_FP4_EXPERTS_FUNCTIONS = FP4ExpertsInterface()


def dequantize_mxfp4(weight, scale, dtype):
    "`(..., N, K // 2)` e2m1 pairs packed in `int8` (low nibble first), times their `(..., N, K // 32)` ue8m0 scales"
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=weight.device)
    weight = weight.view(torch.uint8)
    values = torch.stack([lut[(weight & 0xF).int()], lut[(weight >> 4).int()]], dim=-1).flatten(-2)
    return (values.unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1)).flatten(-2).to(dtype)


def dequantize_convertops(blocks, scales, dtype):
    dequantized = convert_moe_packed_tensors(blocks, scales, dtype=dtype)
    return torch.nn.Parameter(dequantized)


def swizzled(module, proj, missing_keys, full_layer_name):
    "The serialized params of `proj` (`gate_up_proj_blocks`, `gate_up_proj_scales`) now live in its Triton tensor"
    prefix = full_layer_name.rpartition(".")[0]
    for name in (f"{proj}_blocks", f"{proj}_scales"):
        module._parameters.pop(name, None)
        missing_keys.discard(f"{prefix}.{name}")


def swizzle_mxfp4_convertops(blocks, scales, module, proj, target_device, triton_kernels_hub):
    """
    This transforms the weights obtained using `convert_gpt_oss.py` to load them into `Mxfp4GptOssExperts`.
    """
    PrecisionConfig, FlexCtx, InFlexData = (
        triton_kernels_hub.matmul_ogs.PrecisionConfig,
        triton_kernels_hub.matmul_ogs.FlexCtx,
        triton_kernels_hub.matmul_ogs.InFlexData,
    )

    local_experts = blocks.size(0)
    if (
        getattr(target_device, "type", target_device) == "cpu"
        and (accelerator := get_current_accelerator()) is not None
    ):
        target_device = accelerator.type

    blocks = blocks.to(target_device).contiguous()
    scales = scales.to(target_device).contiguous()

    if proj == "gate_up_proj":
        blocks = blocks.reshape(local_experts, module.intermediate_size * 2, -1)
    else:
        blocks = blocks.reshape(local_experts, -1, module.intermediate_size // 2)

    with on_device(target_device):
        triton_weight_tensor, weight_scale = swizzle_mxfp4(
            blocks.transpose(-2, -1), scales.transpose(-2, -1), triton_kernels_hub
        )
    # need to overwrite the shapes for the kernels
    if proj == "gate_up_proj":
        triton_weight_tensor.shape = torch.Size([local_experts, module.hidden_size, module.intermediate_size * 2])
    else:
        triton_weight_tensor.shape = torch.Size([local_experts, module.intermediate_size, module.hidden_size])

    # triton_weight_tensor is what needs to be passed in oai kernels. It stores the data, the shapes and any more objects. It's like a subtensor
    setattr(module, proj, triton_weight_tensor)
    setattr(
        module,
        f"{proj}_precision_config",
        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
    )


def replace_with_mxfp4_linear(model, quantization_config=None, modules_to_not_convert: list[str] | None = None):
    """
    Public method that replaces the expert layers of the given model with mxfp4 quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`Mxfp4Config`, defaults to `None`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
    """
    if quantization_config.dequantize:
        return model

    from ..models.gpt_oss.modeling_gpt_oss import GptOssMLP
    from .hub_kernels import get_kernel

    global triton_kernels_hub
    triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)

    has_been_replaced = False
    for module_name, module in list(model.named_modules()):
        if isinstance(module, GptOssMLP) and should_convert_module(f"{module_name}.experts", modules_to_not_convert):
            with torch.device("meta"):
                model.set_submodule(module_name, Mxfp4GptOssMLP(module.router, Mxfp4GptOssExperts(model.config)))
            has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
