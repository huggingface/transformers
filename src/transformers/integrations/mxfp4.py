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

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging


if is_torch_available():
    import torch
    from torch import nn
from typing import Optional

from ..core_model_loading import ConversionOps


if is_accelerate_available():
    from accelerate import init_empty_weights

from contextlib import contextmanager

from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module


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


@contextmanager
def on_device(dev):
    if is_torch_available():
        import torch

        if isinstance(dev, torch.Tensor):
            dev = dev.device
        elif isinstance(dev, str):
            dev = torch.device(dev)
        dev_type = getattr(dev, "type", None)
        if dev_type == "cuda":
            with torch.cuda.device(dev):
                yield
                return
        if dev_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(dev):
                yield
                return
    # other: CPU
    yield


class Mxfp4Quantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: Optional[torch.nn.Module] = None,
        missing_keys: Optional[list[str]] = None,
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

                if proj in module._parameters:
                    # Remove the nn.Parameter registration so we can attach the Triton tensor
                    del module._parameters[proj]

                setattr(module, proj, triton_weight_tensor)
                setattr(
                    module,
                    f"{proj}_precision_config",
                    PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
                )

                missing_keys.discard(f"{full_layer_name}")
                module._is_hf_initialized = True

                return {}


class Mxfp4Dequantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: Optional[torch.nn.Module] = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        param_data = {}
        if "_blocks" in input_dict.keys():
            if isinstance(input_dict["_blocks"], list):
                param_data["_blocks"] = input_dict["_blocks"][0]
            else:
                param_data["_blocks"] = input_dict["_blocks"]
        if "_scales" in input_dict.keys():
            if isinstance(input_dict["_scales"], list):
                param_data["_scales"] = input_dict["_scales"][0]
            else:
                param_data["_scales"] = input_dict["_scales"]

        # Here we are dequantizing the weights
        dequantized = dequantize_convertops(param_data["_blocks"], param_data["_scales"], param_data["_blocks"].device)
        return {full_layer_name: dequantized}


class Mxfp4Deserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: Optional[torch.nn.Module] = None,
        full_layer_name: str | None = None,
        missing_keys: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        param_data = {}
        if "_blocks" in input_dict.keys():
            if isinstance(input_dict["_blocks"], list):
                param_data["_blocks"] = input_dict["_blocks"][0]
            else:
                param_data["_blocks"] = input_dict["_blocks"]
        if "_scales" in input_dict.keys():
            if isinstance(input_dict["_scales"], list):
                param_data["_scales"] = input_dict["_scales"][0]
            else:
                param_data["_scales"] = input_dict["_scales"]

        # Eagerly set tensors on the module and perform swizzle
        module, _ = get_module_from_name(model, full_layer_name)
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
        swizzle_mxfp4_convertops(
            param_data["_blocks"],
            param_data["_scales"],
            module,
            proj,
            param_data["_blocks"].device,
            triton_kernels_hub,
        )
        missing_keys.discard(f"{full_layer_name}")
        module._is_hf_initialized = True
        # We return an empty mapping since the module was updated in-place. This prevents
        # the loader from trying to materialize the original meta-parameter names again.
        # We don't use set_param_for_module since it expects mainly a torch.nn.Parameter or a safetensors pointer
        return {}


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


# Copied from GPT_OSS repo
# TODO: Add absolute link when the repo is public
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
    import math

    blocks = blocks.to(torch.uint8)
    # Check if blocks and scales are on CPU, and move to GPU if so
    if not blocks.is_cuda and torch.cuda.is_available():
        blocks = blocks.cuda()
        scales = scales.cuda()
    elif (blocks.device.type != "xpu") and is_torch_xpu_available():
        blocks = blocks.to("xpu")
        scales = scales.to("xpu")

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

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    del blocks, scales, lut
    return out.transpose(1, 2).contiguous()


class Mxfp4GptOssExperts(nn.Module):
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


def mlp_forward(self, hidden_states):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and hasattr(self, "_is_hooked"):
        routing = routing_torch_dist
    else:
        routing = triton_kernels_hub.routing.routing

    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
    router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)

    with on_device(router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)

    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
    routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)
    return routed_out, router_logits


def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
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
                if target_device == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif target_device == "cpu" and is_torch_xpu_available():
                    torch.xpu.empty_cache()
                setattr(module, proj, torch.nn.Parameter(dequantized.to(target_device)))
                delattr(module, blocks_attr)
                delattr(module, scales_attr)


def dequantize_convertops(blocks, scales, target_device):
    dequantized = convert_moe_packed_tensors(blocks, scales)
    if target_device == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    dequantized = torch.nn.Parameter(dequantized.to(target_device))
    return dequantized


def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, triton_kernels_hub, **kwargs):
    """
    This transforms the weights obtained using `convert_gpt_oss.py` to load them into `Mxfp4GptOssExperts`.
    """
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
        if getattr(target_device, "type", target_device) == "cpu":
            target_device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
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
    if getattr(target_device, "type", target_device) == "cpu":
        target_device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    blocks = blocks.to(target_device).contiguous()
    scales = scales.to(target_device).contiguous()

    if proj == "gate_up_proj":
        blocks = blocks.reshape(local_experts, module.intermediate_size * 2, -1)
    else:
        blocks = blocks.reshape(local_experts, -1, module.intermediate_size // 2)
    if getattr(target_device, "type", target_device) == "cpu":
        target_device = "cuda"

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
    # Since the Experts module registers gate_up_proj and down_proj as nn.Parameters, we need to remove them so we can attach the Triton tensor
    if proj in module._parameters:
        # Remove the nn.Parameter registration so we can attach the Triton tensor
        del module._parameters[proj]
    setattr(module, proj, triton_weight_tensor)
    setattr(
        module,
        f"{proj}_precision_config",
        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
    )


def replace_with_mxfp4_linear(model, modules_to_not_convert=None, quantization_config=None):
    if quantization_config.dequantize:
        return model

    from kernels import get_kernel

    global triton_kernels_hub
    triton_kernels_hub = get_kernel("kernels-community/triton_kernels")

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        if module.__class__.__name__ == "GptOssExperts" and not quantization_config.dequantize:
            with init_empty_weights():
                model.set_submodule(module_name, Mxfp4GptOssExperts(model.config))
                has_been_replaced = True
        if module.__class__.__name__ == "GptOssMLP" and not quantization_config.dequantize:
            from types import MethodType

            module.forward = MethodType(mlp_forward, module)

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
