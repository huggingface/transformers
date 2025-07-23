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

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights

import re


logger = logging.get_logger(__name__)

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def quantize_to_mxfp4(w, swizzle_mx_value, swizzle_mx_scale): 
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.matmul_ogs import InFlexData, MicroscalingCtx

    swizzle_axis = 2 if swizzle_mx_scale or swizzle_mx_value else None
    w = w.to(torch.bfloat16)

    w, mx_scales, weight_scale_shape = downcast_to_mxfp(
        w,
        torch.uint8,
        axis=1,
        swizzle_axis=swizzle_axis,
        swizzle_scale=swizzle_mx_scale,
        swizzle_value=swizzle_mx_value)
    return w, InFlexData(), MicroscalingCtx(
        weight_scale=mx_scales,
        swizzle_scale=swizzle_mx_scale,
        swizzle_value=swizzle_mx_value,
        actual_weight_scale_shape=weight_scale_shape)

def shuffle_weight(w: "torch.Tensor") -> "torch.Tensor":
    # Shuffle weight along the last dimension so that
    # we folded the weights to adjance location
    # Example:
    # input:
    #       [[1, 2, 3, 4, 5, 6],
    #        [7, 8, 9, 10, 11, 12]]
    # output:
    #       [[1, 4, 2, 5, 3, 6],
    #        [7, 10, 8, 11, 9, 12]]
    # This will be used together with triton swiglu kernel
    shape = w.shape
    N = shape[-1]
    first = w[..., :N // 2]
    second = w[..., N // 2:]

    stacked = torch.stack((first, second), dim=-1)
    w_shuffled = stacked.reshape(shape)
    return w_shuffled

def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    scales = scales.to(torch.int32) - 127

    assert  blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total   = math.prod(prefix_shape) * G

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
        del idx_lo, idx_hi, blk, exp

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    # to match for now existing implementation
    return out

# maybe subclass
class Mxfp4OpenAIMoeExperts(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, "num_experts") else config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj_blocks = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.expert_dim, self.hidden_size//32, 16, dtype=torch.uint8), requires_grad=False,
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.expert_dim, self.hidden_size//32, dtype=torch.uint8), requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(self.num_experts, 2 * self.expert_dim, dtype=torch.float32), requires_grad=False)

        self.down_proj_blocks = nn.Parameter(
            torch.zeros((self.num_experts, self.expert_dim, self.hidden_size//32, 16), dtype=torch.uint8), requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.zeros(self.num_experts, self.expert_dim, self.hidden_size//32, dtype=torch.uint8), requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(self.num_experts, self.expert_dim, dtype=torch.float32), requires_grad=False)
        self.alpha = 1.702


        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None

        smallest_even_divide_number = lambda x, n: (x // n + 1) * n if x % n != 0 else x

        self.gate_up_proj_right_pad = 0#    smallest_even_divide_number(self.intermediate_size * 2, 256) - self.intermediate_size * 2
        self.gate_up_proj_bottom_pad = 0 
        
        self.down_proj_right_pad = 0#smallest_even_divide_number(self.hidden_size, 256) - self.hidden_size
        self.down_proj_bottom_pad = 0#self.gate_up_proj_right_pad // 2
            
        self.hidden_size_pad = 0#smallest_even_divide_number(self.hidden_size, 256) - self.hidden_size
    def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
        """
        To update with moe mxfp4 kernels, for now we just upcast the weights in torch.bfloat16
        """
        # type check, uint8 means mxfp4
        #TODO: fp8 x mxfp4 on blackwell
        # print("routing_data", routing_data)
        # print("gather_idx", gather_idx)
        # print("scatter_idx", scatter_idx)
        assert hidden_states.dtype == torch.bfloat16
        assert self.gate_up_proj_blocks.dtype in (torch.bfloat16, torch.uint8)
        assert self.down_proj_blocks.dtype in (torch.bfloat16, torch.uint8)
        # assert self.gate_up_proj_bias.dtype == torch.float32, "expected float32 for gate_up_proj_bias and got " + str(self.gate_up_proj_bias.dtype)
        # assert self.down_proj_bias.dtype == torch.float32, "expected float32 for down_proj_bias and got " + str(self.down_proj_bias.dtype)

        # Shape check, only check non-mxfp4
        if self.gate_up_proj_blocks.dtype != torch.uint8:
            assert hidden_states.ndim == 2
            assert hidden_states.shape[-1] == self.gate_up_proj_blocks.shape[-2]
            assert self.down_proj_blocks.shape[-1] == self.gate_up_proj_blocks.shape[1]

        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
        from triton_kernels.swiglu import swiglu_fn
        # TODO: needed in the context of device_map, maybe not for TP
        with torch.cuda.device(hidden_states.device):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),(self.alpha, None), 2)

            if self.hidden_size_pad is not None:
                hidden_states = torch.nn.functional.pad(hidden_states,
                                    (0, self.hidden_size_pad, 0, 0),
                                    mode="constant",
                                    value=0)

            apply_router_weight_on_input = False

            intermediate_cache1 = matmul_ogs(hidden_states,
                                            self.gate_up_proj,
                                            self.gate_up_proj_bias.to(torch.float32),
                                            routing_data,
                                            gather_indx=gather_idx,
                                            precision_config=self.gate_up_proj_precision_config,
                                            gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
                                            fused_activation=act)

            torch.cuda.synchronize()

        with torch.cuda.device(hidden_states.device):    
            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                self.down_proj_bias.to(torch.float32),
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
                gammas=None if apply_router_weight_on_input else routing_data.gate_scal)
            
            torch.cuda.synchronize()
            # manually crop the tensor since oai kernel pad the output
            output_states = intermediate_cache3[..., :self.hidden_size].contiguous()
        torch.cuda.synchronize()
        return output_states

def create_expert_indices_for_rank(router_logits, top_k, ep_rank, ep_size, total_experts):
    """Create expert indices that only select experts belonging to ep_rank"""
    num_local_experts = total_experts // ep_size
    router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)

    router_scores = router_scores[:, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
    router_indices = router_indices.masked_fill((router_indices // num_local_experts) != ep_rank, 0)
    router_indices = router_indices % num_local_experts
    # Create indices that only point to local experts
    return router_indices


def mlp_forward(self, hidden_states):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        routing = routing_torch_ep_2
    else:
        from triton_kernels.routing import routing
        routing = routing
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
    router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)
    routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k, sm_first=False)
    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
    return routed_out, router_logits
    
def routing_torch_ep_2(
    logits,
    n_expts_act,
    sm_first=False
):
    import os
    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, compute_expt_data_torch

    with torch.cuda.device(logits.device):
        world_size = torch.distributed.get_world_size() 
        # world_size = 1
        rank = int(os.environ.get("LOCAL_RANK", 0))
        replace_value = -1
        
        n_tokens = logits.shape[0]
        n_expts_tot = logits.shape[1]

        n_local_experts = n_expts_tot // world_size
        # rank = 1
        local_expert_start = rank * n_local_experts
        local_expert_end = (rank + 1) * n_local_experts

        # TODO: check why +20 ? 
        n_gates_pad = n_tokens * n_expts_act

        def topk(vals, k, expt_indx):
            tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
            tk_indx = tk_indx.long()
            tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
            return tk_val, tk_indx.int()

        expt_scal, expt_indx = topk(logits, n_expts_act, None)
        expt_scal = torch.softmax(expt_scal, dim=-1)
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        # print("expt_indx and expt_scal")
        # print(expt_indx)
        # print(expt_scal)

        # Flatten and mask for local experts
        expt_scal = expt_scal.reshape(-1)
        # print("local_experts")
        # print(local_expert_start)
        # print(local_expert_end)
        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)[local_expert_start : local_expert_end]

        # for each row, count how many of its experts are local
        # num_local_expts = (expt_indx < local_expert_end) & (local_expert_start <= expt_indx)
        # num_local_expts = num_local_expts.sum(dim=1)
        
        # Count the number of rows that are for local experts, padded to an alignment.
        # n_local_rows = (num_local_expts != 0).sum()
        # n_local_rows = ((n_local_rows + row_align - 1) // row_align) * row_align

        # TODO: check if reorder really impacts or not the performances
        # is_active = torch.argsort((num_local_expts == 0).to(torch.int8), stable=True)
        # expt_indx = expt_indx[is_active]
        expt_indx = expt_indx.view(-1).to(torch.int32)
        # print("expt_indx 2 ")
        # print(expt_indx)

        # Note: Because the number of rows routed to each expert is only known at runtime,
        # we do not drop tokens that are not routed to the local expert. This ensures that
        # the tensor shapes are fixed.
        # Create topk_indx/gate_indx.
        
        # try to move values that were seen to later
        # print(expt_indx)
        expt_indx = torch.where(expt_indx < local_expert_start, 1000, expt_indx)
        # print('after')
        # print(expt_indx)
        topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
        # print("topk_indx")
        # print(topk_indx)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        # print("gate_indx")
        # print(gate_indx)
        # Now filter out all experts 
        expt_indx = torch.where(expt_indx < local_expert_end, expt_indx, replace_value)
        expt_indx = torch.where(local_expert_start <= expt_indx, expt_indx, replace_value)
        # print(expt_indx)

        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_scal = expt_scal[topk_indx]
        # print("updated expt_scal")
        # print(gate_scal)
        
        topk_indx = torch.where(gate_indx[topk_indx] == replace_value, replace_value, topk_indx)

        # print(topk_indx)

        # # Routing metadata for local expert computation
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        
        # n_gates_pad = local_expt_indx.numel()
        expt_data = compute_expt_data_torch(hist, n_local_experts, n_gates_pad)
        # print("expt_data")
        # print(expt_data)
        # hitted_experts = len(local_expt_indx) -> maybe try to get the closest power of 2 later on 
        hitted_experts = n_expts_act
    return RoutingData(gate_scal, hist, n_local_experts, hitted_experts, expt_data), gather_indx, scatter_indx

def should_convert_module(current_key_name, patterns):
    current_key_name_str = ".".join(current_key_name)
    if not any(
        re.match(f"{key}\\.", current_key_name_str) or re.match(f"{key}", current_key_name_str) for key in patterns
    ):
        return True
    return False


def _replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    config=None,
    tp_plan=None,
):
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)
        if not should_convert_module(current_key_name, modules_to_not_convert):
            current_key_name.pop(-1)
            continue
        # if isinstance(module, nn.Linear):
        #     raise NotImplementedError("Mxfp4 linear layer is not implemented yet")
        if module.__class__.__name__ == "OpenAIMoeExperts" and not quantization_config.dequantize:
            with init_empty_weights():
                # tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                _forward_pre_hooks = module._forward_pre_hooks
                _forward_hooks = module._forward_hooks
                model._modules[name] = Mxfp4OpenAIMoeExperts(config)
                has_been_replaced=True
        if module.__class__.__name__ == "OpenAIMoeMLP" and not quantization_config.dequantize:
            from types import MethodType
            module.forward = MethodType(mlp_forward, module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_mxfp4_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                config=config,
                tp_plan=tp_plan,
            )
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_mxfp4_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    config=None,
    tp_plan=None,
):

    if quantization_config.dequantize:
        return model
    
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_mxfp4_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        config=config,
        tp_plan=tp_plan,
    )
    if not has_been_replaced :
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model