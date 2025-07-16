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
        from triton_kernels.matmul_ogs import InFlexData, FlexCtx, PrecisionConfig, MicroscalingCtx

        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, SwizzlingType

        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        # print("#######################expert_dim: ", self.expert_dim)
        # print("#######################num_experts: ", self.num_experts)
        # print(config)
        # dtype torch.float4_e2m1fn not supported yet
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

        self.gate_up_proj_right_pad = 0#smallest_even_divide_number(self.intermediate_size * 2, 256) - self.intermediate_size * 2
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
            # if torch.isnan(hidden_states).any():
            #           raise ValueError("NaNs detected in hidden_states.")

            intermediate_cache1 = matmul_ogs(hidden_states,
                                            self.gate_up_proj,
                                            self.gate_up_proj_bias.to(torch.float32),
                                            routing_data,
                                            gather_indx=gather_idx,
                                            precision_config=self.gate_up_proj_precision_config,
                                            gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
                                            fused_activation=act)

            torch.cuda.synchronize()

            if torch.isnan(intermediate_cache1).any():
                print(f"hidden_states.shape: {hidden_states.shape}, self.gate_up_proj.shape: {self.gate_up_proj.shape}")
                print(f"precision_config: {self.gate_up_proj_precision_config}")
                print(f"hidden_states.dtype: {hidden_states.dtype}, self.gate_up_proj.dtype: {self.gate_up_proj.dtype}, self.gate_up_proj_bias.dtype: {self.gate_up_proj_bias.dtype}")
          
                raise ValueError("NaNs detected in intermediate_cache1 after matmul_ogs (gate_up_proj).")

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
            if torch.isnan(intermediate_cache3).any():
                print(f"routing_data: {routing_data}")
                print(f"scatter_idx: {scatter_idx}")
                print(f"intermediate_cache3.shape: {intermediate_cache3.shape}")
                print(f"self.down_proj.shape: {self.down_proj.shape}")
                print(f"self.down_proj_bias.shape: {self.down_proj_bias.shape}")
                print(f"self.down_proj_precision_config: {self.down_proj_precision_config}")
                raise ValueError("NaNs detected in intermediate_cache3 after matmul_ogs (down_proj).")
                
            # manually crop the tensor since oai kernel pad the output
            output_states = intermediate_cache3[..., :self.hidden_size].contiguous()
        torch.cuda.synchronize()
        return output_states

def mlp_forward(self, hidden_states):
    from triton_kernels.routing import routing
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
    router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)
    routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k, sm_first=False)
    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
    return routed_out, router_logits

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
        if module.__class__.__name__ == "OpenAIMoeExperts":
            with init_empty_weights():
                # tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                model._modules[name] = Mxfp4OpenAIMoeExperts(config)
                has_been_replaced=True
        # if module.__class__.__name__ == "OpenAIMoeMLP":
        #     from types import MethodType
        #     module.forward = MethodType(mlp_forward, module)
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
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
