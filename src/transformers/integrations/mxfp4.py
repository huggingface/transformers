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


def quantize_to_mxfp4(w, swizzle_mx_value, swizzle_mx_scale): 
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.matmul_ogs import InFlexData, MicroscalingCtx
    swizzle_axis = 2 if swizzle_mx_scale else None
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

# maybe subclass
class Mxfp4OpenAIMoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # dtype torch.float4_e2m1fn not supported yet
        self.gate_up_proj = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim, dtype=torch.float8_e5m2),
        )
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(
            torch.zeros((self.num_experts, self.expert_dim, self.hidden_size), dtype=torch.float8_e5m2),
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        
        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None

        smallest_even_divide_number = lambda x, n: (x // n + 1) * n if x % n != 0 else x

        self.gate_up_proj_right_pad = smallest_even_divide_number(self.intermediate_size * 2, 256) - self.intermediate_size * 2
        self.gate_up_proj_bottom_pad = 0 
        
        self.down_proj_right_pad = smallest_even_divide_number(self.hidden_size, 256) - self.hidden_size
        self.down_proj_bottom_pad = self.gate_up_proj_right_pad // 2
            
    def forward(self, hidden_states: torch.Tensor, router_logits=None, topk=None, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        To update with moe mxfp4 kernels, for now we just upcast the weights in torch.bfloat16
        """
        
        # type check, uint8 means mxfp4
        #TODO: fp8 x mxfp4 on blackwell
        assert hidden_states.dtype == torch.bfloat16
        assert self.gate_up_proj.dtype in (torch.bfloat16, torch.uint8)
        assert self.down_proj.dtype in (torch.bfloat16, torch.uint8)
        assert self.gate_up_proj_bias.dtype == torch.float32
        assert self.down_proj_bias.dtype == torch.float32

        # Shape check, only check non-mxfp4
        if self.gate_up_proj.dtype != torch.uint8:
            assert hidden_states.ndim == 2
            assert hidden_states.shape[-1] == self.gate_up_proj.shape[-2]
            assert self.down_proj.shape[-1] == self.gate_up_proj.shape[1]

        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
        from triton_kernels.swiglu import swiglu_fn
        from triton_kernels.routing import routing
        # TODO: needed in the context of device_map, maybe not for TP
        with torch.cuda.device(hidden_states.device):
            renormalize = True
            routing_data, gather_idx, scatter_idx = routing(router_logits, topk, sm_first=not renormalize)
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),(self.alpha, None), 2)

            apply_router_weight_on_input = False
            intermediate_cache1 = matmul_ogs(hidden_states,
                                            self.gate_up_proj,
                                            self.gate_up_proj_bias,
                                            routing_data,
                                            gather_indx=gather_idx,
                                            precision_config=self.gate_up_proj_precision_config,
                                            gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
                                            fused_activation=act)

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                self.down_proj_bias,
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
                gammas=None if apply_router_weight_on_input else routing_data.gate_scal)

            # manually crop the tensor since oai kernel pad the output
            output_states = intermediate_cache3[..., :self.hidden_size].contiguous()
        torch.cuda.synchronize()
        return output_states

def mlp_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = self.router(hidden_states)
    routed_out = self.experts(hidden_states, router_logits=router_logits, topk=self.top_k)
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
    pre_quantized=False,
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
        if isinstance(module, nn.Linear):
            raise NotImplementedError("Mxfp4 linear layer is not implemented yet")
        if module.__class__.__name__ == "OpenAIMoeExperts":
            with init_empty_weights():
                # tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                model._modules[name] = Mxfp4OpenAIMoeExperts(config)
                has_been_replaced=True
        if module.__class__.__name__ == "OpenAIMoeMLP":
            from types import MethodType
            module.forward = MethodType(mlp_forward, module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_mxfp4_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
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
    pre_quantized=False,
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
        pre_quantized=pre_quantized,
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
