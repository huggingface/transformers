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


class Mxfp4Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, weight_dtype=torch.float32):
        super().__init__(in_features, out_features, bias)
        # dtype torch.float4_e2m1fn not supported yet
        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float8_e5m2))
        # self.weight_scale = torch.nn.Parameter(torch.zeros((out_features, 1), dtype=weight_dtype))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((out_features), dtype=weight_dtype))
        else:
            self.bias = None

    def forward(self, x):
        """
        update
        """
        return


# maybe subclass
class Mxfp4OpenaiExperts(nn.Module):
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

        # self.gate_up_proj_scale = torch.nn.Parameter(
        #     torch.zeros((self.num_experts, 1, self.expert_dim * 2))
        # )
        # self.down_proj_scale = torch.nn.Parameter(
        #     torch.zeros((self.num_experts, self.hidden_size, 1))
        # )

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        To update with moe mxfp4 kernels, for now we just upcast the weights in torch.bfloat16
        """
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
                expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted:
                with torch.no_grad():
                    idx, top_x = torch.where(
                        expert_mask[expert_idx][0]
                    )  # idx: top-1/top-2 indicator, top_x: token indices
                current_state = hidden_states[top_x]  # (num_tokens, hidden_dim)
                gate_up = (
                    current_state @ self.gate_up_proj[expert_idx].to(torch.bfloat16) + self.gate_up_proj_bias[expert_idx]
                )  # (num_tokens, 2 * interm_dim)
                gate, up = gate_up.chunk(2, dim=-1)  # (num_tokens, interm_dim)
                glu = gate * torch.sigmoid(gate * self.alpha)  # (num_tokens, interm_dim)
                gated_output = (up + 1) * glu  # (num_tokens, interm_dim)
                out = (
                    gated_output @ self.down_proj[expert_idx].to(torch.bfloat16) + self.down_proj_bias[expert_idx]
                )  # (num_tokens, hidden_dim)
                weighted_output = out * routing_weights[top_x, idx, None]  # (num_tokens, hidden_dim)
                next_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype)[0])
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj.to(torch.bfloat16)) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj.to(torch.bfloat16)) + self.down_proj_bias[..., None, :]
            next_states = next_states.view(-1, self.hidden_size)
        return next_states

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

        if (isinstance(module, nn.Linear)) and should_convert_module(current_key_name, modules_to_not_convert):
            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features
                model._modules[name] = Mxfp4Linear(
                    in_features,
                    out_features,
                    module.bias is not None,
                )
                has_been_replaced = True
                model._modules[name].requires_grad_(False)
        if module.__class__.__name__ == "OpenaiExperts" and should_convert_module(
            current_key_name, modules_to_not_convert
        ):
            with init_empty_weights():
                # tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                model._modules[name] = Mxfp4OpenaiExperts(config)
                has_been_replaced=True

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
