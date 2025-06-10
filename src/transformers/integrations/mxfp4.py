# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ..activations import ACT2FN
from ..utils import is_accelerate_available, is_torch_available, logging

if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights


logger = logging.get_logger(__name__)

class Mxfp4Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, weight_dtype=torch.float32):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features

        # dtype torch.float4_e2m1fn not supported yet
        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.weight_scale = torch.nn.Parameter(torch.zeros((out_features, 1), dtype=weight_dtype))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.out_features), dtype=weight_dtype))
        else:
            self.bias = None

    def forward(self, x):
        """
        update
        """
        return

class Mxfp4OaiTextExperts(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        
        # 
        self.gate_up_proj = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, 2 * self.expert_dim), dtype=torch.float8_e4m3fn)
        )
        # self.gate_up_proj_scale = torch.nn.Parameter(
        #     torch.zeros((self.num_experts, 1, self.expert_dim * 2), dtype=torch.float32)
        # )
        self.down_proj = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.expert_dim, self.hidden_size), dtype=torch.float8_e4m3fn)
        )
        # self.down_proj_scale = torch.nn.Parameter(
        #     torch.zeros((self.num_experts, self.hidden_size, 1), dtype=torch.float32)
        # )

    def forward(self, hidden_states):
        # Reshape hidden states for expert computation
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        num_tokens = None

        # Pre-allocate tensor for all expert outputs with same shape as hidden_states
        next_states = torch.empty_like(hidden_states)

        for i in range(self.num_experts):
            expert_hidden = hidden_states[i]
            expert_hidden_reshaped = expert_hidden.reshape(-1, self.hidden_size)
            """
            Update 
            
            """
            # next_states[i] = ...
        next_states = next_states.to(hidden_states.device)
        return next_states.view(-1, self.hidden_size)

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
    import re

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    in_features = module.in_features
                    out_features = module.out_features
                    model._modules[name] = Mxfp4Linear(
                        in_features,
                        out_features,
                        module.bias is not None,
                    )
                    has_been_replaced = True
                    model._modules[name].requires_grad_(False)
        if module.__class__.__name__ == "Llama4TextExperts" and name not in modules_to_not_convert:
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                    model._modules[name] = Mxfp4OaiTextExperts(
                        config.text_config,
                    )
                model._modules[name].input_scale_ub = torch.tensor(
                    [quantization_config.activation_scale_ub], dtype=torch.float
                )

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