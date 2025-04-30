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
from ..utils import is_accelerate_available, is_fbgemm_gpu_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights

if is_fbgemm_gpu_available():
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

logger = logging.get_logger(__name__)


class FbgemmFp8Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, weight_dtype=torch.float32):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.weight_scale = torch.nn.Parameter(torch.zeros((out_features, 1), dtype=weight_dtype))
        self.register_buffer("input_scale_ub", torch.zeros([1], dtype=torch.float), persistent=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.out_features), dtype=weight_dtype))
        else:
            self.bias = None

    def forward(self, x):
        # quantize_fp8_per_row will squash the leading dimensions, so save the desired shape here
        output_shape = (*x.shape[:-1], -1)
        # x_quantized and x_scale are not necessarily on the same device as x, this is an issue.
        # https://github.com/pytorch/FBGEMM/blob/e08af8539c391437f447173863df0f3f6f6f1855/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L1237C3-L1237C45
        x_quantized, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
            x.view(-1, x.shape[-1]).contiguous(), scale_ub=self.input_scale_ub
        )
        # moving x_quantized, x_scale here creates glibberish output ... However, if we move the output, it works
        # x_quantized, x_scale = x_quantized.to(x.device), x_scale.to(x.device)

        # The computation still happens on the device where self.weight is even if x_quantized is not on the same device as self.weight
        weight_scale_float32 = self.weight_scale.to(torch.float32)
        output = torch.ops.fbgemm.f8f8bf16_rowwise(
            x_quantized, self.weight, x_scale, weight_scale_float32, use_fast_accum=True
        )
        output = output + self.bias if self.bias is not None else output
        # Hacky for now, we have the output to the device of x
        output = output.to(x.device)
        output = output.reshape(output_shape)
        del x_quantized, x_scale
        return output


class FbgemmFp8Llama4TextExperts(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        # Register FP8 buffers for gate_up_proj
        self.gate_up_proj = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, 2 * self.expert_dim), dtype=torch.float8_e4m3fn)
        )
        self.gate_up_proj_scale = torch.nn.Parameter(
            torch.zeros((self.num_experts, 1, self.expert_dim * 2), dtype=torch.float32)
        )
        # Register FP8 buffers for down_proj
        self.down_proj = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.expert_dim, self.hidden_size), dtype=torch.float8_e4m3fn)
        )
        self.down_proj_scale = torch.nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, 1), dtype=torch.float32)
        )
        # Register input scale upper bound
        self.register_buffer("input_scale_ub", torch.zeros([1], dtype=torch.float), persistent=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
        Returns:
            torch.Tensor: (batch_size * token_num, hidden_size)
        """
        # Reshape hidden states for expert computation
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        num_tokens = None

        # Pre-allocate tensor for all expert outputs with same shape as hidden_states
        next_states = torch.empty_like(hidden_states)

        for i in range(self.num_experts):
            # Extract expert's hidden states
            expert_hidden = hidden_states[i]
            expert_hidden_reshaped = expert_hidden.reshape(-1, self.hidden_size)
            # Quantize for this expert
            expert_quantized, expert_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                expert_hidden_reshaped, num_tokens, self.input_scale_ub
            )
            sharded_expert_dim = self.gate_up_proj.shape[-1] // 2
            gate_up_proj_scale_float32 = self.gate_up_proj_scale.to(torch.float32)

            gate = torch.ops.fbgemm.f8f8bf16_rowwise(
                expert_quantized,
                self.gate_up_proj[i].transpose(0, 1)[:sharded_expert_dim].contiguous(),
                expert_scale,
                gate_up_proj_scale_float32[i][0][:sharded_expert_dim].view(-1, 1).contiguous(),
                use_fast_accum=True,
            )

            up = torch.ops.fbgemm.f8f8bf16_rowwise(
                expert_quantized,
                self.gate_up_proj[i].transpose(0, 1)[sharded_expert_dim:].contiguous(),
                expert_scale,
                gate_up_proj_scale_float32[i][0][sharded_expert_dim:].view(-1, 1).contiguous(),
                use_fast_accum=True,
            )

            activated = up * self.act_fn(gate)

            activated_quantized, activated_scale = torch.ops.fbgemm.quantize_fp8_per_row(
                activated, num_tokens, self.input_scale_ub
            )

            down_proj_scale_float32 = self.down_proj_scale.to(torch.float32)
            expert_output = torch.ops.fbgemm.f8f8bf16_rowwise(
                activated_quantized,
                self.down_proj[i].transpose(0, 1).contiguous(),
                activated_scale,
                down_proj_scale_float32[i].view(-1, 1).contiguous(),
                use_fast_accum=True,
            )

            next_states[i] = expert_output
        next_states = next_states.to(hidden_states.device)
        return next_states.view(-1, self.hidden_size)


def _replace_with_fbgemm_fp8_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
    config=None,
    tp_plan=None,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """

    import re

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    in_features = module.in_features
                    out_features = module.out_features
                    model._modules[name] = FbgemmFp8Linear(
                        in_features,
                        out_features,
                        module.bias is not None,
                    )
                    has_been_replaced = True

                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                # set non persistent buffer outside of init_empty_weights
                model._modules[name].input_scale_ub = torch.tensor(
                    [quantization_config.activation_scale_ub],
                    dtype=torch.float,
                )
        if module.__class__.__name__ == "Llama4TextExperts" and name not in modules_to_not_convert:
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights(include_buffers=True):
                    tp_plan[re.sub(r"\d+", "*", current_key_name_str + ".down_proj_scale")] = None
                    model._modules[name] = FbgemmFp8Llama4TextExperts(
                        config.text_config,
                    )
                model._modules[name].input_scale_ub = torch.tensor(
                    [quantization_config.activation_scale_ub], dtype=torch.float
                )

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_fbgemm_fp8_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
                config=config,
                tp_plan=tp_plan,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_fbgemm_fp8_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    pre_quantized=False,
    config=None,
    tp_plan=None,
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FbgemmFp8Linear` modules.
    This will enable running your models using high performance fp8 kernel from FBGEMM library.

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `FP8Linear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """

    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_fbgemm_fp8_linear(
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
            "You are loading your model using FP8 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
