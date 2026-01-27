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

from functools import lru_cache

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module
from ..utils import (
    is_accelerate_available,
    is_fbgemm_gpu_available,
    is_torch_available,
    is_torch_xpu_available,
    logging,
)


if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights

_is_torch_xpu_available = is_torch_xpu_available()

if is_fbgemm_gpu_available() and not _is_torch_xpu_available:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

logger = logging.get_logger(__name__)


class FbgemmFp8Quantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_key, value = tuple(input_dict.items())[0]
        value = value[0]

        from ..integrations import FbgemmFp8Llama4TextExperts

        module, tensor_name = get_module_from_name(model, target_key)

        if isinstance(module, FbgemmFp8Llama4TextExperts):
            if tensor_name == "gate_up_proj":
                # Process each expert separately
                # Transpose the second and third dimension
                transposed_param = value.transpose(1, 2)

                # Reshape to 2D for quantization
                original_shape = transposed_param.shape
                flattened_param = transposed_param.reshape(-1, original_shape[-1])

                # Quantize using per row instead of per column
                new_value_flat, weight_scale_flat = quantize_fp8_per_row(flattened_param)

                # Reshape back to original dimensions
                new_value = new_value_flat.reshape(original_shape)
                new_value = new_value.transpose(1, 2)
                weight_scale = weight_scale_flat.reshape(original_shape[0], 1, original_shape[1])
            elif tensor_name == "down_proj":
                # Process each expert separately
                # Transpose the weights for proper quantization
                transposed_param = value.transpose(1, 2)

                # Reshape to 2D for quantization
                original_shape = transposed_param.shape
                flattened_param = transposed_param.reshape(-1, original_shape[-1])

                # Quantize using per column
                new_value_flat, weight_scale_flat = quantize_fp8_per_row(flattened_param)

                # Reshape back to original dimensions
                new_value = new_value_flat.reshape(original_shape)
                new_value = new_value.transpose(1, 2)
                weight_scale = weight_scale_flat.reshape(original_shape[0], original_shape[1], 1)
        else:
            new_value, weight_scale = quantize_fp8_per_row(value)
            weight_scale = torch.nn.Parameter(weight_scale.view(weight_scale.shape[0], 1))

        return {target_key: torch.nn.Parameter(new_value), f"{target_key}_scale": weight_scale}


class FbgemmFp8Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, dtype=torch.float8_e4m3fn):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features), dtype=dtype))
        self.weight_scale = torch.nn.Parameter(torch.zeros((out_features, 1), dtype=torch.float32))
        self.register_buffer("input_scale_ub", torch.zeros([1], dtype=torch.float), persistent=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.out_features), dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        # quantize_fp8_per_row will squash the leading dimensions, so save the desired shape here
        output_shape = (*x.shape[:-1], -1)
        # x_quantized and x_scale are not necessarily on the same device as x, this is an issue.
        # https://github.com/pytorch/FBGEMM/blob/e08af8539c391437f447173863df0f3f6f6f1855/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L1237C3-L1237C45
        x_quantized, x_scale = quantize_fp8_per_row(x.view(-1, x.shape[-1]).contiguous(), scale_ub=self.input_scale_ub)
        # moving x_quantized, x_scale here creates glibberish output ... However, if we move the output, it works
        # x_quantized, x_scale = x_quantized.to(x.device), x_scale.to(x.device)

        # The computation still happens on the device where self.weight is even if x_quantized is not on the same device as self.weight
        weight_scale_float32 = self.weight_scale.to(torch.float32)
        if _is_torch_xpu_available:
            output = torch._scaled_mm(
                x_quantized,
                self.weight.t(),
                scale_a=x_scale.unsqueeze(-1),
                scale_b=weight_scale_float32.t(),
                out_dtype=x.dtype,
                bias=self.bias,
            )
        else:
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
            expert_quantized, expert_scale = quantize_fp8_per_row(
                expert_hidden_reshaped, num_tokens, self.input_scale_ub
            )
            sharded_expert_dim = self.gate_up_proj.shape[-1] // 2
            gate_up_proj_scale_float32 = self.gate_up_proj_scale.to(torch.float32)
            if _is_torch_xpu_available:
                gate = torch._scaled_mm(
                    expert_quantized,
                    self.gate_up_proj[i].transpose(0, 1)[:sharded_expert_dim].contiguous().t(),
                    scale_a=expert_scale.unsqueeze(-1),
                    scale_b=gate_up_proj_scale_float32[i][0][:sharded_expert_dim].view(-1, 1).contiguous().t(),
                    out_dtype=hidden_states.dtype,
                )
                up = torch._scaled_mm(
                    expert_quantized,
                    self.gate_up_proj[i].transpose(0, 1)[sharded_expert_dim:].contiguous().t(),
                    scale_a=expert_scale.unsqueeze(-1),
                    scale_b=gate_up_proj_scale_float32[i][0][sharded_expert_dim:].view(-1, 1).contiguous().t(),
                    out_dtype=hidden_states.dtype,
                )
            else:
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

            activated_quantized, activated_scale = quantize_fp8_per_row(activated, num_tokens, self.input_scale_ub)

            down_proj_scale_float32 = self.down_proj_scale.to(torch.float32)
            if _is_torch_xpu_available:
                expert_output = torch._scaled_mm(
                    activated_quantized,
                    self.down_proj[i].transpose(0, 1).contiguous(),
                    scale_a=activated_scale.unsqueeze(-1),
                    scale_b=down_proj_scale_float32[i].view(-1, 1).contiguous().t(),
                    out_dtype=hidden_states.dtype,
                )
            else:
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


@lru_cache(maxsize=1)
def get_quantize_fp8_per_row():
    if _is_torch_xpu_available:
        from .hub_kernels import get_kernel

        return get_kernel("kernels-community/fp8-fbgemm").quantize_fp8_per_row
    return torch.ops.fbgemm.quantize_fp8_per_row


def replace_with_fbgemm_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False, tp_plan=None
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FbgemmFp8Linear` modules.
    This will enable running your models using high performance fp8 kernel from FBGEMM library.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert. In practice we keep the `lm_head` in full precision for numerical stability reasons.
        quantization_config (`FbgemmFp8Config`):
            The quantization config object that contains the quantization parameters.
        pre_quantized (`book`, defaults to `False`):
            Whether the model is pre-quantized or not
    """
    global quantize_fp8_per_row
    quantize_fp8_per_row = get_quantize_fp8_per_row()

    has_been_replaced = False
    module_kwargs = {} if pre_quantized else {"dtype": None}

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        new_module = None
        with init_empty_weights(include_buffers=True):
            if module.__class__.__name__ == "Llama4TextExperts":
                # TODO: make sure tp works later
                # if tp_plan is not None:
                #     tp_key = re.sub(r"\d+", "*", f"{module_name}.down_proj_scale")
                #     tp_plan[tp_key] = None
                text_config = getattr(model.config, "text_config", model.config)
                new_module = FbgemmFp8Llama4TextExperts(text_config or model.config)
            elif isinstance(module, nn.Linear):
                new_module = FbgemmFp8Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **module_kwargs,
                )
                new_module.requires_grad_(False)

        if new_module is None:
            continue

        if hasattr(new_module, "input_scale_ub"):
            new_module.input_scale_ub = torch.tensor(
                [quantization_config.activation_scale_ub],
                dtype=torch.float,
            )

        model.set_submodule(module_name, new_module)
        has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using FP8 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
