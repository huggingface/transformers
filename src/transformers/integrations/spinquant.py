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

from typing import Optional

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)

### Copied partially from executorch library
class Int8DynActInt8WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int

    """
    This module implements a dynamic quantized linear layer with int8 weight.
    Weights are per channel quantized. Parameters of importance
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, 1),
                dtype=torch.float32,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from torchao.quantization.utils import per_token_dynamic_quant
        input = per_token_dynamic_quant(input.to(self.precision))
        n_bit = 8
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1
        w_dq = torch.ops.quantized_decomposed.dequantize_per_channel(
            self.weight,
            self.scales,
            None,
            0,
            quant_min,
            quant_max,
            torch.int8,
            out_dtype=self.precision,
        )
        return torch.nn.functional.linear(input, w_dq)

class Int8QuantizedEmbedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.precision = precision

        self.register_buffer(
            "weight",
            torch.zeros(
                (vocab_size, embedding_dim), dtype=torch.int8
            ),
        )
        self.register_buffer(
            "scales", torch.ones((vocab_size, 1), dtype=torch.float32)
        )

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized_decomposed.embedding_byte.dtype(
            self.weight, self.scales, None, -128, 127, indices, dtype=self.precision
        )


class Int8DynActInt4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Parameters of importance
    groupsize: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert (
            in_features % groupsize == 0
        ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // groupsize),
                dtype=self.precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from torchao._executorch_ops import _quantized_decomposed_dequantize_per_channel_group_wrapper
        from torchao.quantization.utils import per_token_dynamic_quant

        input = per_token_dynamic_quant(input.to(self.precision))
        n_bit = 4
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1
        w_dq = _quantized_decomposed_dequantize_per_channel_group_wrapper(
            self.weight,
            self.scales,
            None,
            quant_min,
            quant_max,
            torch.int8,
            self.groupsize,
            self.precision,
        )
        return torch.nn.functional.linear(input, w_dq)

def spinquant_r3_forward(self, x):
    """
    SpinQuant needs two Hadmard matrixes: R3 and R4. Here we are only injecting R4 in the feed forward layer.
    R3 needs to be injected as well when KV cache quantization is enabled.
    """
    w = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    n = w.shape[-1]
    return self.down_proj(torch.ops.llama.fast_hadamard_transform(w.contiguous()) / torch.tensor(n).sqrt())

def _replace_with_spinquant_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)
        if name not in modules_to_not_convert:
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                # Check if the current key is not in the `modules_to_not_convert`
                with init_empty_weights(include_buffers=True):
                    if isinstance(module, nn.Linear) and "lm_head" in current_key_name_str:
                        model._modules[name] = Int8DynActInt8WeightLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            precision=torch.float32,
                        )
                        model._modules[name].requires_grad_(False)
                        has_been_replaced = True
                    elif isinstance(module, nn.Linear) and "lm_head" not in current_key_name_str:
                        model._modules[name] = Int8DynActInt4WeightLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            precision=torch.float32,
                            groupsize=32,
                        )
                        model._modules[name].requires_grad_(False)
                        has_been_replaced = True
                    elif isinstance(module, nn.Embedding):
                        model._modules[name] = Int8QuantizedEmbedding(
                        vocab_size=module.weight.shape[0],
                        embedding_dim=module.weight.shape[1],
                        precision=torch.float32,
                        )
                        model._modules[name].requires_grad_(False)
                        has_been_replaced = True
                    elif module.__class__.__name__ == "LlamaMLP":
                        model.forward = spinquant_r3_forward
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_spinquant_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_spinquant_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FbgemmFp8Linear` modules.

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

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_spinquant_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using FP8 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
