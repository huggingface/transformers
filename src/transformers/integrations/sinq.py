# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from typing import Any

from transformers.utils import is_torch_available, logging

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn as nn


def replace_with_sinq_linear(
    model: torch.nn.Module,
    modules_to_not_convert: list[str] | None = None,
    quant_config: dict | None = None,
    compute_dtype: torch.dtype = None,
    device: str = "cuda:0",
    pre_quantized: bool = False,
) -> torch.nn.Module:
    """
    Replace nn.Linear modules with empty SINQLinear modules.

    Args:
        model: The model to modify
        modules_to_not_convert: List of module names to skip
        quant_config: SINQ quantization config dict (None for pre-quantized models)
        compute_dtype: Computation dtype for the quantized layers
        device: Device string for the quantized layers
        pre_quantized: Whether loading a pre-quantized checkpoint

    Returns:
        The modified model with SINQLinear modules
    """
    from sinq.sinqlinear_hf import SINQLinear

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_convert_module(full_name, modules_to_not_convert):
            continue

        parent_path, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        sinq_layer = SINQLinear(
            in_features=module.in_features if not pre_quantized else None,
            out_features=module.out_features if not pre_quantized else None,
            bias=(module.bias is not None) if not pre_quantized else False,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            use_unpack_kernel=True,
        )

        setattr(parent, child_name, sinq_layer)

    return model


class SinqQuantize(ConversionOps):
    """
    Param-level ConversionOp for SINQ (from FP weights).

    At load time, for each `Linear.weight` that should be quantized:
      - The SINQLinear module already exists (created in _process_model_before_weight_loading)
      - We just call quantize() on it with the loaded weight tensor
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, Any],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, values = next(iter(input_dict.items()))
        weight_tensor = values[0] if isinstance(values, list) else values

        module, tensor_name = get_module_from_name(model, full_layer_name)

        module.quantize(weight_tensor)

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        module._is_hf_initialized = True

        return {}


class SinqDeserialize(ConversionOps):
    """
    ConversionOp for loading *pre-quantized* SINQ checkpoints.

    Checkpoint layout (what `SINQLinear.state_dict` produces) is, per module:
        <prefix>.W_q
        <prefix>.bias
        <prefix>.meta

    WeightConverter in the quantizer is configured so that:
      - we group ".W_q", ".meta", ".bias" as input_dict
      - conceptually treat them as belonging to "<prefix>.weight"
      - and call this SinqDeserialize.convert to load the state into the existing SINQLinear.

    The returned dict is {} because we load directly into the module.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, Any],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        for k, v in list(input_dict.items()):
            if isinstance(v, list):
                input_dict[k] = v[0]

        W_q = input_dict.get(".W_q")
        meta = input_dict.get(".meta")
        bias = input_dict.get(".bias")

        # Fallback path: if W_q or meta is missing, this is not a valid SINQ checkpoint.
        # Return the tensor as-is so standard HF weight loading can handle it.
        if W_q is None or meta is None:
            v = next(iter(input_dict.values()))
            if isinstance(v, list):
                v = v[0]
            return {full_layer_name: v}

        module, _ = get_module_from_name(model, full_layer_name)

        state = {
            "W_q": W_q,
            "meta": meta,
        }
        if bias is not None:
            state["bias"] = bias

        module.load_state_dict(state)
        module._is_hf_initialized = True

        return {}
