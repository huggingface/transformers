# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Dict, Any

from transformers.utils import is_torch_available, logging

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn as nn

class SinqQuantize(ConversionOps):
    """
    Param-level ConversionOp for SINQ (from FP weights).

    At load time, for each `Linear.weight` that should be quantized:
      - The SINQLinear module already exists (created in _process_model_before_weight_loading)
      - We just call quantize() on it with the loaded weight tensor
    """

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: Dict[str, Any],
        model: Optional["torch.nn.Module"] = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:

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

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: Dict[str, Any],
        model: Optional["torch.nn.Module"] = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:

        for k, v in list(input_dict.items()):
            if isinstance(v, list):
                input_dict[k] = v[0]

        W_q = input_dict.get(".W_q", None)
        meta = input_dict.get(".meta", None)
        bias = input_dict.get(".bias", None)

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