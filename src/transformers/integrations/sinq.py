# src/transformers/integrations/sinq.py
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
      - we take the loaded weight tensor,
      - we build a temporary dense nn.Linear (weight-only),
      - we wrap it into SINQLinear,
      - we replace the original module inside the model.

    This is structurally similar to TorchAoQuantize / Fp8Quantize, but
    we delegate SINQ-specific details here instead of in the HfQuantizer.
    """

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def _get_runtime_device_str(self) -> str:
        cfg = self.hf_quantizer.quantization_config
        device_str = getattr(self.hf_quantizer, "_normalized_device_str", None)
        if device_str is not None:
            return device_str
        if getattr(cfg, "device", None) in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return str(cfg.device)

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

        module_path, _, _ = full_layer_name.rpartition(".") 
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        from sinq.sinqlinear import SINQLinear
        from sinq.sinqlinear import sinq_base_quant_config as sinq_base_quant_config_fn

        #device_str = self._get_runtime_device_str()
        #device = torch.device(device_str)
        device = weight_tensor.device
        compute_dtype = self.hf_quantizer.dtype or weight_tensor.dtype

        in_features = weight_tensor.shape[1]
        out_features = weight_tensor.shape[0]

        dense = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=weight_tensor.dtype,
        )
        with torch.no_grad():
            dense.weight.copy_(weight_tensor)#.to(device=device, dtype=weight_tensor.dtype))

        cfg = self.hf_quantizer.quantization_config
        sinq_quant_dict = self.hf_quantizer._build_sinq_quant_dict(cfg)
        device_str = self._get_runtime_device_str()

        sinq_layer = SINQLinear(
            linear_layer=dense,
            quant_config=sinq_quant_dict,
            del_orig=True,
            compute_dtype=compute_dtype,
            device=device_str,
            use_unpack_kernel=True,
            layer_activations=None,
        )

        setattr(parent, child_name, sinq_layer)

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        sinq_layer._is_hf_initialized = True

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
      - and call this SinqDeserialize.convert to reconstruct a SINQLinear.

    The returned dict is {} because we perform module replacement directly.
    """

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def _get_runtime_device_str(self) -> str:
        cfg = self.hf_quantizer.quantization_config
        device_str = getattr(self.hf_quantizer, "_normalized_device_str", None)
        if device_str is not None:
            return device_str
        if getattr(cfg, "device", None) in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return str(cfg.device)

    def convert(
        self,
        input_dict: Dict[str, Any],
        model: Optional["torch.nn.Module"] = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        from sinq.sinqlinear import SINQLinear 

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

        #device_str = self._get_runtime_device_str()
        #device = torch.device(device_str)
        compute_dtype = self.hf_quantizer.dtype or W_q.dtype

        module_path, _, _ = full_layer_name.rpartition(".") 
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        device_str = self._get_runtime_device_str()

        sinq_layer = SINQLinear(
            linear_layer=None,
            quant_config=None,
            del_orig=True,
            compute_dtype=compute_dtype,
            device=device_str,
            use_unpack_kernel=True,
            layer_activations=None,
        )

        state = {
            "W_q": W_q,
            "meta": meta,
        }
        if bias is not None:
            state["bias"] = bias

        sinq_layer.load_state_dict(state)

        setattr(parent, child_name, sinq_layer)
        sinq_layer._is_hf_initialized = True

        return {}