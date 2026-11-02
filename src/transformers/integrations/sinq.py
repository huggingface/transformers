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

import json
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
            # the placeholder `weight` is where the serialized tensors of a pre-quantized checkpoint are collected
            in_features=module.in_features,
            out_features=module.out_features,
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


# The tensors of a `SINQLinear` `meta` (`scale` / `zero` are dicts `x`, `s`, `m` with the "quantAux" methods), saved
# as `<module>.meta.<path>`; the other entries of `meta` (shape, nbits, packing...) go in the checkpoint metadata.


def _set_nested(meta: dict, path: str, value):
    *parents, name = path.split(".")
    for parent in parents:
        meta = meta.setdefault(parent, {})
    meta[name] = value


def _sinq_meta(hf_quantizer, input_dict, prefix, module_name, index=0):
    """
    The `meta` dict of a `SINQLinear`: its non-tensor entries from the checkpoint metadata (`{path: value}` per module,
    dtypes as `"torch.bfloat16"`), its tensors from `input_dict`, under `{prefix}meta.<path>`.
    """
    meta = {}
    for path, value in hf_quantizer.metadata.get(module_name, {}).items():
        if isinstance(value, str) and isinstance(getattr(torch, value.removeprefix("torch."), None), torch.dtype):
            value = getattr(torch, value.removeprefix("torch."))
        _set_nested(meta, path, value)
    for key, value in input_dict.items():
        if key.startswith(f"{prefix}meta."):
            _set_nested(meta, key.removeprefix(f"{prefix}meta."), value[index] if isinstance(value, list) else value)
    return meta


class SinqSerialize(ConversionOps):
    """
    Saves a `SINQLinear` as `W_q`, `bias` and the tensors of its `meta` (`meta.<path>`); the rest of `meta` goes in
    `metadata` (JSON per module), which `SinqDeserialize` / `SinqDequantize` read back.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer
        self.metadata = {}

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, torch.Tensor]:
        module = model.get_submodule(full_layer_name)
        if not getattr(module, "ready", False) or getattr(module, "W_q", None) is None:
            return input_dict  # not a quantized `SINQLinear`
        state = module.state_dict()  # `meta` tensors on cpu, quantAux tuples as dicts
        output = {f"{full_layer_name}.{name}": state[name] for name in ("W_q", "bias") if name in state}
        others = {}

        def flatten(path, value):
            if isinstance(value, dict):
                for name, item in value.items():
                    flatten(f"{path}.{name}" if path else name, item)
            elif isinstance(value, torch.Tensor):
                output[f"{full_layer_name}.meta.{path}"] = value.contiguous()
            else:
                others[path] = str(value) if isinstance(value, torch.dtype) else value

        flatten("", state["meta"])
        self.metadata[f"{full_layer_name}.meta"] = json.dumps(others, default=list)
        return output


class SinqDeserialize(ConversionOps):
    """
    Loads a pre-quantized `SINQLinear`: `W_q`, `bias` and its `meta`, rebuilt from the `meta.<path>` tensors and the
    checkpoint metadata (see `SinqSerialize`). The module is loaded directly, so nothing is returned.
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
        input_dict = {key: value[0] if isinstance(value, list) else value for key, value in input_dict.items()}
        module_name = full_layer_name.removesuffix(".weight")
        if ".W_q" not in input_dict:
            # e.g. the `.bias` of a module that is not quantized
            return {f"{module_name}{key}": value for key, value in input_dict.items()}

        module = model.get_submodule(module_name)
        state = {"W_q": input_dict[".W_q"], "meta": _sinq_meta(self.hf_quantizer, input_dict, ".", module_name)}
        if ".bias" in input_dict:
            state["bias"] = input_dict[".bias"]
        module.load_state_dict(state)
        # the placeholder `weight` is not needed anymore, as after `SINQLinear.quantize`
        module._parameters.pop("weight", None)
        module.weight = None
        module._is_hf_initialized = True
        if missing_keys is not None:
            missing_keys.discard(full_layer_name)
        return {}


class SinqDequantize(ConversionOps):
    """
    Dense weights from the `SINQLinear` state: `W_q`, its `meta` dict (scales, zeros, shape, packing...) and `bias`,
    dequantized by sinq itself as the format has many variants (NF codebooks, tiling, dual scales).
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        from sinq.quantizer import Quantizer

        module_name = full_layer_name.removesuffix(".weight")
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith("W_q"):
                continue  # `meta.*` and `bias`
            prefix = key.removesuffix("W_q")  # `.W_q` -> `.weight`, `experts.*.W_q` -> `experts.*.weight`
            weights = weights if isinstance(weights, list) else [weights]
            output[f"{prefix}weight"] = [
                Quantizer.dequantize(weight, _sinq_meta(self.hf_quantizer, input_dict, prefix, module_name, i))
                for i, weight in enumerate(weights)
            ]
            bias = input_dict.get(f"{prefix}bias")
            if bias is not None:
                output[f"{prefix}bias"] = bias if isinstance(bias, list) else [bias]
        return output
