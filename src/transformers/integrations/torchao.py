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

import re
import types

import torch

from transformers.utils import logging
from transformers.utils.import_utils import is_torch_available, is_torchao_available


if is_torch_available():
    from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name


if is_torchao_available():
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )
    from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao

logger = logging.get_logger(__name__)


def _quantization_type(weight):
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor

    if isinstance(weight, AffineQuantizedTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if isinstance(weight, LinearActivationQuantizedTensor):
        return f"{weight.__class__.__name__}(activation={weight.input_quant_func}, weight={_quantization_type(weight.original_weight_tensor)})"


def _linear_extra_repr(self):
    weight = _quantization_type(self.weight)
    if weight is None:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight=None"
    else:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={weight}"


class TorchAoQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def _quantize(self, module, config, *args, **kwargs):
        """Run quantize_, moving to CUDA first if CPU offloading is active.

        Some torchao quantization ops (e.g. int4 packing) only have CUDA kernels.
        When a layer is destined for CPU (e.g. CPU offloading), we temporarily move
        it to CUDA for quantization, then move the result back to CPU.
        """
        from torchao.quantization import quantize_

        target_device = next(module.parameters()).device
        if self.hf_quantizer.offload_to_cpu and target_device.type == "cpu":
            module.to("cuda")
            quantize_(module, config, *args, **kwargs)
            module.to("cpu")
        else:
            quantize_(module, config, *args, **kwargs)

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        module, tensor_name = get_module_from_name(model, full_layer_name)

        module._parameters[tensor_name] = torch.nn.Parameter(value, requires_grad=value.requires_grad)
        # if we are quantizing tied parameters, to avoid tying the quantized weights
        # the correct order to do it is
        # 1. load the weight to model
        # 2. run tie_weights to populate the weights
        # 3. quantize
        input_embed = model.get_input_embeddings()
        is_embedding_param = id(module) == id(input_embed)
        untie_embedding_weights = self.hf_quantizer.quantization_config.untie_embedding_weights

        if untie_embedding_weights and is_embedding_param:
            setattr(model.config.get_text_config(decoder=True), "tie_word_embeddings", False)

        from torchao.quantization import FqnToConfig

        config = self.hf_quantizer.quantization_config.get_apply_tensor_subclass()
        if isinstance(config, FqnToConfig):
            module_fqn, top_level_param_name = full_layer_name.rsplit(".", 1)
            c = None
            if full_layer_name in config.fqn_to_config:
                assert not module_fqn.startswith("re:"), (
                    "param fqn should not start with`re:`, which is used for specifying regex"
                )
                c = config.module_fqn_to_config[full_layer_name]
            elif module_fqn in config.fqn_to_config:
                assert not module_fqn.startswith("re:"), (
                    "module fqn should not start with`re:`, which is used for specifying regex"
                )
                c = config.module_fqn_to_config[module_fqn]
            # regex match module and param
            else:
                for maybe_module_fqn_pattern in config.fqn_to_config:
                    # if key doesn't start with re, it is an exact fqn key, so we don't regex match
                    if not maybe_module_fqn_pattern.startswith("re:"):
                        continue
                    # see if param matches first
                    elif re.fullmatch(maybe_module_fqn_pattern[3:], full_layer_name):
                        c = config.module_fqn_to_config[maybe_module_fqn_pattern]
                        break
                    elif re.fullmatch(maybe_module_fqn_pattern[3:], module_fqn):
                        # we'll apply the config for first fully matched pattern
                        c = config.module_fqn_to_config[maybe_module_fqn_pattern]
                        break
                else:
                    c = config.module_fqn_to_config.get("_default", None)

            if c is not None:
                if top_level_param_name == "weight":
                    if is_embedding_param and untie_embedding_weights:
                        lm_head = module.weight.clone()
                    # we can apply the module config directly
                    self._quantize(module, c, (lambda x, fqn: True))
                    missing_keys.discard(full_layer_name)
                    module._is_hf_initialized = True
                    # torchao quantizes weights into a module but some models access the weight directly
                    # (e.g. module.o_proj.weight). The _is_hf_initialized flag is set at the module
                    # level only, so we also set it on each parameter to prevent _init_weights from
                    # calling normal_() on already-quantized Float8Tensors.
                    for param in module.parameters(recurse=False):
                        param._is_hf_initialized = True
                    return {"lm_head.weight": lm_head} if is_embedding_param and untie_embedding_weights else {}
                else:
                    # need to apply to custom param name
                    custom_param_fqn_config = FqnToConfig({top_level_param_name: c})
                    self._quantize(module, custom_param_fqn_config, filter_fn=None)
                    missing_keys.discard(full_layer_name)
                    module._is_hf_initialized = True
                    for param in module.parameters(recurse=False):
                        param._is_hf_initialized = True
                    return {}
            return {full_layer_name: value}

        if is_embedding_param and untie_embedding_weights:
            lm_head = module.weight.clone()
        self._quantize(module, self.hf_quantizer.quantization_config.get_apply_tensor_subclass())
        missing_keys.discard(full_layer_name)
        module._is_hf_initialized = True
        for param in module.parameters(recurse=False):
            param._is_hf_initialized = True
        return {"lm_head.weight": lm_head} if is_embedding_param and untie_embedding_weights else {}


class TorchAoDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str] | None = None,
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Consolidates tensor subclass components before reconstructing the object

        For example:
            input_dict: {
                "_weight_qdata": torch.Tensor,
                "_weight_scale": torch.Tensor,
            }
            full_layer_name: "model.layers.0.self_attn.k_proj.weight"

            Given this, we reconstruct a Float8Tensor instance using the qdata and scale
            and return it as a dictionary with the full_layer_name as the key and the recovered
            Float8Tensor instance as the value.
        """
        is_unsafe_serialization = list(input_dict.keys())[0] not in source_patterns

        param_data = {}
        layer_name = ".".join(full_layer_name.split(".")[:-1])
        if is_unsafe_serialization:
            if isinstance(input_dict["weight"], list):
                weight = input_dict["weight"][0]
            else:
                weight = input_dict["weight"]
        else:
            for suffix in input_dict.keys():
                if len(input_dict[suffix]) != 1:
                    raise ValueError(
                        f"Expected a single tensor for {suffix} but got {len(input_dict[suffix])} tensors instead"
                    )
                param_data[f"{layer_name}.{suffix}"] = input_dict[suffix][0]

        # If it's unsafe-serialized (i.e. not safetensors), no need for anything
        if is_unsafe_serialization:
            return {full_layer_name: weight}
        elif not is_metadata_torchao(self.hf_quantizer.metadata):
            raise ValueError("Invalid torchao safetensors metadata")

        unflattened_state_dict, leftover_state_dict = unflatten_tensor_state_dict(
            param_data, self.hf_quantizer.metadata
        )
        assert not leftover_state_dict  # there should be no unprocessed tensors
        new_param = unflattened_state_dict[full_layer_name]

        module, _ = get_module_from_name(model, full_layer_name)
        # Add repr to the module
        if isinstance(module, torch.nn.Linear):
            module.extra_repr = types.MethodType(_linear_extra_repr, module)

        return {full_layer_name: new_param}
