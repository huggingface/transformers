import importlib.metadata
import re
import types
from collections import defaultdict
from typing import Optional, Any

import torch
from packaging import version

from transformers.utils.import_utils import is_torchao_available
from transformers.utils import logging

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name


if is_torchao_available():
    TORCHAO_VERSION = version.parse(importlib.metadata.version("torchao"))
    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.14.0"):
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

    def convert(
        self, input_dict: dict[str, torch.Tensor], model: Optional[torch.nn.Module] = None, missing_keys=None, **kwargs
    ) -> dict[str, torch.Tensor]:
        # print("input_dict", input_dict)
        target_key, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        full_name = target_key
        # update param name to get the weights instead of the quantized stats
        target_key = self.hf_quantizer.get_param_name(target_key)
        module, _ = get_module_from_name(model, target_key)

        """
        Each nn.Linear layer that needs to be quantized is processed here.
        First, we set the value the weight tensor, then we move it to the target device. Finally, we quantize the module.
        """
        from torchao.quantization import quantize_

        full_name = target_key
        # Those are the pre quantized weights
        if ":" in target_key:
            target_key = target_key.rsplit(":", 1)[0]
        module, tensor_name = get_module_from_name(model, target_key)

        if self.hf_quantizer.pre_quantized:
            # If it's a bias, no need to do anything special (except removing the ":_data" part of the key, but was
            # already done) - if it's unsafe-serialized (i.e. not safetensors), not need for anything either
            is_unsafe_serialization = ":" not in full_name
            if tensor_name == "bias" or is_unsafe_serialization:
                return {full_name: value}
            # Sanity check for the new serialization format
            elif not (TORCHAO_VERSION >= version.parse("0.14.0") and is_metadata_torchao(self.hf_quantizer.metadata)):
                raise ValueError("To use `safetensors` serialization, you should have `torchao>=0.14.0` installed")

            # Save the states for later quantization when they are all gathered
            if not hasattr(self.hf_quantizer, "ao_params"):
                self.hf_quantizer.ao_params = defaultdict(dict)
            self.hf_quantizer.ao_params[target_key].update({full_name: value})
            missing_keys.discard(full_name)

            # We are ready for quantization in this case (we retrieved all the needed keys)
            if len(self.hf_quantizer.ao_params[target_key]) == len(self.hf_quantizer.weight_ao_keys):
                new_param = unflatten_tensor_state_dict(self.hf_quantizer.ao_params[target_key], self.hf_quantizer.metadata)[target_key]
                # Free memory
                del self.hf_quantizer.ao_params[target_key]

            # Add repr to the module
            if isinstance(module, torch.nn.Linear):
                module.extra_repr = types.MethodType(_linear_extra_repr, module)

            return {full_name: new_param}
        else:
            module._parameters[tensor_name] = torch.nn.Parameter(
                value, requires_grad=value.requires_grad
            ).to(value.device)
            # if we are quantizing tied parameters, to avoid tying the quantized weights
            # the correct order to do it is
            # 1. load the weight to model
            # 2. run tie_weights to populate the weights
            # 3. quantize
            input_embed = model.get_input_embeddings()
            if self.hf_quantizer.quantization_config.untie_embedding_weights and id(module) == id(input_embed):
                model.tie_weights()
                setattr(model.config.get_text_config(decoder=True), "tie_word_embeddings", False)

            # handle FqnToConfig, introduced in torchao 0.15.0+
            if self.hf_quantizer.quantization_config._get_ao_version() >= version.Version("0.15.0"):
                from torchao.quantization import FqnToConfig

                config = self.hf_quantizer.quantization_config.get_apply_tensor_subclass()
                if isinstance(config, FqnToConfig):
                    module_fqn, top_level_param_name = target_key.rsplit(".", 1)
                    c = None
                    if target_key in config.fqn_to_config:
                        assert not module_fqn.startswith("re:"), (
                            "param fqn should not start with`re:`, which is used for specifying regex"
                        )
                        c = config.module_fqn_to_config[target_key]
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
                            elif re.fullmatch(maybe_module_fqn_pattern[3:], target_key):
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
                            # we can apply the module config directly
                            quantize_(module, c, (lambda x, fqn: True))
                            missing_keys.discard(target_key)
                            module._is_hf_initialized = True
                            return {}
                        else:
                            # need to apply to custom param name
                            custom_param_fqn_config = FqnToConfig({top_level_param_name: c})
                            quantize_(module, custom_param_fqn_config, filter_fn=None)
                            missing_keys.discard(target_key)
                            module._is_hf_initialized = True
                            return {}
                    return {full_name: value}

            # handle ModuleFqnToConfig, introduced in torchao 0.12.0+
            # TODO deprecate this when we deprecate ModuleFqnToConfig
            elif self.hf_quantizer.quantization_config._get_ao_version() >= version.Version("0.12.0"):
                from torchao.quantization import ModuleFqnToConfig

                config = self.hf_quantizer.quantization_config.get_apply_tensor_subclass()
                if isinstance(config, ModuleFqnToConfig):
                    module_fqn, _ = target_key.rsplit(".", 1)
                    c = None
                    if module_fqn in config.module_fqn_to_config:
                        assert not module_fqn.startswith("re:"), (
                            "module fqn should not start with`re:`, which is used for specifying regex"
                        )
                        c = config.module_fqn_to_config[module_fqn]
                    else:
                        for maybe_module_fqn_pattern in config.module_fqn_to_config:
                            if not maybe_module_fqn_pattern.startswith("re:"):
                                continue
                            elif re.fullmatch(maybe_module_fqn_pattern[3:], module_fqn):
                                # we'll apply the config for first fully matched pattern
                                c = config.module_fqn_to_config[maybe_module_fqn_pattern]
                                break
                        else:
                            c = config.module_fqn_to_config.get("_default", None)
                    if c is not None:
                        # filter_fn: not filtering out any modules
                        quantize_(module, c, filter_fn=lambda x, fqn: True)
                        missing_keys.discard(full_name)
                        module._is_hf_initialized = True
                    return {full_name: value}

            quantize_(module, self.hf_quantizer.quantization_config.get_apply_tensor_subclass())
            missing_keys.discard(full_name)
            module._is_hf_initialized = True
            return {}