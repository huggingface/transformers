import json
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import HfApi

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.quantizers import HfQuantizer, get_module_from_name, register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


# Implement INT8 Symmetric Linear layer
class Int8SymmetricLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros((out_features, 1), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        dequant_weight = self.weight * self.weight_scale
        output = F.linear(x, dequant_weight)
        if self.bias is not None:
            output = output + self.bias
        return output


# Function to replace standard linear layers with INT8 symmetric quantized layers
def _replace_with_int8_symmetric_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Recursively replaces nn.Linear modules with Int8SymmetricLinear modules.
    """
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
                    model._modules[name] = Int8SymmetricLinear(
                        in_features, out_features, module.bias is not None, dtype=module.weight.dtype
                    )
                    has_been_replaced = True
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_int8_symmetric_linear(
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


def replace_with_int8_symmetric_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    Main function to replace model layers with INT8 symmetric quantized versions.
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))

    model, has_been_replaced = _replace_with_int8_symmetric_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        raise ValueError(
            "You are loading your model using INT8 symmetric quantization but no linear modules were found in your model."
        )

    return model


@register_quantization_config("int8_symmetric")
class Int8SymmetricConfig(QuantizationConfigMixin):
    """
    Configuration for INT8 symmetric quantization.
    """

    def __init__(self, modules_to_not_convert: Optional[list[str]] = None, **kwargs):
        self.quant_method = "int8_symmetric"
        self.modules_to_not_convert = modules_to_not_convert

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        config_dict = self.to_dict()
        default_config_dict = Int8SymmetricConfig().to_dict()

        serializable_config_dict = {}
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@register_quantizer("int8_symmetric")
class Int8SymmetricQuantizer(HfQuantizer):
    """
    Implementation of INT8 symmetric quantization.

    """

    requires_calibration = False
    requires_parameters_quantization = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def _process_model_before_weight_loading(self, model, **kwargs):
        """
        Replace model's linear layers with quantized versions before loading weights.
        """
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        model = replace_with_int8_symmetric_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def check_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, Int8SymmetricLinear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.int8:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def create_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
    ):
        """
        Quantizes weights to INT8 symmetric format.
        """
        abs_max_per_row = torch.max(torch.abs(param_value), dim=1, keepdim=True)[0].clamp(min=1e-5)

        weight_scale = abs_max_per_row / 127.0

        weight_quantized = torch.round(param_value / weight_scale).clamp(-128, 127).to(torch.int8)

        module, tensor_name = get_module_from_name(model, param_name)
        module._buffers[tensor_name] = weight_quantized.to(target_device)
        module._buffers["weight_scale"] = weight_scale.to(target_device)

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, Int8SymmetricLinear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Post-processing after weights are loaded.
        """
        return True

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False


# Example usage
if __name__ == "__main__":
    model_int8 = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", quantization_config=Int8SymmetricConfig(), dtype=torch.float, device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    input_text = "once there is"
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")
    output = model_int8.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    # Save and upload to HUB
    output_model_dir = "Llama-3.2-1B-INT8-CUSTOM"
    model_int8.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    api = HfApi()
    repo_id = "medmekk/Llama-3.2-1B-INT8-CUSTOM"
    api.create_repo(repo_id, private=False)
    api.upload_folder(folder_path=output_model_dir, repo_id=repo_id, repo_type="model")
