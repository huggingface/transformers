import json
from typing import Any

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.quantizers import HfQuantizer, register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


@register_quantization_config("custom")
class CustomConfig(QuantizationConfigMixin):
    def __init__(self):
        self.quant_method = "custom"
        self.bits = 8

    def to_dict(self) -> dict[str, Any]:
        output = {
            "num_bits": self.bits,
        }
        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        config_dict = self.to_dict()

        default_config_dict = CustomConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@register_quantizer("custom")
class CustomQuantizer(HfQuantizer):
    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config
        self.scale_map = {}
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = kwargs.get("dtype", torch.float32)

    def _process_model_before_weight_loading(self, model, **kwargs):
        return True

    def _process_model_after_weight_loading(self, model, **kwargs):
        return True

    def is_serializable(self) -> bool:
        return True

    def is_trainable(self) -> bool:
        return False


model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", quantization_config=CustomConfig(), dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
input_text = "once there is"
inputs = tokenizer(input_text, return_tensors="pt")
output = model_8bit.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
