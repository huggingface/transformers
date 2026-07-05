from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Gemma4ForConditionalGeneration,
)

from .configuration_quatfit1 import Quatfit1Config


class Quatfit1ForConditionalGeneration(Gemma4ForConditionalGeneration):
    config_class = Quatfit1Config


# Register automatically when imported

AutoConfig.register(
    "quatfit1",
    Quatfit1Config,
)

AutoModel.register(
    Quatfit1Config,
    Quatfit1ForConditionalGeneration,
)

AutoModelForCausalLM.register(
    Quatfit1Config,
    Quatfit1ForConditionalGeneration,
)

AutoModelForImageTextToText.register(
    Quatfit1Config,
    Quatfit1ForConditionalGeneration,
)
