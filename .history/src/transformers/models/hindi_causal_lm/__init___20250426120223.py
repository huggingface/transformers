# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_sentencepiece_available,
)

__all__ = [
    "HindiCausalLMConfig",
    "HindiCausalLMTokenizer",
    "HindiCausalLMModel",
    "HindiCausalLMForCausalLM",
    "HindiCausalLMPreTrainedModel",
    "HindiCausalLMHeadModel",
]

_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# Tokenizer: real or dummy
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .dummy_tokenizer_objects import HindiCausalLMTokenizer
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# Model: real or dummy
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .dummy_pt_objects import (
        HindiCausalLMPreTrainedModel,
        HindiCausalLMModel,
        HindiCausalLMForCausalLM,
    )
    HindiCausalLMHeadModel = HindiCausalLMForCausalLM
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMHeadModel",
    ]

if TYPE_CHECKING:
    from .configuration_hindi_causal_lm import HindiCausalLMConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_pt_objects import (
            HindiCausalLMPreTrainedModel,
            HindiCausalLMModel,
            HindiCausalLMForCausalLM,
        )
        HindiCausalLMHeadModel = HindiCausalLMForCausalLM
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMPreTrainedModel,
            HindiCausalLMModel,
            HindiCausalLMForCausalLM,
            HindiCausalLMHeadModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
