# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)


_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# SentencePiece tokenizer
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["dummy_tokenizer_objects"] = ["HindiCausalLMTokenizer"]
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# PyTorch model
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # Ensure dummy objects are available for type checking and lazy loading
    _import_structure["dummy_pt_objects"] = [
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
    ]
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
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
        # Import dummy objects for type checking
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
    else:
        # Import actual objects for type checking
        from .modeling_hindi_causal_lm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)