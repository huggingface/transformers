# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from typing import TYPE_CHECKING

# Lazy Imports
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)


_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# SentencePiece tokenizer: Required for the slow tokenizer.
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["dummy_tokenizer_objects"] = ["HindiCausalLMTokenizer"]
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# PyTorch models: Check if PyTorch is installed
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # Import dummy objects if PyTorch is not available
    _import_structure["dummy_pt_objects"] = [
        "HindiCausalLMForCausalLM",
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
    ]
else:
    # Import actual PyTorch model classes if PyTorch is available
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMForCausalLM",
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
    ]


# Direct imports for type checking
if TYPE_CHECKING:
    # Configuration is always available
    from .configuration_hindi_causal_lm import HindiCausalLMConfig

    # SentencePiece tokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import dummy tokenizer if SentencePiece is not available
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        # Import actual tokenizer if SentencePiece is available
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    # PyTorch models
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import dummy objects if PyTorch is not available
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
    else:
        # Import actual PyTorch model classes if PyTorch is available
        from .modeling_hindi_causal_lm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

# Set up lazy module loading
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},  # Add any objects available regardless of dependencies here if needed
    )
