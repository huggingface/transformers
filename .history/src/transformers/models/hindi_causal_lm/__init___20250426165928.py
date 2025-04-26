# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the MIT License.
#

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available


_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# Remove this entire block for now
# try:
#     if not is_tokenizers_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["tokenization_hindi_causal_lm_fast"] = ["HindiCausalLMTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMForCausalLM",
    ]

if TYPE_CHECKING:
    from .configuration_hindi_causal_lm import HindiCausalLMConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    # Also remove this block
    # try:
    #     if not is_tokenizers_available():
    #         raise OptionalDependencyNotAvailable()
    # except OptionalDependencyNotAvailable:
    #     pass
    # else:
    #     from .tokenization_hindi_causal_lm_fast import HindiCausalLMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
            HindiCausalLMForCausalLM,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
