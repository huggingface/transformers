# flake8: noqa
# Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.
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
"""Hindi Causal Language Model models."""

from typing import TYPE_CHECKING

# Required for lazy loading
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# Define the lazy loading structure
# Filename mapping to contained objects
_import_structure = {
    "configuration_hindicausallm": ["HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "HindiCausalLMConfig"],
    "generation_config_hindicausallm": ["HindiCausalLMGenerationConfig"], # Added generation config
}

# Add modeling files if PyTorch is installed
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No PyTorch models if torch is not installed
else:
    _import_structure["modeling_hindicausallm"] = [
        # Add PRETRAINED_MODEL_ARCHIVE_LIST if defined in modeling_hindicausallm.py
        # "HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMForSequenceClassification",
        # Add other model variants like HindiCausalLMForTokenClassification if implemented
    ]

# Add slow tokenizer if SentencePiece is installed
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No slow tokenizer if sentencepiece is not installed
else:
    _import_structure["tokenization_hindicausallm"] = ["HindiCausalLMTokenizer"]

# Add fast tokenizer if Tokenizers is installed
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No fast tokenizer if tokenizers is not installed
else:
    _import_structure["tokenization_hindicausallm_fast"] = ["HindiCausalLMTokenizerFast"]


# Direct imports for type checking
if TYPE_CHECKING:
    # Unconditional imports
    from .configuration_hindicausallm import HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig
    from .generation_config_hindicausallm import HindiCausalLMGenerationConfig # Added generation config

    # Conditional imports for modeling files
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Define dummy objects here if needed for type hints when torch is missing
        # For example:
        # class HindiCausalLMPreTrainedModel: pass
        # class HindiCausalLMModel: pass
        # class HindiCausalLMForCausalLM: pass
        # class HindiCausalLMForSequenceClassification: pass
         pass # Or simply pass if dummy objects aren't strictly needed for type hints
    else:
        # from .modeling_hindicausallm import HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST # Uncomment if exists
        from .modeling_hindicausallm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
            # Add other model variants if implemented
        )

    # Conditional imports for slow tokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
         # Define dummy tokenizer if needed
         # class HindiCausalLMTokenizer: pass
         pass
    else:
        from .tokenization_hindicausallm import HindiCausalLMTokenizer

    # Conditional imports for fast tokenizer
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass # No dummy needed for fast tokenizer usually, as slow can be used
    else:
        from .tokenization_hindicausallm_fast import HindiCausalLMTokenizerFast

# Setup the lazy module export system
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)