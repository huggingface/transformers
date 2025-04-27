# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Hindi Causal Language Model by ConvAI Innovations"""

# Need OrderedDict for the sorting logic below
from collections import OrderedDict
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)


# Define the full import structure alphabetically by key
_import_structure = {
    "configuration_hindi_causal_lm": ["HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP", "HindiCausalLMConfig"],
}

# Add modeling if torch is available
if is_torch_available():
    _import_structure["modeling_hindi_causal_lm"] = [
        "HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
    ]

# Add tokenizer if sentencepiece is available
if is_sentencepiece_available():
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# Add fast tokenizer if tokenizers library is available
if is_tokenizers_available():
    _import_structure["tokenization_hindi_causal_lm_fast"] = ["HindiCausalLMTokenizerFast"]

# Conditionally add dummy objects if dependencies are missing
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # Add dummy PyTorch objects
    from .dummy_pt_objects import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
        HindiCausalLMPreTrainedModel,
    )

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # Add dummy sentencepiece objects
    from .dummy_tokenizer_objects import HindiCausalLMTokenizer

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # No specific dummy needed for fast tokenizer
    pass

# Type checking
if TYPE_CHECKING:
    # Configuration is always available
    from .configuration_hindi_causal_lm import HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig

    # Modeling structure (conditional on torch)
    if is_torch_available():
        from .modeling_hindi_causal_lm import (
            HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST,
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
    else:
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

    # Tokenization structure (conditional on sentencepiece)
    if is_sentencepiece_available():
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer
    else:
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer

    # Fast Tokenization (conditional on tokenizers)
    if is_tokenizers_available():
        from .tokenization_hindi_causal_lm_fast import HindiCausalLMTokenizerFast

else:
    # Standard lazy module loading
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)