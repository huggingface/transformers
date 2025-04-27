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
    "modeling_hindi_causal_lm": [ # modeling comes after configuration
        "HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
    ],
    # tokenization entries come after modeling
    "tokenization_hindi_causal_lm": ["HindiCausalLMTokenizer"],
    "tokenization_hindi_causal_lm_fast": ["HindiCausalLMTokenizerFast"],
}

# Conditionally add dummy object paths alphabetically if dependencies are missing
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # Add dummy classes for PyTorch objects if torch is not available
    # Ensure dummy keys are also inserted alphabetically if needed
    # 'dummy_pt_objects' comes between 'configuration' and 'modeling'
    _import_structure = {
        "configuration_hindi_causal_lm": _import_structure["configuration_hindi_causal_lm"],
        "dummy_pt_objects": [
            "HindiCausalLMPreTrainedModel",
            "HindiCausalLMModel",
            "HindiCausalLMForCausalLM",
        ],
        **{k: v for k, v in _import_structure.items() if k != "configuration_hindi_causal_lm"}, # Add remaining items
    }
    # Re-sort just in case (though adding alphabetically should maintain order)
    _import_structure = OrderedDict(sorted(_import_structure.items()))


try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
     # Add dummy class for SentencePiece tokenizer if not available
     # 'dummy_tokenizer_objects' comes between 'dummy_pt_objects' and 'modeling'
     _import_structure = {
         **{k: v for k, v in _import_structure.items() if k < "dummy_tokenizer_objects"}, # Items before
         "dummy_tokenizer_objects": ["HindiCausalLMTokenizer"],
         **{k: v for k, v in _import_structure.items() if k >= "dummy_tokenizer_objects"}, # Items after
     }
     # Re-sort just in case
     _import_structure = OrderedDict(sorted(_import_structure.items()))


try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # No specific dummy needed for fast tokenizer if tokenizers lib is missing
    # But remove the fast tokenizer entry if it exists and tokenizers is not available
    _import_structure.pop("tokenization_hindi_causal_lm_fast", None) # Use pop with default None


# Ensure final structure is sorted before passing to LazyModule
_import_structure = OrderedDict(sorted(_import_structure.items()))


if TYPE_CHECKING:
    # Configuration is always available
    from .configuration_hindi_causal_lm import HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig

    # Modeling structure (conditional on torch)
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import dummy objects if torch is not available
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
    else:
        # Import real modeling classes if torch is available
        from .modeling_hindi_causal_lm import (
            HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST,
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

    # Tokenization structure (conditional on sentencepiece)
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
         # Import dummy object if sentencepiece is not available
         from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        # Import real tokenizer class if sentencepiece is available
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    # Fast Tokenization structure (conditional on tokenizers)
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # No fast tokenizer class to import if tokenizers lib is missing
        pass
    else:
         # Import real fast tokenizer class if tokenizers lib is available
        from .tokenization_hindi_causal_lm_fast import HindiCausalLMTokenizerFast

else:
    # Standard lazy module loading
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)