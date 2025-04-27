# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team.
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
"""ConvAI Innovations Hindi Causal LM model."""

from typing import TYPE_CHECKING

# Make sure TYPE_CHECKING is imported from typing
# from typing import TYPE_CHECKING # Already imported above, but ensure it's present
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,  # Ensure this utility is imported
    is_torch_available,
)


# Define all objects that should be publicly available at the module level
__all__ = [
    "HindiCausalLMConfig",
    "HindiCausalLMTokenizer",
    "HindiCausalLMModel",
    "HindiCausalLMForCausalLM",
    "HindiCausalLMHeadModel", # Include the alias
    "HindiCausalLMPreTrainedModel",
]

# Structure for lazy loading when dependencies ARE available
_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# --- Tokenizer Handling ---
try:
    # Check if sentencepiece is available
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # If sentencepiece is NOT available, import the dummy class...
    from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    # ...and EXPLICITLY assign it to the module attribute name.
    HindiCausalLMTokenizer = HindiCausalLMTokenizer # <<< THIS LINE IS CRUCIAL
else:
    # If sentencepiece IS available, add the real tokenizer to the lazy load structure.
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# --- Model Handling ---
try:
    # Check if PyTorch is available
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # If PyTorch is NOT available, import the dummy classes...
    from .dummy_pt_objects import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
        HindiCausalLMPreTrainedModel,
    )
    # ...and EXPLICITLY assign them to the module attribute names.
    HindiCausalLMModel = HindiCausalLMModel # <<< THIS LINE IS CRUCIAL
    HindiCausalLMForCausalLM = HindiCausalLMForCausalLM
    HindiCausalLMPreTrainedModel = HindiCausalLMPreTrainedModel
    # The alias also needs to point to the dummy when torch is missing
    HindiCausalLMHeadModel = HindiCausalLMForCausalLM
else:
    # If PyTorch IS available, add the real model classes (and alias) to the lazy load structure.
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMHeadModel", # Add the alias here too
    ]

# --- TYPE_CHECKING Block (for static analysis and IDEs) ---
if TYPE_CHECKING:
    # Import config directly
    from .configuration_hindi_causal_lm import HindiCausalLMConfig

    # Import tokenizer (real or dummy based on availability)
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    # Import models (real or dummy based on availability)
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
        # Alias for TYPE_CHECKING when torch is missing
        HindiCausalLMHeadModel = HindiCausalLMForCausalLM
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMHeadModel,  # Import the real alias
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

# --- Lazy Module Setup ---
else:
    import sys
    # Replace the current module with the lazy loader
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )

