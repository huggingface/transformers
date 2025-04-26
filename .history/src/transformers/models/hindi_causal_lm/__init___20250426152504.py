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

# flake8: noqa
# Ignore file added error E402

from typing import TYPE_CHECKING

# Ensure all necessary utilities are imported first
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)

# Define the structure for lazy loading when dependencies ARE available
_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# --- Tokenizer Handling (Real vs Dummy) ---
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # If sentencepiece is NOT available, import the dummy class...
    from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    # ...and EXPLICITLY assign it to the module attribute name.
    # This ensures the attribute ALWAYS exists.
    HindiCausalLMTokenizer = HindiCausalLMTokenizer
else:
    # If sentencepiece IS available, add the real tokenizer to the lazy load structure.
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# --- PyTorch Model Handling (Real vs Dummy) ---
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # If PyTorch is NOT available, import the dummy classes...
    from .dummy_pt_objects import (
        HindiCausalLMModel,
        HindiCausalLMForCausalLM,
     
    )
    # ...and EXPLICITLY assign them to the module attribute names.
    # This ensures the attributes ALWAYS exist.
    HindiCausalLMModel = HindiCausalLMModel
    HindiCausalLMForCausalLM = HindiCausalLMForCausalLM # <<< Focus: Ensure this assignment happens

    # The alias also needs to point to the dummy when torch is missing
    HindiCausalLMHeadModel = HindiCausalLMForCausalLM
else:
    # If PyTorch IS available, add the real model classes (and alias) to the lazy load structure.
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM", # Real class for lazy load
        "HindiCausalLMHeadModel", # Real alias for lazy load
    ]


# --- TYPE_CHECKING Block (for static analysis and IDEs) ---
# This block is ignored at runtime but helps type checkers.
if TYPE_CHECKING:
    from .configuration_hindi_causal_lm import HindiCausalLMConfig

    # Handle tokenizer for type checking
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Use dummy for type checking if missing
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        # Use real for type checking if available
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    # Handle models for type checking
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Use dummies for type checking if missing
        from .dummy_pt_objects import (
            HindiCausalLMModel,
            HindiCausalLMForCausalLM
        )
        HindiCausalLMHeadModel = HindiCausalLMForCausalLM # Alias points to dummy
    else:
        # Use real models for type checking if available
        from .modeling_hindi_causal_lm import (
            HindiCausalLMModel,
            HindiCausalLMForCausalLM,
            HindiCausalLMHeadModel, # Import real alias
        )

# --- Lazy Module Setup ---
# This part should ONLY run if not in TYPE_CHECKING mode
else:
    import sys
    # Replace the current module (__name__) with the lazy loader proxy object.
    # This proxy intercepts attribute access and loads modules from _import_structure on demand.
    # It does NOT handle the cases where dependencies are missing (that's done by the try/except blocks above).
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )

# --- Public API Definition ---
# Optional but good practice: Define what is considered public API.
__all__ = list(_import_structure.keys()) # Start with keys from lazy structure
# Manually add names exposed directly in the except blocks
if not is_sentencepiece_available():
    __all__.append("HindiCausalLMTokenizer")
if not is_torch_available():
    __all__.extend([
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM"
        "HindiCausalLMHeadModel",
    ])
# Ensure no duplicates if added both ways (though structure prevents this)
__all__ = sorted(list(set(__all__)))

