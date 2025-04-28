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
from typing import TYPE_CHECKING

# Step 1: Define the expected structure for lazy loading.
# Each key is the filename (without .py) and the value is a list of object names expected in that file.
_import_structure = {
    "configuration_hindicausallm": ["HindiCausalLMConfig"],
    "generation_config_hindicausallm": ["HindiCausalLMGenerationConfig"],
}

# Step 2: Add conditional imports based on dependencies.
from ...utils import (
    OptionalDependencyNotAvailable,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No slow tokenizer if sentencepiece is missing
else:
    _import_structure["tokenization_hindicausallm"] = ["HindiCausalLMTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No fast tokenizer if tokenizers is missing
else:
    _import_structure["tokenization_hindicausallm_fast"] = ["HindiCausalLMTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass # No PyTorch models if torch is missing
else:
    _import_structure["modeling_hindicausallm"] = [
        "HindiCausalLMForCausalLM",
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMForSequenceClassification",
    ]

# Step 3: Define direct imports for type checking tools (like mypy).
# This section is skipped during runtime but helps linters and IDEs.
if TYPE_CHECKING:
    from .configuration_hindicausallm import HindiCausalLMConfig
    from .generation_config_hindicausallm import HindiCausalLMGenerationConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hindicausallm import HindiCausalLMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hindicausallm_fast import HindiCausalLMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hindicausallm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

# Step 4: Setup the lazy module importer.
# This replaces the current module object with a lazy one that only imports
# submodules and objects when they are actually accessed.
else:
    import sys

    # Standard Hugging Face lazy module setup
    from ...utils import _LazyModule
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure, # Pass the defined structure
        module_spec=__spec__,
        extra_objects={}, # No extra top-level objects needed here
    )