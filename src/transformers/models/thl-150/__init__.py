"""
Copyright 2025 EGen. All rights reserved.

Licensed under the EGen License, Version 0.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://huggingface.co/ErebusTN/EGen_V1/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

# Define import structure
_import_structure = {
    "configuration_thl_150": ["THL150Config"],
    "tokenization_thl_150": ["THL150Tokenizer"],
}

# Add fast tokenizer if tokenizers is available
if is_tokenizers_available():
    _import_structure["tokenization_thl_150_fast"] = ["THL150TokenizerFast"]

# Add modeling classes if torch is available
if is_torch_available():
    _import_structure["modeling_thl_150"] = [
        "THL150ForCausalLM",
        "THL150ForQuestionAnswering",
        "THL150Model",
        "THL150PreTrainedModel",
        "THL150ForSequenceClassification",
        "THL150ForTokenClassification",
    ]

# Handle TYPE_CHECKING imports
if TYPE_CHECKING:
    from .configuration_thl_150 import THL150Config
    from .tokenization_thl_150 import THL150Tokenizer

    if is_tokenizers_available():
        from .tokenization_thl_150_fast import THL150TokenizerFast

    if is_torch_available():
        from .modeling_thl_150 import (
            THL150ForCausalLM,
            THL150ForQuestionAnswering,
            THL150ForSequenceClassification,
            THL150ForTokenClassification,
            THL150Model,
            THL150PreTrainedModel,
        )

# Use lazy loading for runtime
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)