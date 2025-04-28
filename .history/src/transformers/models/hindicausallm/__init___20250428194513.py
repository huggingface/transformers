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

from ...utils import _LazyModule, is_torch_available, is_tokenizers_available, is_sentencepiece_available
from ...utils.import_utils import define_import_structure


_import_structure = {
    "configuration_hindicausallm": ["HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "HindiCausalLMConfig"],
    "generation_config_hindicausallm": ["HindiCausalLMGenerationConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_hindicausallm"] = ["HindiCausalLMTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_hindicausallm_fast"] = ["HindiCausalLMTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_hindicausallm"] = [
        "HindiCausalLMForCausalLM",
        "HindiCausalLMForSequenceClassification",
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_hindicausallm import HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig
    from .generation_config_hindicausallm import HindiCausalLMGenerationConfig
    
    if is_sentencepiece_available():
        from .tokenization_hindicausallm import HindiCausalLMTokenizer
    
    if is_tokenizers_available():
        from .tokenization_hindicausallm_fast import HindiCausalLMTokenizerFast
    
    if is_torch_available():
        from .modeling_hindicausallm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
else:
    import sys
    
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, globals()["_import_structure"], module_spec=__spec__)