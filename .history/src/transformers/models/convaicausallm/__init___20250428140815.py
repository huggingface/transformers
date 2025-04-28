# Copyright 2024 Convai Innovations Inc. and The HuggingFace Inc. team. All rights reserved.
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

from ....utils import _LazyModule, is_sentencepiece_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_convaicausallm": ["CONVAICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvaiCausalLMConfig"],
}


if is_sentencepiece_available():
    _import_structure["tokenization_convaicausallm"] = ["ConvaiCausalLMTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_convaicausallm_fast"] = ["ConvaiCausalLMTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_convaicausallm"] = [
        "ConvaiCausalLMForCausalLM",
        "ConvaiCausalLMModel",
        "ConvaiCausalLMPreTrainedModel",
        # Add other model classes if you implement them (e.g., ForSequenceClassification)
        # "ConvaiCausalLMForSequenceClassification",
    ]


if TYPE_CHECKING:

    if is_sentencepiece_available():
        pass

    if is_tokenizers_available():
        pass

    if is_torch_available():
        pass


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
