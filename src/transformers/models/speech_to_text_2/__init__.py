# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ...utils import _LazyModule, is_sentencepiece_available, is_speech_available, is_torch_available


_import_structure = {
    "configuration_speech_to_text_2": [
        "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Speech2Text2Config",
    ],
    "processing_speech_to_text_2": ["Speech2Text2Processor"],
    "tokenization_speech_to_text_2": ["Speech2Text2Tokenizer"],
}


if is_torch_available():
    _import_structure["modeling_speech_to_text_2"] = [
        "SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speech2Text2ForCausalLM",
        "Speech2Text2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_speech_to_text_2 import SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2Text2Config
    from .processing_speech_to_text_2 import Speech2Text2Processor
    from .tokenization_speech_to_text_2 import Speech2Text2Tokenizer

    if is_torch_available():
        from .modeling_speech_to_text_2 import (
            SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2Text2ForCausalLM,
            Speech2Text2PreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
