# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_speecht5": ["SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "SpeechT5Config"],
    "processing_speecht5": ["SpeechT5Processor"],
    "tokenization_speecht5": ["SpeechT5CTCTokenizer"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_speecht5"] = [
        "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SpeechT5ForAudioFrameClassification",
        "SpeechT5ForCTC",
        "SpeechT5ForPreTraining",
        "SpeechT5ForSequenceClassification",
        "SpeechT5ForXVector",
        "SpeechT5Model",
        "SpeechT5PreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_speecht5 import SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP, SpeechT5Config
    from .processing_speecht5 import SpeechT5Processor
    from .tokenization_speecht5 import SpeechT5CTCTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speecht5 import (
            SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            SpeechT5ForAudioFrameClassification,
            SpeechT5ForCTC,
            SpeechT5ForPreTraining,
            SpeechT5ForSequenceClassification,
            SpeechT5ForXVector,
            SpeechT5Model,
            SpeechT5PreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
