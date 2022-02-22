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

from ...file_utils import _LazyModule, is_torch_available


_import_structure = {
    "configuration_data2vec_audio": [
        "DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecAudioConfig",
    ],
    "configuration_data2vec_text": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecTextConfig",
        "Data2VecTextOnnxConfig",
    ],
}

if is_torch_available():
    _import_structure["modeling_data2vec_audio"] = [
        "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Data2VecAudioForAudioFrameClassification",
        "Data2VecAudioForCTC",
        "Data2VecAudioForSequenceClassification",
        "Data2VecAudioForXVector",
        "Data2VecAudioModel",
        "Data2VecAudioPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_text"] = [
        "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Data2VecTextForCausalLM",
        "Data2VecTextForMaskedLM",
        "Data2VecTextForMultipleChoice",
        "Data2VecTextForQuestionAnswering",
        "Data2VecTextForSequenceClassification",
        "Data2VecTextForTokenClassification",
        "Data2VecTextModel",
        "Data2VecTextPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_data2vec_audio import DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP, Data2VecAudioConfig
    from .configuration_data2vec_text import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecTextConfig,
        Data2VecTextOnnxConfig,
    )

    if is_torch_available():
        from .modeling_data2vec_audio import (
            DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
        )
        from .modeling_data2vec_text import (
            DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
