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

from ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_wav2vec2": ["WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Wav2Vec2Config"],
    "feature_extraction_wav2vec2": ["Wav2Vec2FeatureExtractor"],
    "processing_wav2vec2": ["Wav2Vec2Processor"],
    "tokenization_wav2vec2": ["Wav2Vec2CTCTokenizer", "Wav2Vec2Tokenizer"],
}


if is_torch_available():
    _import_structure["modeling_wav2vec2"] = [
        "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2ForAudioFrameClassification",
        "Wav2Vec2ForCTC",
        "Wav2Vec2ForMaskedLM",
        "Wav2Vec2ForPreTraining",
        "Wav2Vec2ForSequenceClassification",
        "Wav2Vec2ForXVector",
        "Wav2Vec2Model",
        "Wav2Vec2PreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_wav2vec2"] = [
        "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFWav2Vec2ForCTC",
        "TFWav2Vec2Model",
        "TFWav2Vec2PreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_wav2vec2"] = [
        "FlaxWav2Vec2ForCTC",
        "FlaxWav2Vec2ForPreTraining",
        "FlaxWav2Vec2Model",
        "FlaxWav2Vec2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_wav2vec2 import WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2Config
    from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
    from .processing_wav2vec2 import Wav2Vec2Processor
    from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer

    if is_torch_available():
        from .modeling_wav2vec2 import (
            WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForCTC,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForXVector,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_wav2vec2 import (
            TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWav2Vec2ForCTC,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )

    if is_flax_available():
        from .modeling_tf_wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
