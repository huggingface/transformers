# Copyright 2024 The HuggingFace Team. All rights reserved.
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
    "configuration_wav2vec2_bert": [
        "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2BERTConfig",
    ],
    "feature_extraction_wav2vec2_bert": ["Wav2Vec2BERTFeatureExtractor"],
    "processing_wav2vec2_bert": ["Wav2Vec2BERTProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_wav2vec2_bert"] = [
        "WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2BERTForAudioFrameClassification",
        "Wav2Vec2BERTForCTC",
        "Wav2Vec2BERTForPreTraining",
        "Wav2Vec2BERTForSequenceClassification",
        "Wav2Vec2BERTForXVector",
        "Wav2Vec2BERTModel",
        "Wav2Vec2BERTPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_wav2vec2_bert import (
        WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2BERTConfig,
    )
    from .feature_extraction_wav2vec2_bert import Wav2Vec2BERTFeatureExtractor
    from .processing_wav2vec2_bert import Wav2Vec2BERTProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_wav2vec2_bert import (
            WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2BERTForAudioFrameClassification,
            Wav2Vec2BERTForCTC,
            Wav2Vec2BERTForPreTraining,
            Wav2Vec2BERTForSequenceClassification,
            Wav2Vec2BERTForXVector,
            Wav2Vec2BERTModel,
            Wav2Vec2BERTPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
