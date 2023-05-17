# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)


_import_structure = {
    "configuration_fastspeech2_conformer": [
        "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP",
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
    ],
    "processing_fastspeech2_conformer": ["FastSpeech2ConformerProcessor"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_fastspeech2_conformer"] = ["FastSpeech2ConformerTokenizer"]

try:
    if not is_speech_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_fastspeech2_conformer"] = ["FastSpeech2ConformerFeatureExtractor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_fastspeech2_conformer"] = [
        "FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FastSpeech2ConformerForSpeechToText",
        "FastSpeech2ConformerForSpeechToSpeech",
        "FastSpeech2ConformerForTextToSpeech",
        "FastSpeech2ConformerModel",
        "FastSpeech2ConformerPreTrainedModel",
        "FastSpeech2ConformerHifiGan",
    ]

if TYPE_CHECKING:
    from .configuration_fastspeech2_conformer import (
        FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
    )
    from .processing_fastspeech2_conformer import FastSpeech2ConformerProcessor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_fastspeech2_conformer import FastSpeech2ConformerTokenizer

    try:
        if not is_speech_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_fastspeech2_conformer import FastSpeech2ConformerFeatureExtractor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_fastspeech2_conformer import (
            FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            FastSpeech2ConformerForSpeechToSpeech,
            FastSpeech2ConformerForSpeechToText,
            FastSpeech2ConformerForTextToSpeech,
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
