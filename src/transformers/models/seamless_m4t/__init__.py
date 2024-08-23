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
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_seamless_m4t": ["SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP", "SeamlessM4TConfig"],
    "feature_extraction_seamless_m4t": ["SeamlessM4TFeatureExtractor"],
    "processing_seamless_m4t": ["SeamlessM4TProcessor"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_seamless_m4t"] = ["SeamlessM4TTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_seamless_m4t_fast"] = ["SeamlessM4TTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_seamless_m4t"] = [
        "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SeamlessM4TForTextToSpeech",
        "SeamlessM4TForSpeechToSpeech",
        "SeamlessM4TForTextToText",
        "SeamlessM4TForSpeechToText",
        "SeamlessM4TModel",
        "SeamlessM4TPreTrainedModel",
        "SeamlessM4TCodeHifiGan",
        "SeamlessM4THifiGan",
        "SeamlessM4TTextToUnitForConditionalGeneration",
        "SeamlessM4TTextToUnitModel",
    ]

if TYPE_CHECKING:
    from .configuration_seamless_m4t import SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP, SeamlessM4TConfig
    from .feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
    from .processing_seamless_m4t import SeamlessM4TProcessor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t import SeamlessM4TTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_seamless_m4t_fast import SeamlessM4TTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_seamless_m4t import (
            SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST,
            SeamlessM4TCodeHifiGan,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
            SeamlessM4THifiGan,
            SeamlessM4TModel,
            SeamlessM4TPreTrainedModel,
            SeamlessM4TTextToUnitForConditionalGeneration,
            SeamlessM4TTextToUnitModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
