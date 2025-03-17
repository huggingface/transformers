# Copyright 2025 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
    is_torch_available,
    is_torchaudio_available,
)
from ...utils.import_utils import define_import_structure

_import_structure = {
    "configuration_granite_speech": [
        "GraniteSpeechConfig",
        "GraniteSpeechEncoderConfig",
        "GraniteSpeechProjectorConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_granite_speech"] = [
        "GraniteSpeechForConditionalGeneration",
    ]

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_granite_speech"] = ["GraniteSpeechFeatureExtractor"]
    _import_structure["processing_granite_speech"] = ["GraniteSpeechProcessor"]



if TYPE_CHECKING:
    from .configuration_granite_speech import (
        GraniteSpeechConfig,
        GraniteSpeechEncoderConfig,
        GraniteSpeechProjectorConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_granite_speech import (
            GraniteSpeechForConditionalGeneration,
            GraniteSpeechPretrainedModel,
        )

    try:
        if not is_torchaudio_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_granite_speech import GraniteSpeechFeatureExtractor
        from .processing_granite_speech import GraniteSpeechProcessor
else:
    import sys
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
