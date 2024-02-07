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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_torchaudio_available,
)


_import_structure = {
    "configuration_musicgen_melody": [
        "MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_musicgen_melody"] = [
        "MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MusicgenMelodyForConditionalGeneration",
        "MusicgenMelodyForCausalLM",
        "MusicgenMelodyModel",
        "MusicgenMelodyPreTrainedModel",
    ]

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_musicgen_melody"] = ["MusicgenMelodyFeatureExtractor"]
    _import_structure["processing_musicgen_melody"] = ["MusicgenMelodyProcessor"]


if TYPE_CHECKING:
    from .configuration_musicgen_melody import (
        MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_musicgen_melody import (
            MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST,
            MusicgenMelodyForCausalLM,
            MusicgenMelodyForConditionalGeneration,
            MusicgenMelodyModel,
            MusicgenMelodyPreTrainedModel,
        )

    try:
        if not is_torchaudio_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
        from .processing_musicgen_melody import MusicgenMelodyProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
