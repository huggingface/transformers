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

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_clip": ["CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP", "CLIPConfig", "CLIPTextConfig", "CLIPVisionConfig"],
    "tokenization_clip": ["CLIPTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_clip_fast"] = ["CLIPTokenizerFast"]

if is_vision_available():
    _import_structure["feature_extraction_clip"] = ["CLIPFeatureExtractor"]
    _import_structure["processing_clip"] = ["CLIPProcessor"]

if is_torch_available():
    _import_structure["modeling_clip"] = [
        "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CLIPModel",
        "CLIPPreTrainedModel",
        "CLIPTextModel",
        "CLIPVisionModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_clip"] = [
        "FlaxCLIPModel",
        "FlaxCLIPPreTrainedModel",
        "FlaxCLIPTextModel",
        "FlaxCLIPTextPreTrainedModel",
        "FlaxCLIPVisionModel",
        "FlaxCLIPVisionPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_clip import CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    from .tokenization_clip import CLIPTokenizer

    if is_tokenizers_available():
        from .tokenization_clip_fast import CLIPTokenizerFast

    if is_vision_available():
        from .feature_extraction_clip import CLIPFeatureExtractor
        from .processing_clip import CLIPProcessor

    if is_torch_available():
        from .modeling_clip import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPVisionModel,
        )

    if is_flax_available():
        from .modeling_flax_clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
