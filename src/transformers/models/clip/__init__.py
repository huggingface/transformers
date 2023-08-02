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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPConfig",
        "CLIPOnnxConfig",
        "CLIPTextConfig",
        "CLIPVisionConfig",
    ],
    "processing_clip": ["CLIPProcessor"],
    "tokenization_clip": ["CLIPTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_clip_fast"] = ["CLIPTokenizerFast"]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_clip"] = ["CLIPFeatureExtractor"]
    _import_structure["image_processing_clip"] = ["CLIPImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_clip"] = [
        "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CLIPModel",
        "CLIPPreTrainedModel",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "CLIPVisionModel",
        "CLIPVisionModelWithProjection",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_clip"] = [
        "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCLIPModel",
        "TFCLIPPreTrainedModel",
        "TFCLIPTextModel",
        "TFCLIPVisionModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_clip"] = [
        "FlaxCLIPModel",
        "FlaxCLIPPreTrainedModel",
        "FlaxCLIPTextModel",
        "FlaxCLIPTextPreTrainedModel",
        "FlaxCLIPTextModelWithProjection",
        "FlaxCLIPVisionModel",
        "FlaxCLIPVisionPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPConfig,
        CLIPOnnxConfig,
        CLIPTextConfig,
        CLIPVisionConfig,
    )
    from .processing_clip import CLIPProcessor
    from .tokenization_clip import CLIPTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_clip_fast import CLIPTokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_clip import CLIPFeatureExtractor
        from .image_processing_clip import CLIPImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_clip import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_clip import (
            TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
