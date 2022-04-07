# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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

from ...utils import _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_flava": [
        "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FLAVACodebookConfig",
        "FLAVAConfig",
        "FLAVAImageConfig",
        "FLAVAMultimodalConfig",
        "FLAVATextConfig",
    ],
}

if is_vision_available():
    _import_structure["feature_extraction_flava"] = ["FLAVACodebookFeatureExtractor", "FLAVAFeatureExtractor"]
    _import_structure["processing_flava"] = ["FLAVAProcessor"]

if is_torch_available():
    _import_structure["modeling_flava"] = [
        "FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FLAVACodebook",
        "FLAVAForPreTraining",
        "FLAVAImageModel",
        "FLAVAModel",
        "FLAVAMultimodalModel",
        "FLAVAPreTrainedModel",
        "FLAVATextModel",
    ]

if TYPE_CHECKING:
    from .configuration_flava import (
        FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FLAVACodebookConfig,
        FLAVAConfig,
        FLAVAImageConfig,
        FLAVAMultimodalConfig,
        FLAVATextConfig,
    )

    if is_vision_available():
        from .feature_extraction_flava import FLAVACodebookFeatureExtractor, FLAVAFeatureExtractor
        from .processing_flava import FLAVAProcessor

    if is_torch_available():
        from .modeling_flava import (
            FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FLAVACodebook,
            FLAVAForPreTraining,
            FLAVAImageModel,
            FLAVAModel,
            FLAVAMultimodalModel,
            FLAVAPreTrainedModel,
            FLAVATextModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
