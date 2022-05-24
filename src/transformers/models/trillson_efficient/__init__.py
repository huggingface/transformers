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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_speech_available,
    is_torch_available,
)


_import_structure = {
    "configuration_trillson_efficient": [
        "TRILLSON_EFFICIENT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Trillson_efficientConfig",
    ],
}

try:
    if not is_speech_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_trillson_efficient"] = ["Speech2TextFeatureExtractor"]

    if ():
        _import_structure["processing_trillson_efficient"] = ["Speech2TextProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_trillson_efficient"] = [
        "TRILLSON_EFFICIENT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Trillson_efficientForConditionalGeneration",
        "Trillson_efficientModel",
        "Trillson_efficientPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_trillson_efficient import TRILLSON_EFFICIENT_PRETRAINED_CONFIG_ARCHIVE_MAP, Trillson_efficientConfig

    try:
        if not is_speech_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:

        if ():

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_trillson_efficient import (
            TRILLSON_EFFICIENT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Trillson_efficientForConditionalGeneration,
            Trillson_efficientModel,
            Trillson_efficientPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
