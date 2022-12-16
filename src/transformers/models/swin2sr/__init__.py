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

# rely on isort to merge the imports
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_swin2sr": ["SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swin2SRConfig"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_swin2sr"] = [
        "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Swin2SRForImageSuperResolution",
        "Swin2SRModel",
        "Swin2SRPreTrainedModel",
    ]


try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_swin2sr"] = ["Swin2SRImageProcessor"]


if TYPE_CHECKING:
    from .configuration_swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swin2sr import (
            SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_swin2sr import Swin2SRImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
