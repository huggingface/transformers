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
    is_torch_available,
)


_import_structure = {
    "configuration_sam2": [
        "Sam2Config",
        "Sam2ImageEncoderConfig",
        "Sam2MemoryAttentionConfig",
        "Sam2MemoryEncoderConfig",
    ],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    pass
    _import_structure["modeling_sam2"] = [
        "Sam2ImagePredictor",
        "Sam2Model",
        "Sam2PreTrainedModel",
        "Sam2VideoPredictor",
    ]

if TYPE_CHECKING:
    from .configuration_sam2 import Sam2Config, Sam2MaskDecoderConfig, Sam2VisionConfig

    # from .processing_sam import SamProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_sam2 import (
            Sam2ImageEncoder,
            Sam2Model,
            Sam2PreTrainedModel,
            Sam2VideoPredictor,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
