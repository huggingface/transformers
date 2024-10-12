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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_llava_onevision": ["LlavaOnevisionConfig"],
    "processing_llava_onevision": ["LlavaOnevisionProcessor"],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_llava_onevision"] = ["LlavaOnevisionImageProcessor"]

    _import_structure["video_processing_llava_onevision"] = ["LlavaOnevisionVideoProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_llava_onevision"] = [
        "LlavaOnevisionForConditionalGeneration",
        "LlavaOnevisionPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_llava_onevision import LlavaOnevisionConfig
    from .processing_llava_onevision import LlavaOnevisionProcessor

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_llava_onevision import LlavaOnevisionImageProcessor
        from .video_processing_llava_onevision import LlavaOnevisionVideoProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_llava_onevision import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
