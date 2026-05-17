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

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_chinese_clip import ChineseCLIPConfig, ChineseCLIPTextConfig, ChineseCLIPVisionConfig
    from .image_processing_chinese_clip import ChineseCLIPImageProcessor
    from .image_processing_chinese_pil_clip import ChineseCLIPImageProcessorPil
    from .modeling_chinese_clip import (
        ChineseCLIPModel,
        ChineseCLIPPreTrainedModel,
        ChineseCLIPTextModel,
        ChineseCLIPVisionModel,
    )
    from .processing_chinese_clip import ChineseCLIPProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
