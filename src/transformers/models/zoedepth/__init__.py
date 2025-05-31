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


_import_structure = {
    "configuration_zoedepth": ["ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP", "ZoeDepthConfig"],
    "image_processing_zoedepth": ["ZoeDepthImageProcessor"],
    "image_processing_zoedepth_fast": ["ZoeDepthImageProcessorFast"],
    "modeling_zoedepth": [
        "ZoeDepthForDepthEstimation",
        "ZoeDepthPreTrainedModel",
    ],
}

if TYPE_CHECKING:
    from .configuration_zoedepth import ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP, ZoeDepthConfig
    from .image_processing_zoedepth import ZoeDepthImageProcessor
    from .image_processing_zoedepth_fast import ZoeDepthImageProcessorFast
    from .modeling_zoedepth import (
        ZoeDepthForDepthEstimation,
        ZoeDepthForDepthEstimation,
        ZoeDepthPreTrainedModel,
    )
else:
    # Attempt direct import for runtime availability before lazy loading
    from .image_processing_zoedepth_fast import ZoeDepthImageProcessorFast

    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
