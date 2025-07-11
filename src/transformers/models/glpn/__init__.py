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


import_structure = {
    "configuration_glpn": ["GLPNConfig"],
    "feature_extraction_glpn": ["GLPNFeatureExtractor"],
    "image_processing_glpn": ["GLPNImageProcessor"],
    "image_processing_glpn_fast": ["GLPNImageProcessorFast"],
    "modeling_glpn": [
        "GLPNModel",
        "GLPNForDepthEstimation",
        "GLPNPreTrainedModel",
    ],
}

if TYPE_CHECKING:
    from .configuration_glpn import GLPNConfig
    from .feature_extraction_glpn import GLPNFeatureExtractor
    from .image_processing_glpn import GLPNImageProcessor
    from .image_processing_glpn_fast import GLPNImageProcessorFast
    from .modeling_glpn import (
        GLPNForDepthEstimation,
        GLPNModel,
        GLPNPreTrainedModel,
    )
else:
    import os
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, import_structure, module_spec=__spec__)
