# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
# Depth-Anything-V2-Small model is under the Apache-2.0 license.
# Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

"""Convert Depth Anything V2 checkpoints from the original repository. URL:
https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2"""

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_torch_available
from ...utils import OptionalDependencyNotAvailable


_import_structure = {"configuration_depth_anything_v2": ["DepthAnythingV2Config"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_depth_anything_v2"] = [
        "DepthAnythingV2ForDepthEstimation",
        "DepthAnythingV2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_depth_anything_v2 import DepthAnythingV2Config
    from .modeling_depth_anything_v2 import DepthAnythingV2

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_depth_anything_v2 import (
            DepthAnythingV2ForDepthEstimation,
            DepthAnythingV2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    
