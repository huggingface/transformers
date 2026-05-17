# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
    from .configuration_ernie4_5_vl_moe import (
        Ernie4_5_VL_MoeConfig,
        Ernie4_5_VL_MoeTextConfig,
        Ernie4_5_VL_MoeVisionConfig,
        Ernie4_5_VLMoeConfig,
        Ernie4_5_VLMoeTextConfig,
        Ernie4_5_VLMoeVisionConfig,
    )
    from .image_processing_ernie4_5_vl_moe import Ernie4_5_VL_MoeImageProcessor, Ernie4_5_VLMoeImageProcessor
    from .image_processing_pil_ernie4_5_vl_moe import Ernie4_5_VL_MoeImageProcessorPil, Ernie4_5_VLMoeImageProcessorPil
    from .modeling_ernie4_5_vl_moe import (
        Ernie4_5_VL_MoeForConditionalGeneration,
        Ernie4_5_VL_MoeModel,
        Ernie4_5_VL_MoePreTrainedModel,
        Ernie4_5_VL_MoeTextModel,
        Ernie4_5_VL_MoeVariableResolutionResamplerModel,
        Ernie4_5_VL_MoeVisionTransformerPretrainedModel,
        Ernie4_5_VLMoeForConditionalGeneration,
        Ernie4_5_VLMoeModel,
        Ernie4_5_VLMoePreTrainedModel,
        Ernie4_5_VLMoeTextModel,
        Ernie4_5_VLMoeVariableResolutionResamplerModel,
        Ernie4_5_VLMoeVisionTransformerPretrainedModel,
    )
    from .processing_ernie4_5_vl_moe import Ernie4_5_VLMoeProcessor
    from .video_processing_ernie4_5_vl_moe import Ernie4_5_VLMoeVideoProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
