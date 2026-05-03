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
    from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeTalkerConfig, Qwen3OmniMoeThinkerConfig
    from .modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
        Qwen3OmniMoeCode2WavDecoderBlock,
        Qwen3OmniMoeCode2WavTransformerModel,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoePreTrainedModel,
        Qwen3OmniMoePreTrainedModelForConditionalGeneration,
        Qwen3OmniMoeTalkerCodePredictorModel,
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
        Qwen3OmniMoeTalkerForConditionalGeneration,
        Qwen3OmniMoeTalkerModel,
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextModel,
        Qwen3OmniMoeThinkerTextPreTrainedModel,
    )
    from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
