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
    from .configuration_qwen2_5_omni import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniTalkerConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniToken2WavConfig,
    )
    from .modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniPreTrainedModel,
        Qwen2_5OmniPreTrainedModelForConditionalGeneration,
        Qwen2_5OmniTalkerForConditionalGeneration,
        Qwen2_5OmniTalkerModel,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniThinkerTextModel,
        Qwen2_5OmniToken2WavBigVGANModel,
        Qwen2_5OmniToken2WavDiTModel,
        Qwen2_5OmniToken2WavModel,
    )
    from .processing_qwen2_5_omni import Qwen2_5OmniProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
