# Copyright 2025 The HuggingFace Team. All rights reserved.
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
    from .configuration_phi4_multimodal import (
        Phi4MultimodalAudioConfig,
        Phi4MultimodalConfig,
        Phi4MultimodalVisionConfig,
    )
    from .feature_extraction_phi4_multimodal import Phi4MultimodalFeatureExtractor
    from .image_processing_phi4_multimodal import Phi4MultimodalImageProcessor
    from .modeling_phi4_multimodal import (
        Phi4MultimodalAudioModel,
        Phi4MultimodalAudioPreTrainedModel,
        Phi4MultimodalForCausalLM,
        Phi4MultimodalModel,
        Phi4MultimodalPreTrainedModel,
        Phi4MultimodalVisionModel,
        Phi4MultimodalVisionPreTrainedModel,
    )
    from .processing_phi4_multimodal import Phi4MultimodalProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
