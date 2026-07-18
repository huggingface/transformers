# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from ..lightglue.configuration_lightglue import LightGlueConfig
from ..lightglue.image_processing_lightglue import LightGlueImageProcessor, LightGlueImageProcessorKwargs
from ..lightglue.image_processing_pil_lightglue import LightGlueImageProcessorPil
from ..lightglue.modeling_lightglue import (
    LightGlueAttention,
    LightGlueForKeypointMatching,
    LightGlueKeypointMatchingOutput,
    LightGlueMatchAssignmentLayer,
    LightGlueMLP,
    LightGluePositionalEncoder,
    LightGluePreTrainedModel,
    LightGlueTokenConfidenceLayer,
    LightGlueTransformerLayer,
)


class LoMaConfig(LightGlueConfig):
    pass


class LoMaKeypointMatchingOutput(LightGlueKeypointMatchingOutput):
    pass


class LoMaPositionalEncoder(LightGluePositionalEncoder):
    pass


class LoMaAttention(LightGlueAttention):
    pass


class LoMaMLP(LightGlueMLP):
    pass


class LoMaTransformerLayer(LightGlueTransformerLayer):
    pass


class LoMaMatchAssignmentLayer(LightGlueMatchAssignmentLayer):
    pass


class LoMaTokenConfidenceLayer(LightGlueTokenConfidenceLayer):
    pass


class LoMaPreTrainedModel(LightGluePreTrainedModel):
    pass


class LoMaForKeypointMatching(LightGlueForKeypointMatching):
    pass


class LoMaImageProcessorKwargs(LightGlueImageProcessorKwargs):
    pass


class LoMaImageProcessor(LightGlueImageProcessor):
    pass


class LoMaImageProcessorPil(LightGlueImageProcessorPil):
    pass


__all__ = [
    "LoMaConfig",
    "LoMaPreTrainedModel",
    "LoMaForKeypointMatching",
    "LoMaImageProcessor",
    "LoMaImageProcessorPil",
]
