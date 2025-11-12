# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig
from ..glm4v.modeling_glm4v import Glm4vModel, Glm4vPreTrainedModel, Glm4vTextModel
from ..glm4v.processing_glm4v import Glm4vProcessor


class Glm46VTextConfig(Glm4vTextConfig):
    pass


class Glm46VConfig(Glm4vConfig):
    pass


class Glm46VPreTrainedModel(Glm4vPreTrainedModel):
    pass


class Glm46VTextModel(Glm4vTextModel):
    pass


class Glm46VModel(Glm4vModel):
    pass


class Glm46VProcessor(Glm4vProcessor):
    def replace_frame_token_id(self, timestamp_sec):
        return f"<|begin_of_image|>{self.image_token}<|end_of_image|>{timestamp_sec:.1f} seconds"


__all__ = [
    "Glm46VConfig",
    "Glm46VTextConfig",
    "Glm46VModel",
    "Glm46VPreTrainedModel",
    "Glm46VTextModel",
    "Glm46VProcessor",
]
