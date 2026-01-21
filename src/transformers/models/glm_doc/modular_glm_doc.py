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

import torch.nn as nn

from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vTextAttention,
)


class GlmDocVisionConfig(Glm4vVisionConfig):
    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="silu",
        attention_bias=True,
        num_heads=16,
        image_size=336,
        out_hidden_size=1536,
        intermediate_size=4608,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmDocTextConfig(Glm4vTextConfig):
    def __init__(
        self,
        vocab_size: int | None = 59246,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 4096,
        num_hidden_layers: int | None = 16,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmDocConfig(Glm4vConfig, nn.Module):
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=59280,
        video_token_id=59281,
        image_start_token_id=59256,
        image_end_token_id=59257,
        video_start_token_id=59258,
        video_end_token_id=59259,
        tie_word_embeddings=False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmDocTextAttention(Glm4vTextAttention, nn.Module):
    def __init__(self, config: GlmDocTextConfig, layer_idx: int | None = None):
        super().__init__()
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)


class GlmDocPreTrainedModel(Glm4vPreTrainedModel):
    pass


class GlmDocModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmDocModel(Glm4vModel):
    pass


class GlmDocForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


__all__ = [
    "GlmDocConfig",
    "GlmDocModel",
    "GlmDocPreTrainedModel",
    "GlmDocForConditionalGeneration",
]
