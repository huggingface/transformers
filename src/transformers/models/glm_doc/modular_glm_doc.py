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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vTextAttention,
    Glm4vVisionModel,
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
    _keys_to_ignore_on_load_unexpected = [r"model\.language_model.\.layers\.16.*"]


class GlmDocModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmDocVisionModel(Glm4vVisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        del self.embeddings
        del self.post_conv_layernorm

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        hidden_states = self.merger(hidden_states)
        return hidden_states


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
