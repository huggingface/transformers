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

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4VisionMlp,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vRMSNorm,
    Glm4vTextAttention,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vVisionModel,
    Glm4vVisionPatchMerger,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    is_flash_attention_requested,
)


class GlmOcrRMSNorm(Glm4vRMSNorm):
    pass


class GlmOcrVisionMlp(Glm4VisionMlp):
    def __init__(self, config, bias: bool = True):
        super().__init__()
        self.intermediate_size = config.intermediate_size


class GlmOcrVisionConfig(Glm4vVisionConfig):
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


class GlmOcrTextConfig(Glm4vTextConfig):
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


class GlmOcrConfig(Glm4vConfig, nn.Module):
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


class GlmOcrTextAttention(Glm4vTextAttention, nn.Module):
    def __init__(self, config: GlmOcrTextConfig, layer_idx: int | None = None):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)


class GlmOcrPreTrainedModel(Glm4vPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.language_model\.layers\.16.*"]


class GlmOcrModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmOcrVisionAttention(Glm4vVisionAttention):
    def __init__(self, config: GlmOcrVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = GlmOcrRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GlmOcrRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class GlmOcrVisionBlock(Glm4vVisionBlock):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = GlmOcrVisionMlp(config, bias=config.attention_bias)


class GlmOcrVisionPatchMerger(Glm4vVisionPatchMerger):
    pass


class GlmOcrVisionModel(Glm4vVisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        del self.embeddings
        del self.post_conv_layernorm
        self.merger = GlmOcrVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.out_hidden_size * config.in_channels,
            hidden_act=config.hidden_act,
        )

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


class GlmOcrModel(Glm4vModel):
    pass


class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


__all__ = [
    "GlmOcrConfig",
    "GlmOcrTextConfig",
    "GlmOcrVisionConfig",
    "GlmOcrModel",
    "GlmOcrPreTrainedModel",
    "GlmOcrForConditionalGeneration",
]
