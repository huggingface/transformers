# TODO(guarin): Update license header
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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTBackbone,
    DINOv3ViTBackboneOutput,
    Dinov3ViTDropPath,
    DINOv3ViTEmbeddings,
    DINOv3ViTEncoder,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayer,
    DINOv3ViTLayerScale,
    DINOv3ViTMLP,
    DINOv3ViTModel,
    DINOv3ViTPreTrainedModel,
    DINOv3ViTRopePositionEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


# TODO (guarin): Double check if we want this checkpoint as default. Motiviation is that
# it is the smallest checkpoint which supports all tasks.
# @auto_docstring(checkpoint="facebook/sapiens2-pretrain-0.4b")
@strict
class Sapiens2Config(DINOv3ViTConfig):
    r"""
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to queries and keys before RoPE in attention layers.
    num_key_value_heads (`int`, *optional*):
        Number of key/value heads for GQA layers. Defaults to `num_attention_heads // 2`.
        Set to `None` to disable GQA and use full multi-head attention everywhere.
    first_k_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of initial transformer layers that use full multi-head attention.
        Layers at or after this index switch to GQA with `num_key_value_heads`.
    last_k_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of final transformer layers that use full multi-head attention.
        Layers before `num_hidden_layers - last_k_full_attention_layers` use GQA with `num_key_value_heads`.
    """

    model_type = "sapiens2"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    num_register_tokens: int = 8
    use_qk_norm: bool = True
    num_key_value_heads: int | None = None
    first_k_full_attention_layers: int = 8
    last_k_full_attention_layers: int = 8

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads // 2
        super().__post_init__(**kwargs)


class Sapiens2BackboneOutput(DINOv3ViTBackboneOutput):
    pass


class Sapiens2Embeddings(DINOv3ViTEmbeddings):
    pass


class Sapiens2RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    pass


class Sapiens2Attention(nn.Module):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.use_qk_norm = config.use_qk_norm

        is_full_attention = (
            config.num_key_value_heads is None
            or layer_idx < config.first_k_full_attention_layers
            or layer_idx >= config.num_hidden_layers - config.last_k_full_attention_layers
        )
        self.num_kv_heads = self.num_heads if is_full_attention else config.num_key_value_heads
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.k_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.value_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(factor, dim=1)
            value_states = value_states.repeat_interleave(factor, dim=1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Sapiens2LayerScale(DINOv3ViTLayerScale):
    pass


class Sapiens2MLP(DINOv3ViTMLP):
    pass


class Sapiens2GatedMLP(DINOv3ViTGatedMLP):
    pass


class Sapiens2DropPath(Dinov3ViTDropPath):
    pass


class Sapiens2Layer(DINOv3ViTLayer):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        self.attention = Sapiens2Attention(config, layer_idx=layer_idx)


class Sapiens2PreTrainedModel(DINOv3ViTPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if isinstance(module, nn.RMSNorm):
            init.ones_(module.weight)


class Sapiens2Encoder(DINOv3ViTEncoder):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([Sapiens2Layer(config, layer_idx=i) for i in range(config.num_hidden_layers)])


class Sapiens2Model(DINOv3ViTModel):
    pass


class Sapiens2Backbone(DINOv3ViTBackbone):
    pass


class Sapiens2ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 768, "width": 1024}
    do_resize = True
    do_rescale = False
    do_normalize = True


__all__ = [
    "Sapiens2Config",
    "Sapiens2Model",
    "Sapiens2PreTrainedModel",
    "Sapiens2Backbone",
    "Sapiens2ImageProcessor",
]
