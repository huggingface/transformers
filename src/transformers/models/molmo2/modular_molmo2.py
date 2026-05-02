# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""PyTorch Molmo2 model."""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple, logging
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModelOutputWithPast,
)
from ..olmo2.modeling_olmo2 import Olmo2Attention
from ..phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3MLP,
)
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2EncoderLayer,
    Siglip2MLP,
)
from .configuration_molmo2 import Molmo2AdapterConfig, Molmo2Config, Molmo2TextConfig, Molmo2VitConfig


logger = logging.get_logger(__name__)


# Output dataclasses - same structure as LLaVA
class Molmo2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Molmo2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


# ===================== Vision Components (from Siglip2) =====================


class Molmo2VisionMLP(Siglip2MLP):
    pass


class Molmo2VisionAttention(Siglip2Attention):
    """Vision attention with GQA support."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Molmo2VisionEncoderLayer(Siglip2EncoderLayer):
    def __init__(self, config: Molmo2VitConfig):
        super().__init__(config)
        self.self_attn = Molmo2VisionAttention(config)
        self.mlp = Molmo2VisionMLP(config)


class Molmo2PoolingAttention(nn.Module):
    """Cross-attention module used for image feature pooling in the vision adapter."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        input_dim: int,
        attention_dropout: float = 0.0,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.attn_implementation = attn_implementation
        self.is_causal = False

        self.q_proj = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(input_dim, self.num_key_value_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        self.attention_dropout = attention_dropout

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        batch_size = inputs_q.shape[0]
        queries = self.q_proj(inputs_q)
        keys = self.k_proj(inputs_k)
        values = self.v_proj(inputs_v)

        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)

        attn_output, _ = attention_interface(
            self,
            queries,
            keys,
            values,
            attn_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Molmo2VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Molmo2VisionEncoderLayer`].

    Args:
        config: Molmo2VitConfig
    """

    def __init__(self, config: Molmo2VitConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Molmo2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        """Returns a list of hidden states, one per encoder layer."""
        hidden_states = inputs_embeds
        all_hidden_states = []
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask, **kwargs)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class Molmo2VisionModel(PreTrainedModel):
    config_class = Molmo2VitConfig
    _no_split_modules = ["Molmo2VisionEncoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=self.config.initializer_range)
        else:
            super()._init_weights(module)

    def __init__(self, config: Molmo2VitConfig):
        super().__init__(config)
        self.config = config
        self.image_default_input_size = config.image_default_input_size

        # positional embeddings
        self.scale = config.hidden_size**-0.5
        self.num_prefix_tokens: int = 0  # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.hidden_size),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
        )

        self.encoder = Molmo2VisionEncoder(config)

        self.post_init()

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int | None = None, **kwargs) -> list[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch

        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.encoder(x)
        return hidden_states


# ===================== Vision Backbone / Adapter =====================


class Molmo2ImageProjectorMLP(nn.Module):
    def __init__(self, config: Molmo2AdapterConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.text_hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class Molmo2VisionBackbone(nn.Module):
    def __init__(self, vit_config: Molmo2VitConfig, adapter_config: Molmo2AdapterConfig):
        super().__init__()
        self.adapter_config = adapter_config
        # `vit_config.num_hidden_layers` and `adapter_config.vit_layers` are normalized in `Molmo2Config.__post_init__`.
        self.vit_layers = list(adapter_config.vit_layers)
        self.image_vit = Molmo2VisionModel(vit_config)

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = Molmo2PoolingAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            attention_dropout=adapter_config.attention_dropout,
            attn_implementation=adapter_config._attn_implementation or "eager",
        )
        self.image_projector = Molmo2ImageProjectorMLP(adapter_config)
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        image_features = image_features.view(B, T, N, -1)
        return image_features

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features = self.encode_image(images)

        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]]
        )

        # Now [batch, num_high_res_features, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(to_pool.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)
        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        pooled_features = self.image_projector(pooled_features)
        return pooled_features.view(-1, pooled_features.shape[-1])[valid_token.flatten()]


# ===================== Text Components (from Phi3/Llama) =====================


class Molmo2RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: Molmo2TextConfig, rope_type: str | None = None):
        # Molmo2 has custom rope_type handling (not using config.rope_parameters)
        if rope_type is not None:
            self.rope_type = rope_type
        elif hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            # BC: "rope_type" was originally "type"
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Molmo2TextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_theta
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        dim = int(head_dim)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class Molmo2RMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps=eps)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=hidden_states.device.type):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Molmo2Attention(Olmo2Attention):
    """Molmo2 attention: Olmo2-style q/k RMSNorm with a fused QKV projection and renamed output projection."""

    def __init__(self, config: Molmo2TextConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )
        self.att_proj = nn.Linear(config.hidden_size, sum(self.fused_dims), bias=config.qkv_bias)
        self.attn_out = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

        self.q_norm = Molmo2RMSNorm(config.num_attention_heads * config.head_dim, eps=config.layer_norm_eps)
        self.k_norm = Molmo2RMSNorm(config.num_key_value_heads * config.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.view(q_shape)
        key_states = key_states.view(kv_shape)
        value_states = value_states.view(kv_shape)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if (self.config._attn_implementation or "eager") != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_out(attn_output)
        return attn_output, attn_weights


class Molmo2MLP(Phi3MLP):
    def __init__(self, input_dim: int, intermediate_size: int, hidden_act: str):
        nn.Module.__init__(self)
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False)
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = self.act(gate) * x
        x = self.ff_out(x)
        return x


class Molmo2DecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Molmo2TextConfig, layer_idx: int | None = None):
        GradientCheckpointingLayer.__init__(self)
        self.config = config

        self.self_attn = Molmo2Attention(config, layer_idx)
        self.attn_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.mlp = Molmo2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.ff_norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class Molmo2PostNormDecoderLayer(Molmo2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = self.attn_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ff_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class Molmo2Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(num_embeddings, features))
        self.new_embedding = nn.Parameter(torch.zeros(num_new_embeddings, features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


# ===================== PreTrainedModel =====================


class Molmo2PreTrainedModel(LlamaPreTrainedModel):
    config: Molmo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Molmo2DecoderLayer",
        "Molmo2PostNormDecoderLayer",
        "Molmo2VisionEncoderLayer",
        "Molmo2VisionAttention",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Molmo2DecoderLayer,
        "attentions": Molmo2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, Molmo2Embedding):
            init.normal_(module.embedding, mean=0.0, std=std)
            init.normal_(module.new_embedding, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, Molmo2RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, Molmo2VisionModel):
            init.normal_(module.positional_embedding, mean=0.0, std=std)
        elif isinstance(module, Molmo2RotaryEmbedding):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)


class Molmo2TextModel(Molmo2PreTrainedModel):
    config: Molmo2TextConfig
    _input_embed_layer = "wte"

    def __init__(self, config: Molmo2TextConfig):
        super().__init__(config)
        if config.additional_vocab_size is not None:
            self.wte = Molmo2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        decoder_layer = Molmo2PostNormDecoderLayer if config.norm_after else Molmo2DecoderLayer
        self.blocks = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Molmo2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.rope_scaling_layers is not None:
            self.rotary_embs = nn.ModuleDict(
                {
                    "default": Molmo2RotaryEmbedding(config, rope_type="default"),
                    "scaling": Molmo2RotaryEmbedding(config),
                }
            )
        else:
            self.rotary_emb = Molmo2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": self.rotary_embs["default"](hidden_states, position_ids),
                "scaling": self.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_block in enumerate(self.blocks[: self.config.num_hidden_layers]):
            if self.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in self.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            hidden_states = decoder_block(
                hidden_states,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings_i,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# Adapted from ...models.gemma3.modeling_gemma3
def token_type_ids_mask_function(group_ids: torch.Tensor) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    Args:
        group_ids (`torch.Tensor`):
            A tensor of shape `(bs, len)` assigning each token to a multimodal group. Tokens with the same group
            come from the same input image or video span. Text is denoted by `-1`.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        seq_length = group_ids.shape[-1]

        # Clamp indices because with static cache they can go beyond `group_ids.shape[-1]`.
        q_idx_clamped = q_idx.clamp(max=seq_length - 1)
        kv_idx_clamped = kv_idx.clamp(max=seq_length - 1)

        # Unmask if q and kv come from the same multimodal group, which is not -1 (i.e. non-text).
        q_group = group_ids[batch_idx, q_idx_clamped]
        kv_group = group_ids[batch_idx, kv_idx_clamped]
        q_group = torch.where(q_idx < seq_length, q_group, -1)
        kv_group = torch.where(kv_idx < seq_length, kv_group, -1)
        return (q_group == kv_group) & (q_group >= 0)

    return inner_mask


# Adapted from ...models.gemma3.modeling_gemma3
def create_causal_mask_mapping(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    token_type_ids: torch.Tensor | None = None,
    has_multimodal_inputs: bool = False,
    is_training: bool = False,
    is_first_iteration: bool | None = None,
    **kwargs,
) -> dict:
    """
    Create the causal mask mapping for Molmo2 forward passes. Multimodal spans use bidirectional attention within
    each contiguous image/video group.
    """
    if is_training and token_type_ids is None:
        raise ValueError("`token_type_ids` is required as a model input when training")

    mask_kwargs = {
        "config": config.get_text_config(),
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    is_first_iteration = (
        is_first_iteration
        if is_first_iteration is not None
        else (past_key_values is None or not past_key_values.is_initialized or has_multimodal_inputs)
    )
    if token_type_ids is not None and is_first_iteration:
        is_multimodal = (token_type_ids > 0).to(inputs_embeds.device)
        is_previous_multimodal = nn.functional.pad(is_multimodal, (1, 0), value=0)[:, :-1]
        new_multimodal_start = is_multimodal & ~is_previous_multimodal
        group_ids = torch.cumsum(new_multimodal_start.int(), dim=1) - 1
        group_ids = torch.where(is_multimodal, group_ids, -1)
        mask_kwargs["or_mask_function"] = token_type_ids_mask_function(group_ids)

    return create_causal_mask(**mask_kwargs)


class Molmo2Model(Molmo2PreTrainedModel):
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)
        self.language_model: Molmo2TextModel = Molmo2TextModel(config.text_config)
        self.image_col_id = config.image_col_id
        self.image_low_res_id = config.image_low_res_id
        self.vision_backbone: Molmo2VisionBackbone | None = None
        if config.vit_config is not None and config.adapter_config is not None:
            self.vision_backbone = Molmo2VisionBackbone(config.vit_config, config.adapter_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.language_model.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.language_model.wte = value

    def build_batched_images(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize inputs to flattened image/crop layout expected by the model.
        if pixel_values.dim() == 4:
            batch_size, num_crops, n_patches, pixels_per_patch = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_crops, n_patches, pixels_per_patch)
        if image_num_crops is None:
            raise ValueError("`image_num_crops` must be provided when `pixel_values` is passed.")
        if image_token_pooling.dim() == 3:
            image_token_pooling = image_token_pooling.reshape(-1, image_token_pooling.size(-1))

        # 1) Count the number of images in each example
        raw_counts = (input_ids == self.config.image_end_token_id).sum(1)  # [N]
        # Each image is represented by global view and high-res view
        # so we divide by 2 to get the number of images
        counts = raw_counts // 2
        N = counts.size(0)
        device = input_ids.device

        # Total number of images in the batch
        num_images = int(counts.sum().item())
        if image_grids is not None and image_grids.size(0) == N and num_images != image_grids.size(0):
            counts = torch.ones_like(counts)
            num_images = int(counts.sum().item())

        # Sanity check
        assert image_grids.size(0) == num_images, f"Expected {num_images} image grids, but got {image_grids.size(0)}"
        assert image_num_crops.size(0) == num_images, (
            f"Expected {num_images} image num crops, but got {image_num_crops.size(0)}"
        )

        # 1-1) Compute per-image pooled patch count from image grids
        with torch.no_grad():
            first_prod = image_grids[:, :2].prod(dim=1)  # [num_images]
            second_prod = image_grids[:, 2:].prod(dim=1)  # [num_images]
            num_pooled_patches_per_image = (first_prod + second_prod).to(image_num_crops.dtype)  # [num_images]

        # pixel_values: [n_crops, n_patches, pixels_per_patch]
        n_crops, n_patches, pixels_per_patch = pixel_values.shape

        # 2) Map each image index → example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_image = torch.arange(N, device=device).repeat_interleave(counts)  # [num_images]
        assert example_ids_for_image.numel() == num_images

        # 2-1) Compute crops_per_example by summing per-image crop counts
        crops_per_example = torch.zeros(N, dtype=image_num_crops.dtype, device=image_num_crops.device)
        crops_per_example.index_add_(0, example_ids_for_image, image_num_crops)  # [N]

        # 2-2) Per-image number of patches = (crops per image) * n_patches
        patches_per_image = image_num_crops * n_patches  # [num_images]

        # 2-3) Compute per-example per-image patch offsets
        counts_list = counts.tolist()
        index_offset_per_example_list = []
        offset_img = 0
        for c in counts_list:
            per_img_patches = patches_per_image[offset_img : offset_img + c]  # [c]
            # Offsets: [0, img0_total_patches, img0+img1_total_patches, ...]
            index_offset = [0] + per_img_patches.cumsum(0).tolist()[:-1]
            index_offset_per_example_list.append(index_offset)
            offset_img += c

        # 2-4) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N, dtype=num_pooled_patches_per_image.dtype, device=num_pooled_patches_per_image.device
        )
        num_pooled_patches_per_example.index_add_(0, example_ids_for_image, num_pooled_patches_per_image)

        # Sanity checks
        total_crops = int(crops_per_example.sum().item())
        assert total_crops == n_crops, f"Expected {total_crops} crops, but got {n_crops}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == image_token_pooling.size(0), (
            f"Expected {total_num_pooled_patches} pooled patches, but got {image_token_pooling.size(0)}"
        )

        # 3) Build images tensor filled with -1
        M = int(crops_per_example.max().item())
        images = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        # 4) Fill images with per-example slices from pixel_values
        offset_crop = 0
        for i in range(N):
            num = int(crops_per_example[i].item())
            cur = pixel_values[offset_crop : offset_crop + num]  # [num, n_patches, pixels_per_patch]
            images[i, :num] = cur
            offset_crop += num

        # Sanity check
        assert offset_crop == n_crops

        # 5) Build new_token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = image_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=image_token_pooling.dtype,
            device=image_token_pooling.device,
        )

        # 6) Fill token_pooling with per-example slices, adding per-image patch offsets
        patch_offset = 0
        img_offset = 0

        for i, c in enumerate(counts_list):
            num_patches = int(num_pooled_patches_per_example[i].item())

            # Subsequence of pooled tokens belonging to this example
            cur = image_token_pooling[patch_offset : patch_offset + num_patches].clone()  # [num_patches, dim]

            index_offset_per_example = index_offset_per_example_list[i]  # length = c
            per_img_pooled = num_pooled_patches_per_image[img_offset : img_offset + c]  # [c]

            assert len(index_offset_per_example) == per_img_pooled.numel()

            # Apply per-image offsets to the (ragged) subsequence
            offset = 0
            for j in range(c):
                index_offset = int(index_offset_per_example[j])
                n = int(per_img_pooled[j].item())
                cur_slice = cur[offset : offset + n]

                # Apply offset across all columns
                cur[offset : offset + n] = torch.where(
                    cur_slice >= 0,
                    cur_slice + index_offset,
                    cur_slice,
                )
                offset += n

            new_token_pooling[i, :num_patches] = cur

            patch_offset += num_patches
            img_offset += c

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches
        assert img_offset == num_images

        return images, new_token_pooling

    def build_batched_videos(
        self,
        input_ids: torch.LongTensor,
        pixel_values_videos: torch.Tensor,
        video_token_pooling: torch.Tensor,
        video_grids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Count the number of videos in each example
        if self.config.use_frame_special_tokens:
            end_token_id = self.config.frame_end_token_id
        else:
            end_token_id = self.config.image_end_token_id
        counts = (input_ids == end_token_id).any(dim=1).long()  # [N]
        N = counts.size(0)
        device = input_ids.device

        # Total number of videos in the batch
        num_videos = int(counts.sum().item())

        # Sanity check
        assert video_grids.size(0) == num_videos, f"Expected {num_videos} videos, but got {video_grids.size(0)}"

        video_num_frames = video_grids[:, 0]  # [num_videos]
        num_pooled_patches_per_video = video_grids.prod(dim=1)  # [num_videos]

        # pixel_values_videos: [n_frames, n_patches, pixels_per_patch]
        n_frames, n_patches, pixels_per_patch = pixel_values_videos.shape

        # 2) Map each video index -> example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_video = torch.arange(N, device=device).repeat_interleave(counts)  # [num_videos]
        assert example_ids_for_video.numel() == num_videos

        # 2-1) Compute frames_per_example by summing per-video frame counts
        frames_per_example = torch.zeros(
            N,
            dtype=video_num_frames.dtype,
            device=device,
        )
        frames_per_example.index_add_(0, example_ids_for_video, video_num_frames)  # [N]

        # 2-2) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N,
            dtype=num_pooled_patches_per_video.dtype,
            device=num_pooled_patches_per_video.device,
        )
        num_pooled_patches_per_example.index_add_(
            0,
            example_ids_for_video,
            num_pooled_patches_per_video,
        )

        # Sanity checks
        total_frames = int(frames_per_example.sum().item())
        assert total_frames == n_frames, f"Expected {total_frames} frames, but got {n_frames}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == video_token_pooling.size(0), (
            f"Expected {total_num_pooled_patches} pooled patches, but got {video_token_pooling.size(0)}"
        )

        # 3) Build videos tensor filled with -1
        M = int(frames_per_example.max().item())
        videos = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values_videos.dtype,
            device=device,
        )

        # 4) Fill videos with per-examples slices from pixel_values_videos
        offset_frame = 0
        for i in range(N):
            num = int(frames_per_example[i].item())
            cur = pixel_values_videos[offset_frame : offset_frame + num]  # [num, n_patches, pixels_per_patch]
            videos[i, :num] = cur
            offset_frame += num

        # Sanity check
        assert offset_frame == n_frames

        # 5) Build new token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = video_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=video_token_pooling.dtype,
            device=video_token_pooling.device,
        )

        # 6) Fill new token_pooling with per-examples slices from video_token_pooling
        patch_offset = 0
        for i in range(N):
            num_patches = int(num_pooled_patches_per_example[i].item())
            cur = video_token_pooling[patch_offset : patch_offset + num_patches]  # [num_patches, dim]
            new_token_pooling[i, :num_patches] = cur
            patch_offset += num_patches

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches

        return videos, new_token_pooling

    def merge_visual_inputs(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")
        elif pixel_values is not None:
            if input_ids is None:
                return None, None
            images, token_pooling = self.build_batched_images(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )
        elif pixel_values_videos is not None:
            if input_ids is None:
                return None, None
            images, token_pooling = self.build_batched_videos(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )
        else:
            images, token_pooling = None, None
        return images, token_pooling

    def build_input_embeddings(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor | None = None,  # image inputs
        token_pooling: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.language_model.wte(input_ids)

        image_features: torch.FloatTensor | None = None
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(x.device)
            is_image_patch = input_ids.view(-1) == self.config.image_patch_id
            assert is_image_patch.sum() == len(image_features)
            x.view(-1, x.shape[-1])[is_image_patch] += image_features

        # shape: (batch_size, seq_len, d_model)
        x = self.language_model.emb_drop(x)  # type: ignore

        return x, image_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2ModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        images, token_pooling = self.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        if images is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both images and inputs_embeds at the same time.")

        if inputs_embeds is None:
            inputs_embeds, image_features = self.build_input_embeddings(
                input_ids,
                images,
                token_pooling,
            )

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                token_type_ids,
                has_multimodal_inputs=images is not None,
                is_training=self.training,
                is_first_iteration=not use_cache,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return Molmo2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if images is not None else None,
        )


class Molmo2ForConditionalGeneration(Molmo2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.wte.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Molmo2Config

    def __init__(self, config: Molmo2Config):
        super().__init__(config)

        self.model = Molmo2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Molmo2CausalLMOutputWithPast:
        r"""
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from ... import AutoProcessor, Molmo2ForConditionalGeneration

        >>> model = Molmo2ForConditionalGeneration.from_pretrained("...")
        >>> processor = AutoProcessor.from_pretrained("...")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return Molmo2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        is_first_iteration: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

    # Adapted from ...models.gemma3.modeling_gemma3
    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and inputs_embeds.shape[1] != 1:
            is_multimodal = (token_type_ids > 0).to(inputs_embeds.device)
            is_previous_multimodal = nn.functional.pad(is_multimodal, (1, 0), value=0)[:, :-1]
            new_multimodal_start = is_multimodal & ~is_previous_multimodal
            group_ids = torch.cumsum(new_multimodal_start.int(), dim=1) - 1
            group_ids = torch.where(is_multimodal, group_ids, -1)
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(group_ids)

        return create_masks_for_generate(**mask_kwargs)


__all__ = [
    "Molmo2ForConditionalGeneration",
    "Molmo2Model",
    "Molmo2PreTrainedModel",
    "Molmo2TextModel",
    "Molmo2VisionBackbone",
    "Molmo2VisionModel",
]
