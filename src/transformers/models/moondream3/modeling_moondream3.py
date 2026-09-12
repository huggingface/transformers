# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from dataclasses import dataclass
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.utils import logging, TransformersKwargs
from .configuration_moondream3 import (
    Moondream3Config,
    Moondream3TextConfig,
    Moondream3VisionConfig,
    Moondream3RegionConfig,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Moondream3Config"


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rot_dim: int = 32,
):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine frequencies [batch, seq_len, rot_dim]
        sin: Sine frequencies [batch, seq_len, rot_dim]
        rot_dim: Number of dimensions to apply rotation to (default: 32)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """

    def apply_rope(x):
        dtype = x.dtype
        x = x.to(torch.float64)
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

        xq_out_r = xq_r * cos - xq_i * sin
        xq_out_i = xq_r * sin + xq_i * cos

        xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)

        return torch.cat([xq_out, x_pass], dim=-1)

    return apply_rope(q), apply_rope(k)


class Moondream3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Moondream3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Moondream3Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        """
        base = config.rope_parameters["rope_theta"]
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim //= 2

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
        )
        if device is not None:
            inv_freq = inv_freq.to(device=device)
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .to(torch.float32)
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].to(torch.float32)

        freqs = (
            inv_freq_expanded.to(torch.float32)
            @ position_ids_expanded.to(torch.float32)
        ).transpose(1, 2)
        cfreqs = (
            torch.exp(1j * freqs)
            .unsqueeze(1)
            .expand(-1, self.config.num_attention_heads, -1, -1)
        )

        return cfreqs.real, cfreqs.imag


class Moondream3Attention(nn.Module):
    def __init__(
        self,
        config: Moondream3TextConfig | Moondream3VisionConfig,
        layer_idx: Optional[int] = None,
        use_tau: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        attention_bias = config.attention_bias
        self.attention_dropout = config.attention_dropout

        if isinstance(config, Moondream3TextConfig):
            self.is_causal = True
        elif isinstance(config, Moondream3VisionConfig):
            self.is_causal = False
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_tau = use_tau

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )

        if self.use_tau:
            # In original, tau weights are (n_heads, qkv_dim) where qkv_dim is the combined QKV dimension
            qkv_dim = (
                self.num_heads * self.head_dim
                + 2 * self.num_key_value_heads * self.head_dim
            )
            self.tau_wq = nn.Linear(qkv_dim, self.num_heads, bias=False)
            self.tau_wv = nn.Linear(qkv_dim, self.num_heads, bias=False)
            self.tau_alpha = nn.Parameter(torch.empty(self.num_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        input_shape = hidden_states.shape[:-1]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.use_tau:
            qkv_out = torch.cat([query_states, key_states, value_states], dim=-1)
            tok_feat = F.gelu(qkv_out)
            tok_q = torch.tanh(self.tau_wq(tok_feat)).permute(0, 2, 1)
            tok_v = torch.tanh(self.tau_wv(tok_feat)).permute(0, 2, 1)

            pos = position_ids.to(tok_q.dtype) + 1
            alpha = self.tau_alpha.to(tok_q.dtype)
            tau_pos = 1 + (
                torch.sigmoid(alpha[None, :, None] * pos[:, None, :].log()) - 0.5
            )
            tau_q = (tok_q + tau_pos).unsqueeze(-1)
            tau_v = (tok_v + tau_pos).unsqueeze(-1)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.use_tau:
            query_states = query_states * tau_q

            if self.num_key_value_groups > 1:
                tau_v_repeated = tau_v.repeat(1, self.num_key_value_groups, 1, 1)[
                    :, : self.num_key_value_heads, :, :
                ]
            else:
                tau_v_repeated = tau_v
            value_states = value_states * tau_v_repeated

        cos, sin = None, None
        if position_embeddings is not None:
            cos, sin = position_embeddings

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        query_states, key_states = (
            query_states.to(value_states.dtype),
            key_states.to(value_states.dtype),
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Moondream3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu_pytorch_tanh",
        out_size: int | None = None,
        gated: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_size = self.hidden_size if out_size is None else out_size
        self.hidden_act = hidden_act
        self.gated = gated
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_size, bias=bias)
        self.gate_proj = None
        if self.gated:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=bias
            )
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x) -> torch.Tensor:
        if self.gated:
            h = self.up_proj(x)
            g = self.gate_proj(x)
            x = self.act_fn(h) * (g + 1)
        else:
            x = self.act_fn(self.up_proj(x))
        return self.down_proj(x)


class Moondream3SparseMoeBlock(nn.Module):
    def __init__(self, config: Moondream3TextConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=True)
        self.experts = nn.ModuleList(
            [
                Moondream3MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.moe_intermediate_size,
                    # hidden_act=self.config.moe_hidden_act,
                    gated=True,
                    bias=False,
                    hidden_act="gelu",
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, cache_position=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits: torch.Tensor = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = (selected_experts == expert_idx).nonzero(as_tuple=True)

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class Moondream3DecoderLayer(nn.Module):
    def __init__(self, config: Moondream3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.self_attn = Moondream3Attention(config, layer_idx, use_tau=True)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_moe_layer = layer_idx >= config.moe_start_layer
        if self.is_moe_layer:
            self.mlp = Moondream3SparseMoeBlock(config, layer_idx=layer_idx)
        else:
            self.mlp = Moondream3MLP(
                self.hidden_size,
                self.intermediate_size,
                # hidden_act=self.config.hidden_act,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        hidden_states_ln = self.input_layernorm(hidden_states)

        hidden_states_attn, self_attn_weights = self.self_attn(
            hidden_states=hidden_states_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.is_moe_layer:
            hidden_states_mlp, router_logits = self.mlp(
                hidden_states_ln, cache_position=cache_position
            )
        else:
            hidden_states_mlp = self.mlp(hidden_states_ln)
            router_logits = None

        # Add both attention and MLP to residual like original
        hidden_states = hidden_states + hidden_states_attn + hidden_states_mlp

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Moondream3PreTrainedModel(PreTrainedModel):
    config_class = Moondream3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Moondream3DecoderLayer", "Moondream3SparseMoeBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Moondream3DecoderLayer,
        "attentions": Moondream3Attention,
    }

class Moondream3TextModel(Moondream3PreTrainedModel):
    config_class = Moondream3TextConfig

    def __init__(self, config: Moondream3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id if hasattr(config, "pad_token_id") else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Moondream3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Moondream3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size = hidden_states.shape[0]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Moondream3VisionPatchEmbeddings(nn.Module):
    def __init__(self, config: Moondream3VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.crop_size = config.crop_size
        self.patch_size = config.patch_size
        self.grid_size = self.crop_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.projection = nn.Linear(
            self.patch_size * self.patch_size * self.num_channels,
            self.hidden_size,
            bias=True,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, config.hidden_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        P1 = P2 = self.patch_size

        x = pixel_values.reshape(B, C, H // P1, P1, W // P2, P2)

        x = x.permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(B, (H // P1) * (W // P2), C * P1 * P2)

        x = self.projection(x)
        return x + self.position_embeddings


class Moondream3VisionEncoderLayer(nn.Module):
    def __init__(self, config: Moondream3VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx

        self.self_attn = Moondream3Attention(
            config, layer_idx=self.layer_idx, use_tau=False
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = Moondream3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            # hidden_act=self.config.hidden_act,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Moondream3VisionModel(Moondream3PreTrainedModel):
    config_class = Moondream3VisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Moondream3VisionEncoderLayer"]

    def __init__(self, config: Moondream3VisionConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.proj_inner_dim = self.config.proj_inner_dim
        self.proj_out_dim = self.config.proj_out_dim

        self.embeddings = Moondream3VisionPatchEmbeddings(config)
        self.layers = nn.ModuleList(
            [
                Moondream3VisionEncoderLayer(config, layer_idx)
                for layer_idx in range(self.num_hidden_layers)
            ]
        )
        self.post_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.vision_projection = Moondream3MLP(
            hidden_size=self.hidden_size * 2,
            intermediate_size=self.proj_inner_dim,
            out_size=self.proj_out_dim,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def _reconstruct_from_crops(
        self,
        crops: torch.Tensor,
        tiling: tuple[int, int],
        overlap_margin: int = 4,
        patch_size: int = 14,
    ) -> torch.Tensor:
        """
        Reconstruct the original image from overlapping crops into a single seamless image.

        Takes a list of overlapping image crops along with their positional metadata and
        reconstructs them into a single coherent image by carefully stitching together
        non-overlapping regions. Handles both numpy arrays and PyTorch tensors.

        Args:
            crops: List of image crops as numpy arrays or PyTorch tensors with shape
                (H,W,C)
            tiling: Tuple of (height,width) indicating crop grid layout
            patch_size: Size in pixels of each patch, default 14
            overlap_margin: Number of overlapping patches on each edge, default 4

        Returns:
            Reconstructed image as numpy array or PyTorch tensor matching input type,
            with shape (H,W,C) where H,W are the original image dimensions
        """
        if isinstance(tiling, torch.Tensor):
            tiling_h, tiling_w = tiling[0].item(), tiling[1].item()
        else:
            tiling_h, tiling_w = tiling
        tiling_h, tiling_w = int(tiling_h), int(tiling_w)
        crop_height, crop_width = crops[0].shape[:2]
        margin_pixels = overlap_margin * patch_size

        output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
        output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels
        reconstructed = torch.zeros(
            (output_h, output_w, crops[0].shape[2]),
            device=crops[0].device,
            dtype=crops[0].dtype,
        )

        for i, crop in enumerate(crops):
            tile_y = i // tiling_w
            tile_x = i % tiling_w

            x_start = 0 if tile_x == 0 else margin_pixels
            x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
            y_start = 0 if tile_y == 0 else margin_pixels
            y_end = (
                crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels
            )

            out_x = tile_x * (crop_width - 2 * margin_pixels)
            out_y = tile_y * (crop_height - 2 * margin_pixels)

            reconstructed[
                out_y + y_start : out_y + y_end, out_x + x_start : out_x + x_end
            ] = crop[y_start:y_end, x_start:x_end]

        return reconstructed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        tiling: Tuple[int, int],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, num_crops = pixel_values.shape[:2]
        # flatten batch_size and num_crops into same dim
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        hidden_states: torch.Tensor = self.embeddings(pixel_values)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__, hidden_states
                )
            else:
                layer_outputs = encoder_layer(hidden_states)

            hidden_states = layer_outputs

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            batch_size, num_crops, *hidden_states.shape[1:]
        )
        outputs = []
        for b in range(batch_size):
            hs = hidden_states[b]
            t = tiling[b]

            global_features = hs[0]
            local_features = hs[1:].view(
                -1,
                self.num_hidden_layers,
                self.num_hidden_layers,
                self.hidden_size,
            )

            reconstructed = self._reconstruct_from_crops(
                local_features,
                t,
                patch_size=1,
                overlap_margin=self.config.overlap_margin,
            )

            reconstructed = reconstructed.permute(2, 0, 1)
            reconstructed = F.adaptive_avg_pool2d(
                reconstructed,
                output_size=(self.num_hidden_layers, self.num_hidden_layers),
            )
            reconstructed = reconstructed.permute(1, 2, 0).view(
                self.num_hidden_layers * self.num_hidden_layers, self.hidden_size
            )
            final_features = torch.cat([global_features, reconstructed], dim=-1)
            outputs.append(final_features)
        output = torch.stack(outputs, 0)

        hidden_states = self.vision_projection(output)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Moondream3RegionEncoder(nn.Module):
    def __init__(self, config: Moondream3RegionConfig):
        super().__init__()
        self.coord_encoder = nn.Linear(config.coord_feat_dim, config.hidden_size)
        self.size_encoder = nn.Linear(config.size_feat_dim, config.hidden_size)

        coord_freq = torch.randn(config.coord_feat_dim // 2, 1) * 10.0
        size_freq = torch.randn(config.size_feat_dim // 2, 2) * 10.0
        self.register_buffer("coord_freq", coord_freq.T)
        self.register_buffer("size_freq", size_freq.T)

    def fourier_features(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * torch.pi * x @ w
        return torch.cat([x_proj.cos(), x_proj.sin()], dim=-1)

    def encode_coordinate(self, coord: torch.Tensor) -> torch.Tensor:
        fourier_features = self.fourier_features(coord, self.coord_freq)
        return self.coord_encoder(fourier_features)

    def encode_size(self, size: torch.Tensor) -> torch.Tensor:
        fourier_features = self.fourier_features(size, self.size_freq)
        return self.size_encoder(fourier_features)


class Moondream3RegionDecoder(nn.Module):
    def __init__(self, config: Moondream3RegionConfig):
        super().__init__()
        self.coord_decoder = nn.Linear(config.hidden_size, config.coord_out_dim)
        self.size_decoder = nn.Linear(config.hidden_size, config.size_out_dim)

    def decode_coordinate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.coord_decoder(hidden_state)

    def decode_size(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.size_decoder(hidden_state).view(hidden_state.shape[0], 2, -1)


class Moondream3Model(Moondream3PreTrainedModel):
    def __init__(self, config: Moondream3Config):
        super().__init__(config)
        self.config = config
        self.text_model = Moondream3TextModel(config.text_config)
        self.vision_model = Moondream3VisionModel(config.vision_config)
        self.vocab_size = config.text_config.vocab_size

        self.region_encoder = Moondream3RegionEncoder(config.region_config)
        self.region_decoder = Moondream3RegionDecoder(config.region_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.embed_tokens = value

    def set_decoder(self, decoder):
        self.text_model = decoder

    def get_decoder(self):
        return self.text_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        tiling: Tuple[int, int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if not ((pixel_values is not None) ^ (tiling is None)):
            raise ValueError("You must specify both pixel_values and tiling")

        if inputs_embeds is not None and (
            pixel_values is not None or tiling is not None
        ):
            raise ValueError(
                "When inputs_embeds is provided, do not pass pixel_values/tiling; "
                "inputs_embeds must already include BOS+image(+text)."
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.text_model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.vision_model.embeddings.projection.weight.dtype
            )
            image_embeds = self.vision_model(pixel_values, tiling=tiling)[
                "last_hidden_state"
            ]
            prefix = self.text_model.embed_tokens(
                torch.full(
                    (input_ids.shape[0], 1),
                    # self.config.text_config.bos_token_id is None, unsure, so for now just use 0 directly.
                    0,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            )
            embeds = torch.cat([prefix, image_embeds], dim=1)
            cache_pos = torch.arange(embeds.shape[-2], device=embeds.device)
            pos = cache_pos.unsqueeze(0).expand(embeds.shape[0], -1)
            attn_mask = torch.full(
                (embeds.shape[0], 1, embeds.shape[-2], pos.shape[-1]),
                True,
                dtype=torch.bool,
                device=embeds.device,
            )

            outputs = self.text_model(
                input_ids=None,
                attention_mask=attn_mask,
                position_ids=pos,
                past_key_values=past_key_values,
                inputs_embeds=embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_pos,
            )

        attn_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        cache_position[-1] + 1 - attention_mask.shape[-1],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                    attention_mask,
                ],
                dim=-1,
            ),
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        outputs = self.text_model(
            input_ids=None,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    outputs.last_hidden_state,
                    getattr(outputs, "past_key_values", None),
                    getattr(outputs, "hidden_states", None),
                    getattr(outputs, "attentions", None),
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


@dataclass
class Moondream3GenerateOutput(GenerateDecoderOnlyOutput):
    objects: Optional[list[dict[str, float]]] = None


class Moondream3ForConditionalGeneration(Moondream3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Moondream3Config):
        super().__init__(config)
        self.objects = None
        self.model = Moondream3Model(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=True
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.text_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.text_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.text_model = decoder

    def get_decoder(self):
        return self.model.text_model

    def _prepare_generated_length(
        self,
        generation_config,
        **kwargs,
    ):
        generation_config = super()._prepare_generated_length(
            generation_config, **kwargs
        )
        generation_config.max_length += self.config.vision_config.prefix_len
        return generation_config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        tiling: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if pixel_values is not None and inputs_embeds is None:
            position_ids += self.config.vision_config.prefix_len
            cache_position += self.config.vision_config.prefix_len

        model_outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            tiling=tiling,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )
        hidden_states = model_outputs.last_hidden_state

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hs = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, slice):
            hs = hidden_states[:, logits_to_keep, :]
        else:
            hs = hidden_states

        hs = self.model.text_model.norm(hs)
        logits = self.lm_head(hs)

        pred = torch.argmax(logits, dim=-1)
        print(pred)

        pos_ids = position_ids[:, -1:] + 1
        cache_pos = cache_position[-1:] + 1
        mask = torch.ones(
            hidden_states.shape[0], 1, device=self.device, dtype=torch.long
        )
        is_processing_point = torch.any(pred == 5)
        while is_processing_point:
            batch_mask = pred[:, -1] == 5
            hidden_states = hidden_states[:, -1:, :]
            x_logits = self.model.region_decoder.decode_coordinate(hidden_states)
            x_center = torch.argmax(x_logits, dim=-1) / x_logits.size(-1)
            next_embeds = self.model.region_encoder.encode_coordinate(
                x_center.to(x_logits.dtype)
            ).unsqueeze(1)
            model_outputs = self.model(
                input_ids=None,
                pixel_values=None,
                tiling=None,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_pos,
                logits_to_keep=logits_to_keep,
            )
            hidden_states = model_outputs.last_hidden_state
            y_logits = self.model.region_decoder.decode_coordinate(hidden_states)
            y_center = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
            next_embeds = self.model.region_encoder.encode_coordinate(
                y_center.to(y_logits.dtype)
            ).unsqueeze(1)
            coords = torch.cat([x_center, y_center], dim=1)
            coords = coords * (batch_mask).unsqueeze(1)
            pos_ids += 1
            cache_pos = cache_pos + 1
            bbox = None
            if input_ids.shape[-1] > 1 and input_ids[0, 1] == 7235:
                model_outputs = self.model(
                    input_ids=None,
                    pixel_values=None,
                    tiling=None,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=next_embeds,
                    labels=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    cache_position=cache_pos,
                    logits_to_keep=logits_to_keep,
                )
                hidden_states = model_outputs.last_hidden_state
                size_logits = self.model.region_decoder.decode_size(hidden_states)
                bins = torch.argmax(size_logits, dim=-1)
                w_bin = bins[:, 0]
                h_bin = bins[:, 1]

                w = torch.pow(2.0, (w_bin.float() / 1023.0) * 10.0 - 10.0)
                h = torch.pow(2.0, (h_bin.float() / 1023.0) * 10.0 - 10.0)

                next_embeds = (
                    self.model.region_encoder.encode_size(
                        torch.stack([w, h], dim=-1).to(size_logits.dtype)
                    )
                ).unsqueeze(1)
                bbox = [
                    x_center.item() - w.item() / 2,
                    y_center.item() - h.item() / 2,
                    x_center.item() + w.item() / 2,
                    y_center.item() + h.item() / 2,
                ]
                bbox = bbox * (batch_mask).unsqueeze(1)
                pos_ids += 1
                cache_pos = cache_pos + 1

            new = coords.unsqueeze(1) if bbox is None else bbox.unsqueeze(1)
            if self.objects is None:
                self.objects = new
            else:
                self.objects = torch.cat([self.objects, new], dim=1)
            model_outputs = self.model(
                input_ids=None,
                pixel_values=None,
                tiling=None,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_pos,
                logits_to_keep=logits_to_keep,
            )
            pos_ids += 1
            cache_pos = cache_pos + 1
            hidden_states = model_outputs.last_hidden_state

            indices = torch.tensor(
                [
                    self.config.text_config.coord_token_id,
                    0, # self.config.text_config.eos_token_id,
                ],
                device=self.device,
            )

            hidden_states = self.model.text_model.norm(hidden_states)
            logits = (
                hidden_states @ self.lm_head.weight[indices].T
                + self.lm_head.bias[indices]
            )

            logits_full = torch.full(
                (logits.shape[0], logits.shape[1], self.config.text_config.vocab_size),
                float("-inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            logits_full[:, :, torch.tensor([5, 0])] = logits
            logits = logits_full
            pred[batch_mask] = torch.argmax(logits, dim=-1)[batch_mask]
            print(pred)
            is_processing_point = torch.any(pred == 5)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(model_outputs, "past_key_values", None),
            hidden_states=getattr(model_outputs, "hidden_states", None),
            attentions=getattr(model_outputs, "attentions", None),
        )

    def generate(self, **kwargs) -> Union[Moondream3GenerateOutput, torch.LongTensor]:
        outputs = super().generate(**kwargs)
        if self.objects is not None and len(self.objects) > 0:
            if isinstance(outputs, torch.Tensor):
                outputs = self.objects
                self.objects = []
            else:
                outputs = Moondream3GenerateOutput(**outputs, objects=self.objects)
                self.objects = []
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs["position_ids"] += (
            model_inputs["cache_position"].unsqueeze(0) - model_inputs["position_ids"]
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder,
        num_new_tokens: int = 1,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        if model_kwargs["use_cache"] == True:
            model_kwargs["pixel_values"] = None
            model_kwargs["tiling"] = None
        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


__all__ = [
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "Moondream3RegionConfig",
    "Moondream3PreTrainedModel",
    "Moondream3Model",
    "Moondream3TextModel",
    "Moondream3VisionModel",
    "Moondream3ForConditionalGeneration",
]
