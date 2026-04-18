# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3TTS model."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ..qwen2.modeling_qwen2 import eager_attention_forward, rotate_half
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    ECAPA_TimeDelayNet,
)
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP, Qwen3RMSNorm
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerTextMLP,
)
from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .generation_qwen3_tts import Qwen3TTSGenerationMixin


logger = logging.get_logger(__name__)


# ─── Aliases for Qwen3 components ─────────────────────────────────────────────


class Qwen3TTSRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3TTSMlp(Qwen3MLP):
    pass


class Qwen3TTSTalkerTextMLP(Qwen3OmniMoeTalkerTextMLP):
    pass


# ─── 1D Rotary Embedding (Code Predictor) ────────────────────────────────────


class Qwen3TTSRotaryEmbedding(Qwen3OmniMoeRotaryEmbedding):
    pass


# ─── 3D Rotary Embedding (Talker) ─────────────────────────────────────────────


class Qwen3TTSTalkerRotaryEmbedding(Qwen3OmniMoeRotaryEmbedding):
    """3D multimodal rotary embedding (temporal / height / width) for Talker.

    position_ids expected shape: (3, batch, seq).
    """

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # position_ids: (3, batch, seq) for temporal, height, width
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ─── Multimodal RoPE helpers ──────────────────────────────────────────────────


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
    """Apply 3D RoPE to q and k tensors."""
    if mrope_interleaved:

        def _apply_interleaved(x, n_mod):
            x_out = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                x_out[..., i : n * n_mod : n_mod] = x[i, ..., i : n * n_mod : n_mod]
            return x_out

        dim = cos.shape[-1]
        n_mod = len(mrope_section)
        cos = torch.cat([_apply_interleaved(cos[..., : dim // 2], n_mod)] * 2, dim=-1).unsqueeze(unsqueeze_dim)
        sin = torch.cat([_apply_interleaved(sin[..., : dim // 2], n_mod)] * 2, dim=-1).unsqueeze(unsqueeze_dim)
    else:
        mrope_section_2x = mrope_section * 2
        cos = torch.cat(
            [chunk[i % 3] for i, chunk in enumerate(cos.split(mrope_section_2x, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [chunk[i % 3] for i, chunk in enumerate(sin.split(mrope_section_2x, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─── Attention layers ─────────────────────────────────────────────────────────


class Qwen3TTSTalkerAttention(nn.Module):
    """Talker attention with 3D multimodal RoPE."""

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = getattr(config, "sliding_window", None)

        rope_params = config.rope_parameters if config.rope_parameters is not None else {}
        # mrope_section describes half-dimension splits (will be repeated *2 in apply function)
        half_dim = self.head_dim // 2
        self.mrope_section = rope_params.get(
            "mrope_section",
            [half_dim // 3, half_dim // 3, half_dim - 2 * (half_dim // 3)],
        )
        self.mrope_interleaved = rope_params.get("interleaved", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section, self.mrope_interleaved
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3TTSCodePredictorAttention(Qwen3Attention):
    """Code Predictor attention — inherited from Qwen3Attention (1D RoPE)."""

    pass


# ─── Decoder layers ──────────────────────────────────────────────────────────


class Qwen3TTSTalkerDecoderLayer(GradientCheckpointingLayer):
    """Talker decoder layer."""

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTalkerAttention(config, layer_idx)
        self.mlp = Qwen3TTSTalkerTextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class Qwen3TTSDecoderLayer(GradientCheckpointingLayer):
    """Code Predictor decoder layer."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSCodePredictorAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3TTSMlp(config)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ─── Speaker Encoder (ECAPA-TDNN) ─────────────────────────────────────────────
# Components (TimeDelayNetBlock, SqueezeExcitationRes2NetBlock,
# AttentiveStatisticsPooling) imported from qwen2_5_omni.


class Qwen3TTSSpeakerEncoder(ECAPA_TimeDelayNet):
    """ECAPA-TDNN speaker encoder (inherited from qwen2_5_omni)."""

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__(config)


# ─── Utility MLP ──────────────────────────────────────────────────────────────


class Qwen3TTSTalkerResizeMLP(nn.Module):
    """2-layer MLP for text projection."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias: bool = False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


# ─── Output dataclasses ──────────────────────────────────────────────────────


@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    generation_steps: int | None = None


@dataclass
class Qwen3TTSTalkerOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    past_hidden: torch.FloatTensor | None = None
    generation_step: int | None = None
    trailing_text_hidden: torch.FloatTensor | None = None
    tts_pad_embed: torch.FloatTensor | None = None


# ─── PreTrainedModel ──────────────────────────────────────────────────────────


class Qwen3TTSBasePreTrainedModel(PreTrainedModel):
    """Common base for all Qwen3TTS PreTrainedModel classes."""

    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True


class Qwen3TTSPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSConfig
    _no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]
    _supports_cache_class = True
    _supports_static_cache = False


class Qwen3TTSTalkerTextPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    """PreTrainedModel for Talker-related models."""

    _no_split_modules = []
    _supports_cache_class = True
    _supports_static_cache = False


# ─── Talker Model (text-to-acoustic) ──────────────────────────────────────────


class Qwen3TTSTalkerModel(Qwen3TTSTalkerTextPreTrainedModel):
    """Talker model: text encoder with dual codec+text embeddings."""

    config_class = Qwen3TTSTalkerConfig
    base_model_prefix = "talker.model"

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTalkerRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def set_input_embeddings(self, value):
        self.codec_embedding = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ─── Code Predictor Model ────────────────────────────────────────────────────


class Qwen3TTSTalkerCodePredictorModel(Qwen3TTSPreTrainedModel):
    """Code predictor model: sequential multi-codebook refinement."""

    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, embedding_dim: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    def set_input_embeddings(self, value):
        self.codec_embedding = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        generation_steps: int | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ─── For Conditional Generation Classes ──────────────────────────────────────


class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    """Wrapper for CodePredictorModel with generation capabilities."""

    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        generation_steps: int | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prefill stage: derive generation_steps from sequence length
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        # Generation stage: look up step-specific embedding
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    def forward_finetune(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        generation_steps: int | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Fine-tuning forward pass with per-codebook processing."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logits = []
        for i in range(1, self.config.num_code_groups):
            logits.append(self.lm_head[i - 1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            for codebook_idx in range(1, self.config.num_code_groups):
                loss += loss_fct(
                    logits[:, codebook_idx - 1].reshape(-1, self.config.vocab_size),
                    labels[:, codebook_idx - 1].reshape(-1),
                )
            loss = loss / (self.config.num_code_groups - 1)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(loss=loss, logits=logits)


class Qwen3TTSTalkerForConditionalGeneration(Qwen3TTSTalkerTextPreTrainedModel, GenerationMixin):
    """Main Qwen3-TTS model for text-to-acoustic generation."""

    config_class = Qwen3TTSTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TTSTalkerModel(config)
        self.vocab_size = config.vocab_size
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size, config.hidden_act, bias=True
        )

        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config=config.code_predictor_config,
            talker_config=config,
        )
        self.rope_deltas = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.codec_head

    def set_output_embeddings(self, new_embeddings):
        self.codec_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward_sub_talker_finetune(self, codec_ids, talker_hidden_states):
        """Fine-tuning forward pass for code predictor."""
        assert len(codec_ids.shape) == 2
        assert len(talker_hidden_states.shape) == 2
        assert codec_ids.shape[0] == talker_hidden_states.shape[0]
        assert talker_hidden_states.shape[1] == self.config.hidden_size
        assert codec_ids.shape[1] == self.config.num_code_groups

        sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]

        for i in range(self.config.num_code_groups - 1):
            if i == 0:
                sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))
            else:
                sub_talker_inputs_embeds.append(
                    self.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, i : i + 1])
                )
        sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)

        sub_talker_outputs = self.code_predictor.forward_finetune(
            inputs_embeds=sub_talker_inputs_embeds, labels=codec_ids[:, 1:]
        )

        sub_talker_logits = sub_talker_outputs.logits
        sub_talker_loss = sub_talker_outputs.loss
        return sub_talker_logits, sub_talker_loss

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        past_hidden: torch.FloatTensor | None = None,
        trailing_text_hidden: torch.FloatTensor | None = None,
        tts_pad_embed: torch.FloatTensor | None = None,
        generation_step: int | None = None,
        subtalker_dosample: bool | None = None,
        subtalker_top_p: float | None = None,
        subtalker_top_k: int | None = None,
        subtalker_temperature: float | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generation stage
        else:
            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
                    for i in range(self.config.num_code_groups - 1)
                ],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        loss = None
        if labels is not None:
            # Use standard loss computation for now
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Qwen3TTSTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.hidden_states, codec_ids),
            attentions=outputs.attentions,
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

    def get_rope_index(
        self,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the 3D rope index based on temporal, height and width."""
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["past_hidden"] = outputs.past_hidden
        model_kwargs["generation_step"] = outputs.generation_step
        model_kwargs["trailing_text_hidden"] = outputs.trailing_text_hidden
        model_kwargs["tts_pad_embed"] = outputs.tts_pad_embed
        return model_kwargs


class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel, Qwen3TTSGenerationMixin):
    config_class = Qwen3TTSConfig
    main_input_name = "input_ids"

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config

        # Main TTS talker model
        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)

        # Optional speaker encoder for voice cloning (only for "base" model type)
        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Optional: speech_tokenizer and generate_config loaded separately
        self.speech_tokenizer = None
        self.generate_config = None

        # Model metadata
        self.supported_speakers = (
            list(self.config.talker_config.spk_id.keys())
            if hasattr(self.config.talker_config, "spk_id") and self.config.talker_config.spk_id is not None
            else []
        )
        self.supported_languages = ["auto"]
        if (
            hasattr(self.config.talker_config, "codec_language_id")
            and self.config.talker_config.codec_language_id is not None
        ):
            for language_id in self.config.talker_config.codec_language_id.keys():
                if "dialect" not in language_id:
                    self.supported_languages.append(language_id)

        self.speaker_encoder_sample_rate = (
            self.config.speaker_encoder_config.sample_rate if hasattr(self.config, "speaker_encoder_config") else 24000
        )
        self.tokenizer_type = getattr(self.config, "tokenizer_type", "qwen2")
        self.tts_model_size = getattr(self.config, "tts_model_size", "base")
        self.tts_model_type = getattr(self.config, "tts_model_type", "base")

        self.post_init()

    def load_speech_tokenizer(self, speech_tokenizer):
        """Load the speech tokenizer for audio encoding/decoding."""
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(self, generate_config):
        """Load the generation configuration."""
        if isinstance(generate_config, str):
            import json

            with open(generate_config, encoding="utf-8") as f:
                generate_config = json.load(f)
        self.generate_config = generate_config

    def get_supported_speakers(self):
        """Get list of supported speakers."""
        return list(self.supported_speakers)

    def get_supported_languages(self):
        """Get list of supported languages."""
        return self.supported_languages


__all__ = [
    "Qwen3TTSBasePreTrainedModel",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerTextPreTrainedModel",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSForConditionalGeneration",
]
