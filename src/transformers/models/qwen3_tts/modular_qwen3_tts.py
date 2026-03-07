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

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import accumulate

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

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
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ...utils.hub import cached_file
from ..mimi.modeling_mimi import MimiLayerScale, MimiModel
from ..qwen2.modeling_qwen2 import eager_attention_forward, rotate_half
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    AMPBlock,
    DiTAttention,
    DiTCodecEmbedding,
    DiTDecoderLayer,
    DiTMLP,
    DiTTimestepEmbedding,
    ECAPA_TimeDelayNet,
    Qwen2_5_OmniAdaLayerNormZero_Final,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
    Qwen2_5OmniToken2WavModel,
    SinusoidsPositionEmbedding,
    SnakeBeta,
    TorchActivation1d,
)
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavAttention,
    Qwen3OmniMoeCode2WavDecoderResidualUnit,
    Qwen3OmniMoeCode2WavMlp,
    Qwen3OmniMoeCode2WavRMSNorm,
    Qwen3OmniMoeCode2WavTransformerLayer,
    Qwen3OmniMoeConvNeXtBlock,
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerTextMLP,
)
from .configuration_qwen3_tts import (
    MimiConfig,
    Qwen3TTSConfig,
    Qwen3TTSDiTConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV1DecoderBigVGANConfig,
    Qwen3TTSTokenizerV1DecoderConfig,
    Qwen3TTSTokenizerV1EncoderConfig,
    Qwen3TTSTokenizerV2Code2WavConfig,
    Qwen3TTSTokenizerV2Config,
)


logger = logging.get_logger(__name__)


# ─── Aliases for Qwen3 components ─────────────────────────────────────────────


class Qwen3TTSRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3TTSMlp(Qwen3MLP):
    pass


class Qwen3TTSTalkerTextMLP(Qwen3OmniMoeTalkerTextMLP):
    pass


class Qwen3TTSTokenizerV2CausalConvNet(Qwen3OmniMoeCausalConvNet):
    pass


class Qwen3TTSTokenizerV2CausalTransConvNet(Qwen3OmniMoeCausalTransConvNet):
    pass


class Qwen3TTSTokenizerV2ConvNeXtBlock(Qwen3OmniMoeConvNeXtBlock):
    pass


class Qwen3TTSTokenizerV2RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Qwen3TTSTokenizerV2Attention(Qwen3OmniMoeCode2WavAttention):
    pass


class Qwen3TTSTokenizerV2RMSNorm(Qwen3OmniMoeCode2WavRMSNorm):
    pass


class Qwen3TTSTokenizerV2Mlp(Qwen3OmniMoeCode2WavMlp):
    pass


class Qwen3TTSTokenizerV2TransformerLayer(Qwen3OmniMoeCode2WavTransformerLayer):
    pass


class Qwen3TTSTokenizerV2LayerScale(MimiLayerScale):
    pass


class Qwen3TTSTokenizerV2ResidualUnit(Qwen3OmniMoeCode2WavDecoderResidualUnit):
    pass


# ─── V1 Tokenizer Component Aliases ──────────────────────────────────────────


class Qwen3TTSTokenizerV1DiTCodecEmbedding(DiTCodecEmbedding):
    pass


class Qwen3TTSTokenizerV1DiTMLP(DiTMLP):
    pass


class Qwen3TTSTokenizerV1DiTAttention(DiTAttention):
    pass


class Qwen3TTSTokenizerV1DiTDecoderLayer(DiTDecoderLayer):
    pass


class Qwen3TTSTokenizerV1DiTTimestepEmbedding(DiTTimestepEmbedding):
    pass


class Qwen3TTSTokenizerV1SinusoidsPositionEmbedding(SinusoidsPositionEmbedding):
    pass


class Qwen3TTSTokenizerV1AdaLayerNormZero_Final(Qwen2_5_OmniAdaLayerNormZero_Final):
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


class Qwen3TTSTokenizerV2DecoderPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSTokenizerV2Code2WavConfig
    _can_compile_fullgraph = False
    _no_split_modules = ["Qwen3TTSTokenizerV2Block"]


class Qwen3TTSTokenizerV1DecoderBigVGANModel(Qwen2_5OmniToken2WavBigVGANModel):
    config: Qwen3TTSTokenizerV1DecoderBigVGANConfig
    config_class = Qwen3TTSTokenizerV1DecoderBigVGANConfig

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__(config)
        # Override conv_pre: kernel 5 + padding 2 instead of parent's kernel 7 + padding 3
        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 5, 1, padding=2)
        # Override resblocks: V1 uses causal AMPBlock with causal_type parameter
        self.resblocks = nn.ModuleList(
            [
                AMPBlock(
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    dilation,
                    "1" if layer_idx > 1 else "2",
                )
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )


class Qwen3TTSTokenizerV2TransformerModel(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV2Code2WavConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerV2Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerV2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.window_size = config.sliding_window
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is not None:
            inputs_embeds = self.input_proj(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3TTSTokenizerV2Block(Qwen3OmniMoeCode2WavTransformerLayer):
    pass


class Qwen3TTSTokenizerV2DecoderBlock(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV2Code2WavConfig, layer_idx):
        super().__init__(config)
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]
        block = [
            SnakeBeta(in_dim),
            Qwen3TTSTokenizerV2CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]
        for dilation in (1, 3, 9):
            block.append(Qwen3TTSTokenizerV2ResidualUnit(out_dim, dilation))
        self.block = nn.ModuleList(block)
        self.post_init()

    def forward(self, hidden, **kwargs):
        for block in self.block:
            hidden = block(hidden)
        return hidden


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

    _tied_weights_keys = ["lm_head.weight"]
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

    _tied_weights_keys = ["lm_head.weight"]
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


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window
    for STFT (using torch.stft).
    """
    from librosa.filters import mel as librosa_mel_fn

    if torch.min(y) < -1.0:
        logger.warning(f"Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        logger.warning(f"Max value of input waveform signal is {torch.max(y)}")

    device = y.device

    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = dynamic_range_compression_torch(mel_spec)

    return mel_spec


class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel):
    config_class = Qwen3TTSConfig

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
            list(self.config.talker_config.spk_id.keys()) if hasattr(self.config.talker_config, "spk_id") else []
        )
        self.supported_languages = ["auto"]
        if hasattr(self.config.talker_config, "codec_language_id"):
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        import json

        from ...utils.hub import cached_file

        # Hotfix to enable passing the correct attn implementation which is stored in the config but not in kwargs
        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if requested_attn_implementation is None and config and config._attn_implementation:
            requested_attn_implementation = config._attn_implementation

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            attn_implementation=requested_attn_implementation,
            **kwargs,
        )

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=token,
            revision=kwargs.pop("revision", None),
        )
        if generate_config_path is not None:
            with open(generate_config_path, encoding="utf-8") as f:
                generate_config = json.load(f)
            model.load_generate_config(generate_config)

        return model

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        speaker_embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return speaker_embedding

    @torch.inference_mode()
    def generate_speaker_prompt(self, voice_clone_prompt: list[dict]):
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt["ref_spk_embedding"])):
            ref_spk_embedding = (
                voice_clone_prompt["ref_spk_embedding"][index].to(self.talker.device).to(self.talker.dtype)
            )
            voice_clone_spk_embeds.append(ref_spk_embedding)
        return voice_clone_spk_embeds

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i : i + 1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.talker.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.talker_config.codec_bos_id]],
                        device=self.talker.device,
                        dtype=text_id.dtype,
                    )
                ),
                codec_embed,
            ],
            dim=1,
        )
        # compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
                return text_embed + codec_embed, tts_pad_embed

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[torch.Tensor] | None = None,
        instruct_ids: list[torch.Tensor] | None = None,
        ref_ids: list[torch.Tensor] | None = None,
        voice_clone_prompt: list[dict] | None = None,
        languages: list[str] | None = None,
        speakers: list[str] | None = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True),
        }

        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                    )

        # tts text prompt generate
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:  # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                            torch.tensor(spk_id, device=self.talker.device, dtype=input_id.dtype)
                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None

            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker != ""
                and speaker is not None
                and self.config.talker_config.spk_is_dialect[speaker.lower()] is not False
            ):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]

            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]
            else:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=self.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat(
                    [codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1
                )

            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

            # tts_pad * N + tts_bos
            _talker_input_embed = (
                torch.cat(
                    (tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed),
                    dim=1,
                )
                + codec_input_emebdding[:, :-1]
            )

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt["ref_code"] is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                # tts_text_first_token
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4]))
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5])),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[self.config.talker_config.codec_bos_id]],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        # for batch inference
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        # left padding for talker input embeds
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(sequences_reversed, batch_first=True, padding_value=0.0)
        talker_input_embeds = padded_reversed.flip(dims=[1])
        # generate mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        # padding trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        # forward
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
        talker_hidden_states = torch.cat([hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1)[:, :-1]

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        talker_codes_list = [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]
        talker_hidden_states_list = [talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)]

        return talker_codes_list, talker_hidden_states_list


# ─── Speech Tokenizer V2 (12Hz) Model Classes ─────────────────────────────────
# These model classes are internal to Qwen3-TTS. They are defined here because
# they are only used by this model (following the Qwen2.5-Omni Token2Wav pattern).


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: list[torch.LongTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2Output(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: list[torch.FloatTensor] = None


# --------------  Qwen3TTSTokenizerV2Decoder----------------


class Qwen3TTSTokenizerV2EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class Qwen3TTSTokenizerV2VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim: int | None = None, epsilon: float = 1e-5):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim
        requires_projection = codebook_dim != dim
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.epsilon = epsilon
        self._codebook = Qwen3TTSTokenizerV2EuclideanCodebook(
            dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        return quantized.transpose(1, 2)


class Qwen3TTSTokenizerV2ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3TTSTokenizerV2VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class Qwen3TTSTokenizerV2ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: int | None = None,
        output_dimension: int | None = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(self.input_dimension, self.dimension, 1, bias=False)
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)
        self.vq = Qwen3TTSTokenizerV2ResidualVectorQuantization(
            dim=self.dimension, codebook_size=self.bins, num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class Qwen3TTSTokenizerV2SplitResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest."""

    def __init__(self, *, n_q: int = 8, n_q_semantic: int = 1, **kwargs):
        super().__init__()
        assert n_q > n_q_semantic
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = Qwen3TTSTokenizerV2ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = Qwen3TTSTokenizerV2ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic, force_projection=True, q_dropout=q_dropout, **kwargs
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class Qwen3TTSTokenizerV2Decoder(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    config_class = Qwen3TTSTokenizerV2Code2WavConfig

    def __init__(self, config: Qwen3TTSTokenizerV2Code2WavConfig):
        super().__init__(config)
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        self.pre_transformer = Qwen3TTSTokenizerV2TransformerModel._from_config(config)

        self.quantizer = Qwen3TTSTokenizerV2SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerV2CausalConvNet(config.codebook_dim, config.latent_dim, kernel_size=3)

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerV2CausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3TTSTokenizerV2ConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerV2CausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerV2DecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3TTSTokenizerV2CausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.post_init()

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


@auto_docstring
class Qwen3TTSTokenizerV2PreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSTokenizerV2Config
    _can_compile_fullgraph = False


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV2 model.
    """
)
class Qwen3TTSTokenizerV2Encoder(MimiModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None

        self.post_init()


class Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2PreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV2Config):
        super().__init__(config)
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(self.config.encoder_config)
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(self.config.decoder_config)

        self.post_init()

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | Qwen3TTSTokenizerV2EncoderOutput:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
        audio_codes = encoded_frames.audio_codes[:, : self.encoder_valid_num_quantizers]
        audio_codes = [
            code[..., : -(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1)
            for code, mask in zip(audio_codes, padding_mask)
        ]

        if not return_dict:
            return (audio_codes,)

        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes)

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | Qwen3TTSTokenizerV2Output:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)

        audio_values = [a[:length] for a, length in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV2Output(audio_values)


# ----------------------------------V1 Tokenizer------------------


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class AMPBlock(AMPBlock):
    """AMPBlock with CausalConv1d support for Qwen3TTS."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        causal_type="1",
    ):
        nn.Module.__init__(self)

        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[0]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[1]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[2]),
            ]
        )

        if causal_type == "1":
            self.convs2 = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    ),
                ]
            )
        else:
            self.convs2 = nn.ModuleList(
                [
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                    CausalConv1d(channels, channels, kernel_size, 1, dilation=1),
                ]
            )

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

        if causal_type == "2":
            self.pre_conv = nn.Conv1d(
                channels, channels, kernel_size, stride=1, padding=self._get_padding(kernel_size, 1)
            )
            self.pre_act = TorchActivation1d(activation=SnakeBeta(channels))
        else:
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()


class Qwen3TTSTokenizerV1DecoderPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSTokenizerV1DecoderConfig
    _can_compile_fullgraph = False


class Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerV1DecoderDiTModel(Qwen2_5OmniToken2WavDiTModel):
    config: Qwen3TTSDiTConfig
    config_class = Qwen3TTSDiTConfig
    _no_split_modules = ["Qwen3TTSTokenizerV1DiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSDiTConfig):
        super().__init__(config)
        # V1 uses a simpler rotary embedding that takes only x (no position_ids)
        self.rotary_embed = Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(config.head_dim)
        # V1 uses the V1 DiT decoder layer alias
        self.transformer_blocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerV1DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
    ):
        # V1: batch_size accounts for CFG doubling that happens inside input_embed
        batch_size = hidden_states.shape[0] * 2
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        # V1: rotary_embed takes only hidden_states (no separate position_ids)
        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)
        return output

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        # V1: pre-allocate large buffer then slice to needed duration
        noise_initialization = torch.randn(
            [quantized_code.shape[0], 30000, self.mel_dim], dtype=reference_mel_spectrogram.dtype
        )
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(quantized_code.device)
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                return self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        time_embedding = torch.linspace(0, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype)
        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        # V1: Euler ODE solver (parent uses RK4)
        values = initial_state.clone()
        for t0, t1 in zip(time_embedding[:-1], time_embedding[1:]):
            dt = t1 - t0
            vt = ode_function(t0, values)
            values = values + vt * dt

        return values.permute(0, 2, 1)


class Qwen3TTSTokenizerV1Decoder(Qwen2_5OmniToken2WavModel):
    config: Qwen3TTSTokenizerV1DecoderConfig
    config_class = Qwen3TTSTokenizerV1DecoderConfig
    _no_split_modules = ["Qwen3TTSTokenizerV1DecoderDiTModel", "Qwen3TTSTokenizerV1DecoderBigVGANModel"]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig):
        # Skip parent's __init__ to use V1 attribute names (self.dit / self.bigvgan)
        # that match the original checkpoint state dict keys
        PreTrainedModel.__init__(self, config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder must inference with fp32, but flash_attention_2 only supports "
                "fp16 and bf16, attention implementation of Qwen3TTSTokenizerV1Decoder will fallback to "
                "sdpa."  # noqa: E501
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.dit = Qwen3TTSTokenizerV1DecoderDiTModel._from_config(config.dit_config, attn_implementation=attn_impl)
        self.bigvgan = Qwen3TTSTokenizerV1DecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        """Generates a waveform from input code and conditioning parameters."""
        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        waveform = self.bigvgan(mel_spectrogram)
        return waveform


# ─── Speech Tokenizer V1 (25Hz) Encoder classes ──────────────────────────────
# These are inlined from the original Qwen3-TTS vq/ submodule (core_vq.py,
# whisper_encoder.py, speech_vq.py) using V1-prefixed private names so they
# don't conflict with the V2 codec classes defined above.


# ── Helper functions ──────────────────────────────────────────────────────────


@cache
def _v1_mel_filters(device, n_mels: int) -> torch.Tensor:
    """Compute mel filterbank via librosa (replaces the original npz asset)."""
    from librosa.filters import mel as librosa_mel_fn

    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    mel = librosa_mel_fn(sr=16000, n_fft=400, n_mels=n_mels)
    return torch.from_numpy(mel).to(device)


def _v1_log_mel_spectrogram(audio, n_mels=80, padding=0, device=None):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _v1_mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def _v1_get_T_after_cnn(L_in, dilation=1):
    for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def _v1_get_mel_audio(audio, padding=False, audio_vq_ds_rate=1, n_mels=128):
    audio_len = len(audio)
    if padding:
        reduction = 160 * 2 * audio_vq_ds_rate
        audio_pad = math.ceil(audio_len / reduction) * reduction - audio_len
        mel = _v1_log_mel_spectrogram(audio, n_mels=n_mels, padding=audio_pad)
    else:
        mel = _v1_log_mel_spectrogram(audio, n_mels=n_mels)
    return mel


def _v1_sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ── VQ core classes (inference-only port of core_vq.py) ──────────────────────


class _V1EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance (inference subset)."""

    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.99,
        epsilon=1e-5,
        threshold_ema_dead_code=2.0,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        # buffers are held by DistributedResidualVectorQuantization and passed at call-time
        self.inited = None
        self.cluster_size = None
        self.embed = None
        self.embed_avg = None

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices

    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)

    def encode(self, x, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        embed_ind = self.quantize(x)
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind, buffers):
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers
        return self.dequantize(embed_ind)


class _V1VectorQuantization(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        decay=0.99,
        epsilon=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2.0,
        commitment_weight=1.0,
    ):
        super().__init__()
        _codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim
        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        self._codebook = _V1EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    def encode(self, x, buffers):
        x = self.project_in(x)
        return self._codebook.encode(x, buffers)

    def decode(self, embed_ind, buffers):
        quantize = self._codebook.decode(embed_ind, buffers)
        return self.project_out(quantize)


class _V1DistributedRVQ(nn.Module):
    """Distributed residual VQ (inference subset of DistributedResidualVectorQuantization)."""

    def __init__(self, *, num_quantizers, quantize_dropout=False, rand_num_quant=None, **kwargs):
        super().__init__()
        codebook_size = kwargs["codebook_size"]
        codebook_dim = kwargs.get("codebook_dim") or kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]

        if isinstance(kmeans_init, bool):
            if not kmeans_init:
                embed = torch.empty(num_quantizers, codebook_size, codebook_dim)
                nn.init.kaiming_uniform_(embed)
                inited = True
            else:
                embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
                inited = False
        else:
            raise TypeError("kmeans_init should be bool")

        self.register_buffer("inited", torch.Tensor([[inited]] * num_quantizers))
        self.register_buffer("cluster_size", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        self.layers = nn.ModuleList([_V1VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def encode(self, x, n_q=None):
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            buffers = [self.inited[i], self.cluster_size[i], self.embed[i], self.embed_avg[i]]
            indices = layer.encode(residual, buffers)
            quantized = layer.decode(indices, buffers)
            residual = residual - quantized
            all_indices.append(indices)
        return torch.stack(all_indices)

    def decode(self, q_indices):
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            buffers = [self.inited[i], self.cluster_size[i], self.embed[i], self.embed_avg[i]]
            quantized_out = quantized_out + self.layers[i].decode(indices, buffers)
        return quantized_out


class _V1DistributedGroupRVQ(nn.Module):
    """Distributed group RVQ (inference subset of DistributedGroupResidualVectorQuantization)."""

    def __init__(self, *, num_groups, num_quantizers, quantize_dropout=False, rand_num_quant=None, **kwargs):
        super().__init__()
        self.rvqs = nn.ModuleList(
            [
                _V1DistributedRVQ(
                    num_quantizers=num_quantizers,
                    quantize_dropout=quantize_dropout,
                    rand_num_quant=rand_num_quant,
                    **kwargs,
                )
                for _ in range(num_groups)
            ]
        )
        self.num_groups = num_groups

    def encode(self, x, n_q=None):
        x_lst = torch.chunk(x, chunks=self.num_groups, dim=1)
        return torch.stack([mod.encode(item, n_q) for mod, item in zip(self.rvqs, x_lst)], dim=1)

    def decode(self, q_indices):
        q_indices_lst = torch.chunk(q_indices, chunks=self.num_groups, dim=1)
        return torch.cat([mod.decode(item.squeeze(1)) for mod, item in zip(self.rvqs, q_indices_lst)], dim=1)


# ── Whisper encoder classes (port of vq/whisper_encoder.py) ──────────────────


class _V1Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class _V1ConvTranspose1d(nn.ConvTranspose1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class _V1Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class _V1MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = _V1Linear(n_state, n_state)
        self.key = _V1Linear(n_state, n_state, bias=False)
        self.value = _V1Linear(n_state, n_state)
        self.out = _V1Linear(n_state, n_state)

    def forward(self, x, cu_seqlens=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self._qkv_attention_manual(q, k, v, cu_seqlens=cu_seqlens)
        return self.out(x)

    def _qkv_attention_manual(self, q, k, v, cu_seqlens):
        n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        scale = head_dim**-0.5

        q = q.view(n_ctx, self.n_head, head_dim)
        k = k.view(n_ctx, self.n_head, head_dim)
        v = v.view(n_ctx, self.n_head, head_dim)

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)

        q_padded = torch.zeros(batch_size, max_seqlen, self.n_head, head_dim, dtype=q.dtype, device=q.device)
        k_padded = torch.zeros_like(q_padded)
        v_padded = torch.zeros_like(q_padded)

        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_len = seqlens[i]
            q_padded[i, :seq_len] = q[start_idx:end_idx]
            k_padded[i, :seq_len] = k[start_idx:end_idx]
            v_padded[i, :seq_len] = v[start_idx:end_idx]

        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        attn_mask = (
            (torch.arange(max_seqlen, device=q.device)[None, :] < torch.tensor(seqlens, device=q.device)[:, None])
            .unsqueeze(1)
            .unsqueeze(2)
        )
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -torch.finfo(q.dtype).max)

        attn_scores = torch.matmul(q_padded, k_padded.transpose(-2, -1)) * scale + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v_padded)
        context = context.transpose(1, 2).contiguous().view(batch_size, max_seqlen, n_state)
        return torch.cat([context[i, : seqlens[i]] for i in range(batch_size)], dim=0)


class _V1ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, enable_mp=False, sequence_parallel=False):
        super().__init__()
        n_mlp = n_state * 4
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp_ln = nn.LayerNorm(n_state)
        self.attn = _V1MultiHeadAttention(n_state, n_head)
        self.mlp = nn.Sequential(_V1Linear(n_state, n_mlp), nn.GELU(), _V1Linear(n_mlp, n_state))

    def forward(self, x, cu_seqlens=None):
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class _V1WhisperEncoder(nn.Module):
    def __init__(
        self,
        n_mels,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        n_window=1500,
        output_dim=512,
        grad_checkpointing=False,
        enable_mp=False,
        audio_sequence_parallel=False,
    ):
        super().__init__()
        self.conv1 = _V1Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = _V1Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", _v1_sinusoids(n_ctx, n_state))
        self.n_layer = n_layer
        self.n_mels = n_mels
        self.blocks = nn.ModuleList(
            [
                _V1ResidualAttentionBlock(
                    n_state, n_head, enable_mp=enable_mp, sequence_parallel=audio_sequence_parallel
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_post = nn.LayerNorm(n_state)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(n_state, output_dim)
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        self.output_dim = output_dim
        self.n_head = n_head
        self.n_state = n_state
        self.n_window = n_window


class _V1WhisperEncoderVQ(_V1WhisperEncoder):
    """WhisperEncoder extended with a VQ bottleneck (inference-only port of WhisperEncoderVQ)."""

    def __init__(
        self,
        n_mels,
        n_ctx,
        n_state,
        n_head,
        n_layer,
        n_window=1500,
        output_dim=512,
        grad_checkpointing=False,
        enable_mp=False,
        audio_sequence_parallel=False,
        audio_vq_layers=-1,
        audio_vq_type="NULL",
        audio_vq_codebook_size=4096,
        audio_vq_pe=False,
        audio_vq_commit_loss=0.0,
        audio_vq_out_commit_loss=0.0,
        audio_vq_no_quantize=False,
        audio_vq_ff_layer=0,
        audio_vq_threshold_ema_dead_code=0.1,
        audio_vq_codebook_dim=None,
        audio_vq_ds_rate=None,
    ):
        super().__init__(
            n_mels,
            n_ctx,
            n_state,
            n_head,
            n_layer,
            n_window,
            output_dim,
            grad_checkpointing,
            enable_mp,
            audio_sequence_parallel,
        )
        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_type = audio_vq_type
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_commit_loss = audio_vq_commit_loss
        self.audio_vq_out_commit_loss = audio_vq_out_commit_loss
        self.audio_vq_no_quantize = audio_vq_no_quantize
        self.audio_vq_ff_layer = audio_vq_ff_layer

        if audio_vq_layers > 0:
            self.vq_feature_dim = self.n_state
            self.audio_vq_ds_rate = 1
        else:
            raise NotImplementedError(f"Unsupported audio_vq_layers: {audio_vq_layers}")

        if self.audio_vq_ds_rate == audio_vq_ds_rate:
            self.audio_vq_downsample = nn.Identity()
            self.audio_vq_upsample = nn.Identity()
        else:
            assert audio_vq_ds_rate % self.audio_vq_ds_rate == 0
            stride = audio_vq_ds_rate // self.audio_vq_ds_rate
            self.audio_vq_downsample = _V1Conv1d(
                self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride
            )
            self.audio_vq_upsample = _V1ConvTranspose1d(
                self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride
            )
            self.audio_vq_ds_rate = audio_vq_ds_rate

        codebook_dim_for_vq = audio_vq_codebook_dim if audio_vq_codebook_dim is not None else self.vq_feature_dim
        if audio_vq_type == "GRVQ":
            self.audio_quantizer = _V1DistributedGroupRVQ(
                codebook_size=audio_vq_codebook_size,
                dim=self.vq_feature_dim,
                codebook_dim=codebook_dim_for_vq,
                num_groups=1,
                num_quantizers=1,
                kmeans_init=False,
                threshold_ema_dead_code=audio_vq_threshold_ema_dead_code,
            )
        else:
            raise NotImplementedError(f"Unsupported audio_vq_type: {audio_vq_type}")

        if self.audio_vq_pe:
            self.project_after_vq_pe = nn.Linear(self.n_state, self.n_state)

    def _do_quantize(self, x, pe=None, y=None):
        x = x.unsqueeze(0)
        x = self.audio_vq_downsample(x.transpose(1, 2))
        x = x.transpose(1, 2)
        indices = self.audio_quantizer.encode(x)
        x = self.audio_quantizer.decode(indices)
        indices = indices.squeeze(2).squeeze(1)
        x, indices = x.squeeze(0), indices.squeeze(0)
        if self.audio_vq_pe:
            x = x + pe
            x = self.project_after_vq_pe(x)
        x = self.audio_vq_upsample(x.unsqueeze(0).transpose(1, 2))
        x = x.transpose(1, 2).squeeze(0)
        return x, indices, {}

    def forward(
        self, x_list, audio_mellens, audio_aftercnnlens, audio_seqlens, return_indices=False, audio_pitchs=None
    ):
        aftercnn_x_list = []
        pe_for_vq_list = []
        for each_x in x_list:
            for each_x_split in each_x.split(self.n_window * 2, dim=1):
                each_x_split = F.gelu(self.conv1(each_x_split))
                each_x_split = F.gelu(self.conv2(each_x_split))
                each_x_split = each_x_split.permute(1, 0)
                each_positional_embedding_split = self.positional_embedding[: each_x_split.shape[0]]
                aftercnn_x_list.append(each_x_split + each_positional_embedding_split.to(each_x_split.dtype))
                pe_for_vq_split = self.positional_embedding[: each_x_split.shape[0] // self.audio_vq_ds_rate]
                pe_for_vq_list.append(pe_for_vq_split.to(each_x_split.dtype))

        pe_for_vq = torch.cat(pe_for_vq_list, dim=0)
        x = torch.cat(aftercnn_x_list, dim=0)

        output_list = []
        for item in audio_aftercnnlens:
            while item > self.n_window:
                output_list.append(self.n_window)
                item -= self.n_window
            output_list.append(item)

        cu_seqlens_list = list(accumulate(output_list, func=operator.add, initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens_list).to(device=x.device, dtype=torch.int32)

        layer_id = 0
        for block in self.blocks:
            layer_id += 1
            x = block(x, cu_seqlens=cu_seqlens)
            if self.audio_vq_layers == layer_id:
                x, indices, vq_stats = self._do_quantize(x, pe_for_vq)
                if return_indices:
                    return x, indices

        if self.avg_pooler:
            x_list_split = x.split(audio_aftercnnlens, dim=0)
            token_x_list = []
            for xi in x_list_split:
                xi = xi.permute(1, 0)
                xi = self.avg_pooler(xi)
                xi = xi.permute(1, 0)
                token_x_list.append(xi)
            x = torch.cat(token_x_list, dim=0)

        x = self.ln_post(x)
        x = self.proj(x)

        output = torch.zeros((x.size(0) + len(audio_seqlens) * 2, x.size(1)), device=x.device, dtype=x.dtype)
        audio_seqlens_acc = list(accumulate(audio_seqlens, func=operator.add, initial=0))
        start_ids = torch.tensor(audio_seqlens_acc[:-1], device=x.device, dtype=torch.int32)
        end_ids = torch.tensor(audio_seqlens_acc[1:], device=x.device, dtype=torch.int32) - 1
        audio_tokens_mask = torch.ones(output.size(0), device=x.device, dtype=torch.bool)
        audio_tokens_mask[start_ids] = False
        audio_tokens_mask[end_ids] = False
        output[start_ids] = self.audio_bos_eos_token.weight[0].to(x.dtype)
        output[end_ids] = self.audio_bos_eos_token.weight[1].to(x.dtype)
        output[audio_tokens_mask] = x

        if self.audio_vq_type != "NULL":
            return output, vq_stats
        return output


# ── Qwen3TTSTokenizerV1XVectorExtractor (lazy external imports) ─────────────────────────────────


class _V1MelSpectrogramFeatures(nn.Module):
    """Mel spectrogram extractor used by Qwen3TTSTokenizerV1XVectorExtractor."""

    def __init__(
        self,
        filter_length=1024,
        hop_length=160,
        win_length=640,
        n_mel_channels=80,
        mel_fmin=0,
        mel_fmax=8000,
        sampling_rate=16000,
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate

    def extract(self, audio):
        from librosa.filters import mel as librosa_mel_fn

        y = audio
        if len(y.shape) == 3:
            y = y.squeeze(1) if y.shape[1] == 1 else y.squeeze(2)
        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )
        mel_basis = torch.from_numpy(mel).float().to(y.device)
        hann_window = torch.hann_window(self.win_length).to(y.device)
        pad = int((self.filter_length - self.hop_length) / 2)
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(mel_basis, spec)
        return torch.log(torch.clamp(spec, min=1e-5))


class Qwen3TTSTokenizerV1XVectorExtractor(nn.Module):
    """Speaker x-vector extractor using an ONNX model (campplus.onnx).

    External dependencies (onnxruntime, sox, torchaudio) are imported lazily
    so that the main transformers package does not require them.
    """

    def __init__(self, audio_codec_with_xvector):
        super().__init__()
        import onnxruntime
        import sox

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.ort_session = onnxruntime.InferenceSession(
            audio_codec_with_xvector, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.tfm = sox.Transformer()
        self.tfm.norm(db_level=-6)
        self.mel_ext = _V1MelSpectrogramFeatures()

    def extract_code(self, audio):
        import copy

        import torchaudio.compliance.kaldi as kaldi

        with torch.no_grad():
            norm_audio = self._sox_norm(audio)
            norm_audio_tensor = torch.from_numpy(copy.deepcopy(norm_audio)).unsqueeze(0)
            feat = kaldi.fbank(norm_audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            norm_embedding = self.ort_session.run(
                None, {self.ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
            )[0].flatten()
            norm_embedding = F.normalize(torch.from_numpy(norm_embedding), dim=0)
            ref_mel = self.mel_ext.extract(audio=norm_audio_tensor)
        return norm_embedding.numpy(), ref_mel.permute(0, 2, 1).squeeze(0).numpy()

    def _sox_norm(self, audio):
        return self.tfm.build_array(input_array=audio, sample_rate_in=16000)


# ── V1 Encoder PreTrainedModel + Encoder ─────────────────────────────────────


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discrete code embeddings computed using `model.encode`, each tensor has shape `(codes_length_i,)`.
    xvectors (`List[torch.FloatTensor]`):
        X-vector speaker embeddings, each tensor has shape `(xvector_dim,)`.
    ref_mels (`List[torch.FloatTensor]`):
        Reference mel spectrogram, each tensor has shape `(mel_length_i, mel_dim)`.
    """

    audio_codes: list[torch.LongTensor] = None
    xvectors: list[torch.FloatTensor] = None
    ref_mels: list[torch.FloatTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1DecoderOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio waveforms, each tensor has shape `(segment_length_i,)`.
    """

    audio_values: list[torch.FloatTensor] = None


class Qwen3TTSTokenizerV1EncoderPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSTokenizerV1EncoderConfig
    _can_compile_fullgraph = False


class Qwen3TTSTokenizerV1Encoder(Qwen3TTSTokenizerV1EncoderPreTrainedModel):
    """Whisper-based VQ encoder that converts waveforms to discrete audio codes."""

    def __init__(self, config: Qwen3TTSTokenizerV1EncoderConfig):
        super().__init__(config)
        self.tokenizer = _V1WhisperEncoderVQ(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_window=config.n_window,
            output_dim=config.output_dim,
            grad_checkpointing=config.grad_checkpointing,
            enable_mp=config.enable_mp,
            audio_sequence_parallel=config.audio_sequence_parallel,
            audio_vq_type=config.audio_vq_type,
            audio_vq_layers=config.audio_vq_layers,
            audio_vq_codebook_size=config.audio_vq_codebook_size,
            audio_vq_codebook_dim=config.audio_vq_codebook_dim,
            audio_vq_pe=config.audio_vq_pe,
            audio_vq_ds_rate=config.audio_vq_ds_rate,
        )
        self.padding = True
        self.audio_vq_ds_rate = self.tokenizer.audio_vq_ds_rate

    def speech2mel(self, speechs):
        return [
            _v1_get_mel_audio(speech, padding=self.padding, audio_vq_ds_rate=self.audio_vq_ds_rate)
            .to(speech.dtype)
            .to(self.tokenizer.conv1.weight.device)
            for speech in speechs
        ]

    def mel2code(self, mels):
        audio_mellens = [mel.size(-1) for mel in mels]
        audio_aftercnnlens = [_v1_get_T_after_cnn(T) for T in audio_mellens]
        audio_seqlens = [T + 2 for T in audio_aftercnnlens]
        with torch.no_grad():
            _, indices = self.tokenizer(
                x_list=mels,
                audio_mellens=audio_mellens,
                audio_aftercnnlens=audio_aftercnnlens,
                audio_seqlens=audio_seqlens,
                return_indices=True,
            )
        indice_lens = [T // self.tokenizer.audio_vq_ds_rate for T in audio_aftercnnlens]
        indices = pad_sequence(torch.split(indices, indice_lens), batch_first=True, padding_value=0)
        return indices, indice_lens

    def quantize_speech(self, speechs):
        mels = self.speech2mel(speechs)
        return self.mel2code(mels)


# ── V1 Top-level model ────────────────────────────────────────────────────────


@auto_docstring
class Qwen3TTSTokenizerV1PreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSTokenizerV1Config
    _can_compile_fullgraph = False


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV1 model combining a Whisper-based VQ encoder and a DiT-based decoder.
    """
)
class Qwen3TTSTokenizerV1Model(Qwen3TTSTokenizerV1PreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerV1Config):
        super().__init__(config)
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerV1Encoder._from_config(self.config.encoder_config)
        self.decoder = Qwen3TTSTokenizerV1Decoder._from_config(self.config.decoder_config)
        self.encoder_xvector_extractor = None

        self.post_init()

    def load_encoder_xvector_extractor(self, model_path):
        self.encoder_xvector_extractor = Qwen3TTSTokenizerV1XVectorExtractor(model_path)

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        try:
            extractor_path = cached_file(pretrained_model_name_or_path, "campplus.onnx")
            if extractor_path is not None:
                model.load_encoder_xvector_extractor(extractor_path)
        except Exception:
            logger.warning_once(
                "Could not load campplus.onnx for Qwen3TTSTokenizerV1XVectorExtractor. "
                "Call model.load_encoder_xvector_extractor(path) manually before calling encode()."
            )
        return model

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | Qwen3TTSTokenizerV1EncoderOutput:
        """
        Encodes input audio waveforms into discrete codes, x-vectors, and reference mel spectrograms.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Binary mask where 1 = valid, 0 = padding.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        wavs = [value[: mask.sum()] for value, mask in zip(input_values, padding_mask)]
        codes, codes_lens = self.encoder.quantize_speech(wavs)
        codes = [c[:length] for c, length in zip(codes, codes_lens)]

        xvectors = []
        ref_mels = []
        for wav in wavs:
            xvector, ref_mel = self.encoder_xvector_extractor.extract_code(wav.cpu().numpy())
            xvector = torch.tensor(xvector).to(wav.dtype).to(wav.device)
            ref_mel = torch.tensor(ref_mel).to(wav.dtype).to(wav.device)
            xvectors.append(xvector)
            ref_mels.append(ref_mel)

        if not return_dict:
            return (codes, xvectors, ref_mels)
        return Qwen3TTSTokenizerV1EncoderOutput(audio_codes=codes, xvectors=xvectors, ref_mels=ref_mels)

    def decode(
        self,
        audio_codes: torch.Tensor,
        xvectors: torch.Tensor,
        ref_mels: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple | Qwen3TTSTokenizerV1DecoderOutput:
        """
        Decodes discrete codes + speaker conditioning into an audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length)`):
                Discrete code embeddings from `model.encode`.
            xvectors (`torch.FloatTensor` of shape `(batch_size, xvector_dim)`):
                X-vector speaker embeddings from `model.encode`.
            ref_mels (`torch.FloatTensor` of shape `(batch_size, mel_length, mel_dim)`):
                Reference mel spectrogram from `model.encode`.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        audio_values = self.decoder(code=audio_codes, reference_mel=ref_mels, conditioning=xvectors)
        audio_lengths = (audio_codes > 0).sum(1) * self.decode_upsample_rate
        audio_values = [a[:length] for a, length in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)
        return Qwen3TTSTokenizerV1DecoderOutput(audio_values=audio_values)


__all__ = [
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTokenizerV1DecoderPreTrainedModel",
    "Qwen3TTSTokenizerV1DecoderDiTModel",
    "Qwen3TTSTokenizerV1Decoder",
    "Qwen3TTSTokenizerV1EncoderPreTrainedModel",
    "Qwen3TTSTokenizerV1Encoder",
    "Qwen3TTSTokenizerV1Model",
    "Qwen3TTSTokenizerV1PreTrainedModel",
    "Qwen3TTSTokenizerV2Model",
    "Qwen3TTSTokenizerV2PreTrainedModel",
]
