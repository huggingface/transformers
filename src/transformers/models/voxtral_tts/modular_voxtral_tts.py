# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

import math

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from .configuration_voxtral_tts import VoxtralTtsCodecConfig, VoxtralTtsConfig, VoxtralTtsFlowMatchingConfig
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralMLP,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)


class VoxtralTtsMLP(MistralMLP):
    pass


class VoxtralTtsAttention(MistralAttention):
    pass


class VoxtralTtsRMSNorm(MistralRMSNorm):
    pass


class VoxtralTtsDecoderLayer(MistralDecoderLayer):
    pass


class VoxtralTtsRotaryEmbedding(MistralRotaryEmbedding):
    pass


class VoxtralTtsAudioEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.embed_audio_tokens = nn.Embedding(config.audio_vocab_size, config.hidden_size)
        offsets = self._compute_offsets(config)
        self.register_buffer("audio_tokens_offsets", offsets, persistent=False)

    @staticmethod
    def _compute_offsets(config):
        offsets = torch.zeros(config.num_codebooks, dtype=torch.long)
        if config.n_acoustic_codebook > 1:
            acoustic_stride = (
                config.audio_vocab_size - config.semantic_codebook_size - config.acoustic_codebook_size
            ) // (config.n_acoustic_codebook - 1)
        else:
            acoustic_stride = config.acoustic_codebook_size
        for i in range(config.n_acoustic_codebook):
            offsets[i + 1] = config.semantic_codebook_size + i * acoustic_stride
        return offsets

    def forward(self, input_ids):
        inputs_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = inputs_embeds.sum(dim=2)
        return inputs_embeds


class VoxtralTtsPreTrainedModel(MistralPreTrainedModel):
    pass


class VoxtralTtsBackboneModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = VoxtralTtsAudioEmbeddings(config)


class VoxtralTtsFlowMatchingAttention(VoxtralTtsAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False


class VoxtralTtsFlowMatchingDecoderLayer(VoxtralTtsDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = VoxtralTtsFlowMatchingAttention(config=config, layer_idx=layer_idx)


class VoxtralTtsFlowMatchingTransformer(nn.Module):
    def __init__(self, config: VoxtralTtsFlowMatchingConfig):
        super().__init__()
        self.config = config

        self.llm_projection = nn.Linear(config.input_dim, config.hidden_size, bias=False)
        self.time_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.input_projection = nn.Linear(config.acoustic_dim, config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [VoxtralTtsFlowMatchingDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralTtsRotaryEmbedding(config=config)

        self.semantic_codebook_output = nn.Linear(config.hidden_size, config.semantic_vocab_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(config.hidden_size, config.acoustic_dim, bias=False)

    @staticmethod
    def _get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.llm_projection(hidden_states)
        t = self.time_projection(self._get_timestep_embedding(timesteps, self.config.hidden_size))
        x = self.input_projection(acoustic_embeddings)

        combined = h + t.unsqueeze(1) + x

        position_ids = torch.arange(combined.shape[1], device=combined.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(combined, position_ids=position_ids)

        for layer in self.layers:
            combined = layer(
                combined,
                position_embeddings=position_embeddings,
            )

        combined = self.norm(combined)

        semantic_logits = self.semantic_codebook_output(combined)
        acoustic_output = self.acoustic_codebook_output(combined)
        return semantic_logits, acoustic_output


# ==================== Codec Components ====================


class VoxtralTtsSemanticCodebook(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.register_buffer("cluster_usage", torch.ones(config.semantic_codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(config.semantic_codebook_size, config.semantic_dim))

    @property
    def embeddings(self):
        return self.embedding_sum / self.cluster_usage.unsqueeze(-1)

    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        return nn.functional.embedding(indices, self.embeddings)


class VoxtralTtsQuantizer(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.semantic_codebook = VoxtralTtsSemanticCodebook(config)

    def decode_semantic(self, indices: torch.LongTensor) -> torch.Tensor:
        return self.semantic_codebook.decode(indices)


class VoxtralTtsCodecAttention(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.is_causal = config.causal

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.use_qk_norm = config.qk_norm
        if config.qk_norm:
            self.q_norm = VoxtralTtsRMSNorm(self.num_heads * self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = VoxtralTtsRMSNorm(self.num_heads * self.head_dim, eps=config.qk_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=self.is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class VoxtralTtsCodecMLP(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VoxtralTtsCodecTransformerLayer(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = VoxtralTtsCodecAttention(config)
        self.mlp = VoxtralTtsCodecMLP(config)
        self.input_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_layer_scale = config.layer_scale
        if config.layer_scale:
            self.self_attn_layer_scale = nn.Parameter(
                torch.full((config.hidden_size,), config.layer_scale_init)
            )
            self.mlp_layer_scale = nn.Parameter(
                torch.full((config.hidden_size,), config.layer_scale_init)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        if self.use_layer_scale:
            hidden_states = self.self_attn_layer_scale * hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_layer_scale:
            hidden_states = self.mlp_layer_scale * hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class VoxtralTtsCodecTransformerBlock(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [VoxtralTtsCodecTransformerLayer(config, layer_idx=i) for i in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.transpose(1, 2)


class VoxtralTtsCodecConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = True,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.is_transpose = stride > 1

        if self.is_transpose:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
            self.trim_right = kernel_size - stride
            self.left_pad = 0
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
            self.trim_right = 0
            self.left_pad = kernel_size - 1 if causal else (kernel_size - 1) // 2

        if use_weight_norm:
            nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_transpose:
            hidden_states = self.conv(hidden_states)
            if self.trim_right > 0 and self.causal:
                hidden_states = hidden_states[..., : -self.trim_right]
        else:
            if self.causal and self.left_pad > 0:
                hidden_states = nn.functional.pad(hidden_states, (self.left_pad, 0))
            hidden_states = self.conv(hidden_states)
        return hidden_states


class VoxtralTtsCodecModel(nn.Module):

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.config = config
        self.quantizer = VoxtralTtsQuantizer(config)

        decoder_blocks = []
        in_channels = config.semantic_dim + config.acoustic_dim

        for kernel, stride, n_layers in zip(
            config.decoder_conv_kernels,
            config.decoder_conv_strides,
            config.decoder_transformer_lengths,
        ):
            decoder_blocks.append(
                VoxtralTtsCodecConvBlock(
                    in_channels,
                    config.hidden_size,
                    kernel,
                    stride,
                    causal=config.causal,
                    use_weight_norm=config.conv_weight_norm,
                )
            )
            decoder_blocks.append(VoxtralTtsCodecTransformerBlock(config, n_layers))
            in_channels = config.hidden_size

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.output_proj = VoxtralTtsCodecConvBlock(
            config.hidden_size,
            config.patch_size * config.channels,
            config.patch_proj_kernel_size,
            stride=1,
            causal=config.causal,
            use_weight_norm=config.conv_weight_norm,
        )
        self.patch_size = config.patch_size
        self.channels = config.channels

    def decode(
        self,
        semantic_token_ids: torch.LongTensor,
        acoustic_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode semantic codes and acoustic values to an audio waveform.

        Args:
            semantic_token_ids: (batch, seq_len) long tensor of semantic VQ indices
            acoustic_values: (batch, seq_len, acoustic_dim) float tensor of acoustic embeddings

        Returns:
            waveform: (batch, channels, num_samples) float tensor
        """
        semantic_embeddings = self.quantizer.decode_semantic(semantic_token_ids)
        decoder_input = torch.cat([semantic_embeddings, acoustic_values], dim=-1)

        hidden_states = decoder_input.transpose(1, 2)

        for block in self.decoder_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.output_proj(hidden_states)

        batch_size = hidden_states.shape[0]
        num_frames = hidden_states.shape[2]
        hidden_states = hidden_states.view(batch_size, self.channels, self.patch_size, num_frames)
        hidden_states = hidden_states.permute(0, 1, 3, 2).contiguous()
        waveform = hidden_states.view(batch_size, self.channels, num_frames * self.patch_size)

        return waveform


class VoxtralTtsForTextToSpeech(VoxtralTtsPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "embed_text_tokens.weight"}

    def __init__(self, config: VoxtralTtsConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        self.embed_text_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.backbone_model = VoxtralTtsBackboneModel._from_config(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.flow_matching_transformer = VoxtralTtsFlowMatchingTransformer(config.flow_matching_config)
        self.codec_model = VoxtralTtsCodecModel(config.codec_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_text_tokens

    def set_input_embeddings(self, value):
        self.embed_text_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            parts = []
            if audio_codes is not None:
                parts.append(self.backbone_model.embed_tokens(audio_codes))
            if input_ids is not None:
                parts.append(self.embed_text_tokens(input_ids))
            if parts:
                inputs_embeds = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        backbone_outputs = self.backbone_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = backbone_outputs[0]
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )


__all__ = [
    "VoxtralTtsPreTrainedModel",
    "VoxtralTtsBackboneModel",
    "VoxtralTtsCodecModel",
    "VoxtralTtsForTextToSpeech",
]
