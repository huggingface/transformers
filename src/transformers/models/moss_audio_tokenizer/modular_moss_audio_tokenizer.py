# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MossAudioTokenizer model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import auto_docstring, can_return_tuple, logging
from ..dac.feature_extraction_dac import DacFeatureExtractor
from ..dac.modeling_dac import DacVectorQuantize
from ..dinov2.modeling_dinov2 import Dinov2LayerScale
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ..whisper.modeling_whisper import WhisperEncoderLayer
from ..xcodec2.modeling_xcodec2 import Xcodec2DecoderOutput, Xcodec2EncoderOutput, Xcodec2Output
from .configuration_moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderConfig,
    MossAudioTokenizerEncoderConfig,
    MossAudioTokenizerQuantizerConfig,
    MossAudioTokenizerTransformerConfig,
)


logger = logging.get_logger(__name__)


@auto_docstring
@dataclass
class MossAudioTokenizerEncoderOutput(Xcodec2EncoderOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
        Discrete audio codes computed using the encoder and quantizer.
    latents (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`, *optional*):
        Continuous representation of the input audio computed by the encoder, before quantization.
    audio_codes_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.
    """


@auto_docstring
@dataclass
class MossAudioTokenizerDecoderOutput(Xcodec2DecoderOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
        Decoded audio waveform.
    audio_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Upsampled `padding_mask` indicating valid audio samples in `audio_values`.
    """

    audio_mask: torch.Tensor | None = None


@auto_docstring
@dataclass
class MossAudioTokenizerOutput(Xcodec2Output):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
        Decoded audio waveform.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
        Discrete audio codes computed using the encoder and quantizer.
    latents (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`, *optional*):
        Continuous representation of the input audio computed by the encoder, before quantization.
    audio_codes_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.
    """


class MossAudioTokenizerLayerScale(Dinov2LayerScale):
    pass


class MossAudioTokenizerRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class MossAudioTokenizerAttention(LlamaAttention):
    """Multi-head sliding-window causal attention."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class MossAudioTokenizerTransformerLayer(WhisperEncoderLayer):
    """Transformer layer with layer scale, used by both the encoder and the decoder."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig, layer_idx: int):
        super().__init__(config)
        self.self_attn = MossAudioTokenizerAttention(config, layer_idx)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        del self.activation_fn
        del self.activation_dropout
        self.layer_scale_1 = MossAudioTokenizerLayerScale(config)
        self.layer_scale_2 = MossAudioTokenizerLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + self.layer_scale_1(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        hidden_states = residual + self.layer_scale_2(hidden_states)
        return hidden_states


class MossAudioTokenizerTransformer(nn.Module):
    """Stack of transformer layers with sliding-window causal attention."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = MossAudioTokenizerRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [MossAudioTokenizerTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)
        attention_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
        return hidden_states


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Transformer stage with input/output projections."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig):
        super().__init__()
        self.input_proj = (
            nn.Linear(config.input_hidden_size, config.hidden_size, bias=False)
            if config.hidden_size != config.input_hidden_size
            else nn.Identity()
        )
        self.transformer = MossAudioTokenizerTransformer(config)
        self.output_proj = (
            nn.Linear(config.hidden_size, config.output_hidden_size, bias=False)
            if config.hidden_size != config.output_hidden_size
            else nn.Identity()
        )

    def forward(self, hidden_states, input_lengths, **kwargs):
        hidden_states = self.input_proj(hidden_states.transpose(1, 2))
        hidden_states = self.transformer(hidden_states, **kwargs)
        hidden_states = self.output_proj(hidden_states).transpose(1, 2)
        return hidden_states, input_lengths


class MossAudioTokenizerDownsample(nn.Module):
    """Patching module for downsampling."""

    def __init__(self, sampling_ratio: int):
        super().__init__()
        self.sampling_ratio = sampling_ratio

    def forward(self, hidden_states, input_lengths):
        batch_size, num_channels, _ = hidden_states.shape
        sampling_ratio = self.sampling_ratio
        hidden_states = (
            hidden_states.reshape(batch_size, num_channels, -1, sampling_ratio)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, num_channels * sampling_ratio, -1)
        )
        output_lengths = torch.div(input_lengths + sampling_ratio - 1, sampling_ratio, rounding_mode="floor")
        return hidden_states, output_lengths


class MossAudioTokenizerUpsample(nn.Module):
    """Patching module for upsampling."""

    def __init__(self, sampling_ratio: int):
        super().__init__()
        self.sampling_ratio = sampling_ratio

    def forward(self, hidden_states, input_lengths):
        batch_size, patch_channels, sequence_length = hidden_states.shape
        sampling_ratio = self.sampling_ratio
        num_channels = patch_channels // sampling_ratio
        hidden_states = (
            hidden_states.reshape(batch_size, num_channels, sampling_ratio, sequence_length)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, num_channels, sequence_length * sampling_ratio)
        )
        output_lengths = input_lengths * self.sampling_ratio
        return hidden_states, output_lengths


class MossAudioTokenizerLFQ(DacVectorQuantize):
    """LFQ (inference-only) used by ResidualLFQ."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize hidden states into codebook vectors."""
        hidden_states = hidden_states.float()
        projected_hidden_states = self.in_proj(hidden_states).float()
        quantized_hidden_states, indices = self.decode_latents(projected_hidden_states)
        quantized_hidden_states = (
            projected_hidden_states + (quantized_hidden_states - projected_hidden_states).detach()
        ).float()
        quantized_hidden_states = self.out_proj(quantized_hidden_states).float()
        return quantized_hidden_states, indices, projected_hidden_states

    def embed_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(code_indices, self.codebook.weight)

    def decode_code_wo_out_proj(self, code_indices: torch.Tensor) -> torch.Tensor:
        return self.embed_code(code_indices).transpose(1, 2)

    def decode_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        quantized_hidden_states = self.decode_code_wo_out_proj(code_indices).float()
        quantized_hidden_states = self.out_proj(quantized_hidden_states).float()
        return quantized_hidden_states

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match training LFQ: L2-normalize then argmin squared distance."""
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        codebook = self.codebook.weight.float()

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        distances = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).max(1)[1]
        indices = indices.reshape(latents.size(0), -1)
        quantized_hidden_states = self.decode_code_wo_out_proj(indices).float()
        return quantized_hidden_states, indices


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ (inference only)."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__()
        self.input_hidden_size = config.input_hidden_size
        self.hidden_size = config.hidden_size
        self.output_hidden_size = config.output_hidden_size
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.input_proj = nn.Conv1d(self.input_hidden_size, self.hidden_size, kernel_size=1)
        self.output_proj = nn.Conv1d(self.hidden_size, self.output_hidden_size, kernel_size=1)

        self.quantizers = nn.ModuleList([MossAudioTokenizerLFQ(config) for _ in range(self.n_codebooks)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_length: torch.Tensor,
        num_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference quantization."""
        hidden_states = self.input_proj(hidden_states).float()

        batch_size, _, max_time = hidden_states.shape
        mask = torch.arange(max_time, device=hidden_states.device).expand(
            batch_size, max_time
        ) < input_length.unsqueeze(1)

        quantized_out = torch.zeros_like(hidden_states, dtype=torch.float32)
        residual = hidden_states.clone().float()
        all_indices = []

        num_quantizers = num_quantizers or self.n_codebooks
        for index, quantizer in enumerate(self.quantizers):
            if index >= num_quantizers:
                break

            masked_residual = residual * mask.unsqueeze(1)
            quantized_hidden_states, indices, _ = quantizer(masked_residual)

            update_mask = mask.unsqueeze(1)
            quantized_out = quantized_out + quantized_hidden_states * update_mask
            residual = residual - quantized_hidden_states * update_mask
            all_indices.append(indices)

        all_indices = (
            torch.stack(all_indices)
            if all_indices
            else torch.empty(0, batch_size, max_time, device=hidden_states.device, dtype=torch.long)
        )
        quantized_out = self.output_proj(quantized_out)
        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        num_quantizers, batch_size, sequence_length = codes.shape
        embeddings = torch.zeros(
            batch_size, self.hidden_size, sequence_length, device=codes.device, dtype=torch.float32
        )
        for index, quantizer in enumerate(self.quantizers[:num_quantizers]):
            embeddings += quantizer.decode_code(codes[index]).float()
        embeddings = self.output_proj(embeddings)
        return embeddings


class MossAudioTokenizerEncoder(nn.Module):
    """MOSS Audio Tokenizer encoder."""

    def __init__(self, config: MossAudioTokenizerEncoderConfig):
        super().__init__()
        self.config = config
        stage_configs = config.transformer_configs
        for stage_config in stage_configs:
            stage_config._attn_implementation = config._attn_implementation

        self.layers = nn.ModuleList()
        for stage_index, sampling_ratio in enumerate(config.downsampling_ratios):
            self.layers.append(MossAudioTokenizerDownsample(sampling_ratio=sampling_ratio))
            self.layers.append(MossAudioTokenizerProjectedTransformer(stage_configs[stage_index]))

    def forward(self, hidden_states, input_lengths, past_key_values=None, use_cache=None, **kwargs):
        stage_index = 0
        for layer in self.layers:
            if isinstance(layer, MossAudioTokenizerProjectedTransformer):
                stage_cache = past_key_values[stage_index] if past_key_values is not None else None
                hidden_states, input_lengths = layer(
                    hidden_states, input_lengths, past_key_values=stage_cache, use_cache=use_cache, **kwargs
                )
                stage_index += 1
            else:
                hidden_states, input_lengths = layer(hidden_states, input_lengths)
        return hidden_states, input_lengths


class MossAudioTokenizerDecoder(nn.Module):
    """MOSS Audio Tokenizer decoder."""

    def __init__(self, config: MossAudioTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        stage_configs = config.transformer_configs
        for stage_config in stage_configs:
            stage_config._attn_implementation = config._attn_implementation

        self.layers = nn.ModuleList()
        for stage_index, sampling_ratio in enumerate(config.upsampling_ratios):
            self.layers.append(MossAudioTokenizerProjectedTransformer(stage_configs[stage_index]))
            self.layers.append(MossAudioTokenizerUpsample(sampling_ratio=sampling_ratio))

    def forward(self, hidden_states, input_lengths, past_key_values=None, use_cache=None, **kwargs):
        stage_index = 0
        for layer in self.layers:
            if isinstance(layer, MossAudioTokenizerProjectedTransformer):
                stage_cache = past_key_values[stage_index] if past_key_values is not None else None
                hidden_states, input_lengths = layer(
                    hidden_states, input_lengths, past_key_values=stage_cache, use_cache=use_cache, **kwargs
                )
                stage_index += 1
            else:
                hidden_states, input_lengths = layer(hidden_states, input_lengths)
        return hidden_states, input_lengths


@auto_docstring
class MossAudioTokenizerPreTrainedModel(PreTrainedAudioTokenizerBase):
    """Base class for MossAudioTokenizer models."""

    config_class = MossAudioTokenizerConfig
    base_model_prefix = ""
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _no_split_modules = [
        "MossAudioTokenizerTransformerLayer",
        "MossAudioTokenizerResidualLFQ",
    ]

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        for module in self.modules():
            if isinstance(module, nn.Conv1d) and not nn.utils.parametrize.is_parametrized(module, "weight"):
                weight_norm(module)

    def remove_weight_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


@auto_docstring(
    custom_intro="""
    The MossAudioTokenizer neural audio codec model for audio tokenization and synthesis.
    """
)
class MossAudioTokenizerModel(MossAudioTokenizerPreTrainedModel):
    """
    MossAudioTokenizer model for audio tokenization and synthesis.

    This model can encode audio waveforms into discrete tokens and decode
    tokens back into audio waveforms.
    """

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__(config)

        encoder_config = config.encoder_config
        encoder_config._attn_implementation = config._attn_implementation
        decoder_config = config.decoder_config
        decoder_config._attn_implementation = config._attn_implementation

        self.encoder = MossAudioTokenizerEncoder(encoder_config)
        self.quantizer = MossAudioTokenizerResidualLFQ(config.quantizer_config)
        self.decoder = MossAudioTokenizerDecoder(decoder_config)

        self.post_init()

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        past_key_values: list[Cache] | None = None,
        use_cache: bool | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        """Tokenize audio waveform into discrete tokens."""
        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        batch_size, _, sequence_length = input_values.shape
        device = input_values.device

        if input_lengths is None:
            input_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if sequence_length % self.config.hop_length != 0:
            raise ValueError(
                "`input_values` length must be a multiple of `config.hop_length`. "
                "Use `MossAudioTokenizerFeatureExtractor` to prepare and pad audio inputs."
            )

        encoder_hidden_states, encoder_hidden_states_lengths = self.encoder(
            input_values, input_lengths, past_key_values=past_key_values, use_cache=use_cache
        )

        _, audio_codes, audio_codes_lengths = self.quantizer(
            encoder_hidden_states, encoder_hidden_states_lengths, num_quantizers
        )

        audio_codes = audio_codes.transpose(0, 1).contiguous()
        code_positions = torch.arange(audio_codes.shape[-1], device=device)
        audio_codes_mask = code_positions[None, :] < audio_codes_lengths[:, None]

        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes,
            latents=encoder_hidden_states,
            audio_codes_mask=audio_codes_mask,
        )

    def _check_chunk_duration(self, chunk_duration: float):
        if chunk_duration <= 0:
            raise ValueError("`chunk_duration` must be > 0 when provided.")
        if chunk_duration > self.config.sliding_window_duration:
            raise ValueError(
                "`chunk_duration` must be <= `config.sliding_window_duration` "
                f"({self.config.sliding_window_duration}), got {chunk_duration}."
            )

        chunk_length = int(round(chunk_duration * self.config.sampling_rate))
        if chunk_length <= 0:
            raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
        if chunk_length % self.config.hop_length != 0:
            raise ValueError(
                "`chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`. "
                f"Got chunk_length={chunk_length}, hop_length={self.config.hop_length}."
            )
        return chunk_length

    @can_return_tuple
    @auto_docstring
    def encode(  # type: ignore[override]
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
        past_key_values: list[Cache] | None = None,
        use_cache: bool | None = None,
    ):
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate valid audio samples.
        num_quantizers (`int`, *optional*):
            Number of quantizers to use. By default, all quantizers are used.
        chunk_duration (`float`, *optional*):
            If provided, encode the input waveform in successive chunks of `chunk_duration` seconds while keeping a
            KV cache for the causal transformers.

            `chunk_duration` must be <= `config.sliding_window_duration`, and
            `chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`.
        past_key_values (`list[Cache]`, *optional*):
            KV caches, one per encoder transformer stage, updated in place when caching is enabled. Pass them to
            encode audio incrementally, e.g. for streaming. If `None` and caching is enabled, fresh caches are
            created.
        use_cache (`bool`, *optional*):
            Whether to use KV caching. Defaults to `True` when `past_key_values` are provided or when encoding in
            chunks via `chunk_duration`, and to `False` otherwise.

        Returns:
            `MossAudioTokenizerEncoderOutput` or tuple containing audio codes and lengths.
        """
        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        batch_size, _, sequence_length = input_values.shape
        device = input_values.device

        if padding_mask is not None:
            input_lengths = padding_mask.sum(dim=-1).long()
        else:
            input_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if chunk_duration is None:
            if use_cache is None and past_key_values is None:
                use_cache = False
            return self._encode_frame(
                input_values, input_lengths, num_quantizers, past_key_values=past_key_values, use_cache=use_cache
            )

        chunk_length = self._check_chunk_duration(chunk_duration)
        if int(input_lengths.max()) <= chunk_length:
            # Single-chunk input: honor explicitly passed caches (e.g. true streaming), otherwise run uncached.
            if use_cache is None and past_key_values is None:
                use_cache = False
            return self._encode_frame(
                input_values, input_lengths, num_quantizers, past_key_values=past_key_values, use_cache=use_cache
            )

        if past_key_values is None:
            past_key_values = [
                DynamicCache(config=stage_config) for stage_config in self.encoder.config.transformer_configs
            ]
        use_cache = True if use_cache is None else use_cache

        codes_chunks: list[torch.Tensor] = []
        latent_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        padded_length = input_values.shape[-1]
        for start_idx in range(0, padded_length, chunk_length):
            chunk_input_lengths = torch.clamp(input_lengths - start_idx, max=chunk_length)
            if int(chunk_input_lengths.max()) <= 0:
                break

            input_values_i = input_values[..., start_idx : start_idx + chunk_length]
            result_i = self._encode_frame(
                input_values_i,
                chunk_input_lengths,
                num_quantizers,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            audio_codes_mask_i = result_i.audio_codes_mask
            if audio_codes_mask_i is None:
                raise RuntimeError("Internal error: `_encode_frame` returned no `audio_codes_mask`.")

            keep_i = int(audio_codes_mask_i.sum(dim=-1).max())
            codes_chunks.append(result_i.audio_codes[:, :, :keep_i])
            latent_chunks.append(result_i.latents[:, :, :keep_i])
            mask_chunks.append(audio_codes_mask_i[:, :keep_i])

        return MossAudioTokenizerEncoderOutput(
            audio_codes=torch.cat(codes_chunks, dim=-1),
            latents=torch.cat(latent_chunks, dim=-1),
            audio_codes_mask=torch.cat(mask_chunks, dim=-1),
        )

    @can_return_tuple
    @auto_docstring
    def decode(  # type: ignore[override]
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        chunk_duration: float | None = None,
        num_quantizers: int | None = None,
        past_key_values: list[Cache] | None = None,
        use_cache: bool | None = None,
    ):
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`):
            Discrete code embeddings computed using `model.encode`.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate valid code positions.
        chunk_duration (`float`, *optional*):
            If provided, decode the input codes in successive chunks of `chunk_duration` seconds while keeping a
            KV cache for the causal transformers.
        num_quantizers (`int`, *optional*):
            Number of quantizers to use. By default, all quantizers in `audio_codes` are used.

            `chunk_duration` must be <= `config.sliding_window_duration`, and
            `chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`.
        past_key_values (`list[Cache]`, *optional*):
            KV caches, one per decoder transformer stage, updated in place when caching is enabled. Pass them to
            decode codes incrementally, e.g. for streaming. If `None` and caching is enabled, fresh caches are
            created.
        use_cache (`bool`, *optional*):
            Whether to use KV caching. Defaults to `True` when `past_key_values` are provided or when decoding in
            chunks via `chunk_duration`, and to `False` otherwise.

        Returns:
            `MossAudioTokenizerDecoderOutput` or tuple containing decoded audio.
        """
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(1)
        elif audio_codes.dim() == 3:
            audio_codes = audio_codes.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "`audio_codes` must have shape `(num_quantizers, sequence_length)` or `(batch_size, num_quantizers, sequence_length)`."
            )

        if num_quantizers is not None:
            if num_quantizers > audio_codes.shape[0]:
                raise ValueError(
                    f"`num_quantizers` ({num_quantizers}) must be <= audio_codes.shape[0] ({audio_codes.shape[0]})."
                )
            audio_codes = audio_codes[:num_quantizers]

        _, batch_size, sequence_length = audio_codes.shape
        device = audio_codes.device

        if padding_mask is not None:
            codes_lengths = padding_mask.sum(dim=-1).long()
        else:
            codes_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if chunk_duration is None:
            if use_cache is None and past_key_values is None:
                use_cache = False
            audio_values = self._decode_frames(
                audio_codes, codes_lengths, past_key_values=past_key_values, use_cache=use_cache
            )
        else:
            chunk_length = self._check_chunk_duration(chunk_duration)
            chunk_frame_length = chunk_length // self.config.hop_length
            if int(codes_lengths.max()) <= chunk_frame_length:
                # Single-chunk input: honor explicitly passed caches (e.g. true streaming), otherwise run uncached.
                if use_cache is None and past_key_values is None:
                    use_cache = False
                audio_values = self._decode_frames(
                    audio_codes[..., : int(codes_lengths.max())],
                    codes_lengths,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            else:
                if past_key_values is None:
                    past_key_values = [
                        DynamicCache(config=stage_config) for stage_config in self.decoder.config.transformer_configs
                    ]
                use_cache = True if use_cache is None else use_cache

                wav_chunks: list[torch.Tensor] = []
                for start_idx in range(0, int(codes_lengths.max()), chunk_frame_length):
                    chunk_codes_lengths = torch.clamp(codes_lengths - start_idx, max=chunk_frame_length)
                    if int(chunk_codes_lengths.max()) <= 0:
                        break

                    codes_i = audio_codes[..., start_idx : start_idx + chunk_frame_length]
                    audio_i = self._decode_frames(
                        codes_i,
                        chunk_codes_lengths,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                    keep_i = int(chunk_codes_lengths.max()) * self.config.hop_length
                    wav_chunks.append(audio_i[:, :, :keep_i])

                audio_values = torch.cat(wav_chunks, dim=-1)

        audio_lengths = codes_lengths * self.config.hop_length
        audio_positions = torch.arange(audio_values.shape[-1], device=device)
        audio_mask = audio_positions[None, :] < audio_lengths[:, None]
        return MossAudioTokenizerDecoderOutput(audio_values=audio_values, audio_mask=audio_mask)

    def _decode_frames(
        self,
        audio_codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        past_key_values: list[Cache] | None = None,
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        """Decode discrete tokens into audio waveform."""
        decoder_hidden_states = self.quantizer.decode_codes(audio_codes)
        audio_values, _ = self.decoder(
            decoder_hidden_states, codes_lengths, past_key_values=past_key_values, use_cache=use_cache
        )
        return audio_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        audio_codes: torch.Tensor | None = None,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerOutput:  # type: ignore[override]
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Raw audio input converted to Float.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid computing on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`.
        num_quantizers (`int`, *optional*):
            Number of quantizers (codebooks) to use. By default, all quantizers are used.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import MossAudioTokenizerModel

        >>> model = MossAudioTokenizerModel.from_pretrained("moss_audio_tokenizer-model")

        >>> # Create dummy audio input
        >>> audio = torch.randn(1, 1, 24000)  # 1 second of audio at 24kHz

        >>> outputs = model(input_values=audio)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```
        """
        output_audio_codes: torch.Tensor | None = None
        output_audio_codes_mask: torch.Tensor | None = None
        output_latents: torch.Tensor | None = None
        output_audio_values: torch.Tensor | None = None
        decoded_from_encoded_codes = False

        if input_values is not None:
            encoder_output = self.encode(input_values, padding_mask, num_quantizers, return_dict=True)
            output_audio_codes = encoder_output.audio_codes
            output_audio_codes_mask = encoder_output.audio_codes_mask
            output_latents = encoder_output.latents

            if audio_codes is None:
                audio_codes = output_audio_codes
                decoded_from_encoded_codes = True

        if audio_codes is not None:
            audio_codes_padding_mask = output_audio_codes_mask if decoded_from_encoded_codes else padding_mask
            decoder_output = self.decode(
                audio_codes,
                padding_mask=audio_codes_padding_mask,
                num_quantizers=num_quantizers,
                return_dict=True,
            )
            output_audio_values = decoder_output.audio_values

        return MossAudioTokenizerOutput(
            audio_values=output_audio_values,
            audio_codes=output_audio_codes,
            latents=output_latents,
            audio_codes_mask=output_audio_codes_mask,
        )


class MossAudioTokenizerFeatureExtractor(DacFeatureExtractor):
    r"""
    Constructs an MossAudioTokenizer feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used for padding.
        hop_length (`int`, *optional*, defaults to 1920):
            Overlap length between successive windows.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        hop_length: int = 1920,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            hop_length=hop_length,
            **kwargs,
        )


__all__ = [
    "MossAudioTokenizerDecoder",
    "MossAudioTokenizerEncoder",
    "MossAudioTokenizerFeatureExtractor",
    "MossAudioTokenizerModel",
    "MossAudioTokenizerPreTrainedModel",
]
