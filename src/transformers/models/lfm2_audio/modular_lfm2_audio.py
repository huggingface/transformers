# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoModel
from .configuration_lfm2_audio import Lfm2AudioConfig, Lfm2AudioDepthConfig, Lfm2Config


TEXT_MODALITY = 1
AUDIO_INPUT_MODALITY = 2
AUDIO_OUTPUT_MODALITY = 3


@auto_docstring(custom_intro="Base class for LFM2-Audio backbone outputs.")
@dataclass
class Lfm2AudioModelOutputWithPast(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Hidden states returned by the LFM2 backbone.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
        Pre-computed key and value states used for autoregressive decoding.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the LFM2 backbone.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of the LFM2 backbone.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        FastConformer features after projection to the LFM2 hidden size.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(custom_intro="Base class for LFM2-Audio conditional-generation outputs.")
@dataclass
class Lfm2AudioConditionalGenerationOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*, returned when labels are provided):
        Combined text and audio loss.
    text_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
        Next-token text loss.
    audio_loss (`torch.FloatTensor`, *optional*, returned when `audio_labels` is provided):
        Weighted next-codebook audio loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
        Text-token prediction scores.
    audio_logits (`torch.FloatTensor` of shape `(audio_tokens, codebooks, audio_vocab_size)`, *optional*):
        Audio-codebook prediction scores for supervised audio positions.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
        Pre-computed key and value states used for autoregressive decoding.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the LFM2 backbone.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of the LFM2 backbone.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        FastConformer features after projection to the LFM2 hidden size.
    """

    loss: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None
    audio_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    audio_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(custom_intro="Base class for LFM2-Audio generation outputs.")
@dataclass
class Lfm2AudioGenerateOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(1, generated_text_length)`):
        Generated text tokens.
    audio_codes (`torch.LongTensor` of shape `(1, codebooks, generated_audio_length)`):
        Generated Mimi codes, including an end-of-audio frame when one was sampled.
    modalities (`torch.LongTensor` of shape `(1, generation_steps)`):
        Modality generated at each step: 1 for text and 3 for audio output.
    """

    sequences: torch.LongTensor | None = None
    audio_codes: torch.LongTensor | None = None
    modalities: torch.LongTensor | None = None


class Lfm2AudioRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).to(input_dtype)


class Lfm2AudioSharedEmbedding(nn.Module):
    """Embedding and normalized output projection, optionally sharing their weights."""

    def __init__(self, hidden_size: int, vocab_size: int, tie_embeddings: bool, norm_eps: float = 1e-5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding_norm = Lfm2AudioRMSNorm(hidden_size, eps=norm_eps)
        self.to_logits = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings:
            self.to_logits.weight = self.embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.to_logits(self.embedding_norm(hidden_states))


class Lfm2AudioCodebookEmbedding(nn.Module):
    """Average embeddings from the eight generated audio codebooks."""

    def __init__(self, hidden_size: int = 512, codebooks: int = 8, vocab_size: int = 2048):
        super().__init__()
        self.emb = nn.Embedding(codebooks * vocab_size, hidden_size)
        self.codebooks = codebooks
        self.vocab_size = vocab_size

    def forward(self, audio_codes: torch.LongTensor) -> torch.FloatTensor:
        offsets = torch.arange(self.codebooks, device=audio_codes.device) * self.vocab_size
        offset_codes = audio_codes + offsets[None, :, None]
        return self.emb(offset_codes).mean(dim=1)


class Lfm2AudioInverseShortTimeFourierTransform(nn.Module):
    """Inverse STFT with same padding for the bundled LFM audio detokenizer."""

    def __init__(self, n_fft: int = 1280, hop_length: int = 320, win_length: int = 1280):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spectrogram: torch.Tensor) -> torch.FloatTensor:
        if spectrogram.ndim != 3:
            raise ValueError("`spectrogram` must have shape `(batch_size, frequency_bins, frames)`.")

        num_frames = spectrogram.shape[-1]
        padding = (self.win_length - self.hop_length) // 2
        inverse = torch.fft.irfft(spectrogram, self.n_fft, dim=1, norm="backward")
        inverse = inverse * self.window[None, :, None]

        output_size = (num_frames - 1) * self.hop_length + self.win_length
        waveform = F.fold(
            inverse,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, padding:-padding]

        squared_window = self.window.square().expand(1, num_frames, -1).transpose(1, 2)
        window_envelope = F.fold(
            squared_window,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, padding:-padding]
        if not torch.all(window_envelope > 1e-11):
            raise ValueError("The inverse STFT window envelope contains zeros.")
        return waveform / window_envelope


@auto_docstring(custom_intro="LFM2-based audio detokenizer bundled with LFM2.5-Audio checkpoints.")
class Lfm2AudioDetokenizer(PreTrainedModel):
    """Decode LFM2-Audio codebooks with the detokenizer bundled in LFM2.5-Audio checkpoints."""

    config_class = Lfm2Config
    main_input_name = "audio_codes"
    _supports_sdpa = True

    def __init__(self, config: Lfm2Config):
        config = copy.deepcopy(config)
        # The released detokenizer predates the canonical LFM2 layer name. Its explicit sliding-window mask below
        # preserves the original behavior while the native LFM2 layer uses its supported attention implementation.
        config.layer_types = [
            "full_attention" if layer_type == "sliding_attention" else layer_type for layer_type in config.layer_types
        ]
        super().__init__(config)
        self.emb = Lfm2AudioCodebookEmbedding(hidden_size=config.hidden_size)
        self.lfm = AutoModel.from_config(config)
        self.lin = nn.Linear(config.hidden_size, config.output_size)
        self.istft = Lfm2AudioInverseShortTimeFourierTransform()
        self.sliding_window_size = getattr(config, "sliding_window", 30)
        self.post_init()

    @auto_docstring
    def forward(self, audio_codes: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]) -> torch.FloatTensor:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, 8, audio_timesteps)`):
            Generated audio codebooks with values in the range `[0, 2047]`.
        """
        if audio_codes.ndim != 3 or audio_codes.shape[1] != self.emb.codebooks:
            raise ValueError("`audio_codes` must have shape `(batch_size, 8, audio_timesteps)`.")
        if torch.any((audio_codes < 0) | (audio_codes >= self.emb.vocab_size)):
            raise ValueError("Audio codes must be in the range [0, 2047].")

        hidden_states = self.emb(audio_codes)
        hidden_states = F.interpolate(
            hidden_states.transpose(1, 2), size=6 * hidden_states.shape[1], mode="nearest-exact"
        ).transpose(1, 2)

        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        relative_positions = positions - positions[:, None]
        attention_mask = ((relative_positions <= 0) & (relative_positions > -self.sliding_window_size))[None, None]
        hidden_states = self.lfm(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        spectrogram = self.lin(hidden_states).transpose(1, 2).contiguous()
        log_magnitude, phase = torch.chunk(spectrogram, 2, dim=1)
        return self.istft(torch.polar(log_magnitude.exp(), phase))


class Lfm2AudioDepthMLP(nn.Module):
    def __init__(self, config: Lfm2AudioDepthConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


def _apply_depth_rotary(
    query_states: torch.Tensor, key_states: torch.Tensor, frequencies: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query_states.dtype
    key_dtype = key_states.dtype
    query_complex = torch.view_as_complex(query_states.float().reshape(*query_states.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key_states.float().reshape(*key_states.shape[:-1], -1, 2))
    frequencies = frequencies.view(1, frequencies.shape[0], 1, frequencies.shape[1])
    query_states = torch.view_as_real(query_complex * frequencies).flatten(-2)
    key_states = torch.view_as_real(key_complex * frequencies).flatten(-2)
    return query_states.to(query_dtype), key_states.to(key_dtype)


class Lfm2AudioDepthAttentionCore(nn.Module):
    def __init__(self, config: Lfm2AudioDepthConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.dim // config.num_attention_heads
        self.q_layernorm = Lfm2AudioRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2AudioRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        frequencies: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, sequence_length = query_states.shape[:2]
        query_states = query_states.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)
        query_states, key_states = _apply_depth_rotary(query_states, key_states, frequencies)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        present_key_value = (key_states, value_states)

        query = query_states.transpose(1, 2)
        key = key_states.transpose(1, 2)
        value = value_states.transpose(1, 2)
        query_length, key_length = query.shape[-2], key.shape[-2]

        attention_mask = None
        is_causal = query_length == key_length
        if not is_causal and query_length > 1:
            past_length = key_length - query_length
            query_positions = torch.arange(query_length, device=query.device) + past_length
            key_positions = torch.arange(key_length, device=query.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=is_causal,
            enable_gqa=self.num_attention_heads != self.num_key_value_heads,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return hidden_states, present_key_value


class Lfm2AudioDepthAttention(nn.Module):
    def __init__(self, config: Lfm2AudioDepthConfig):
        super().__init__()
        self.hidden_size = config.dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.dim // config.num_attention_heads
        kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = nn.Linear(config.dim, config.dim + 2 * kv_size, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.bounded_attention = Lfm2AudioDepthAttentionCore(config)
        self.rope_theta = config.rope_theta
        self.frequencies: torch.Tensor | None = None

    def get_frequencies(self, device: torch.device) -> torch.Tensor:
        # Initialize lazily because `from_pretrained` may construct the module on the meta device. Keep this complex
        # tensor outside module buffers so a later bfloat16 cast cannot discard its imaginary component.
        if self.frequencies is None or self.frequencies.device != device:
            frequencies = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
            positions = torch.arange(64, dtype=torch.float32, device=device)
            self.frequencies = torch.polar(
                torch.ones((64, self.head_dim // 2), device=device), positions[:, None] * frequencies
            )
        return self.frequencies

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        projected_states = self.qkv_proj(hidden_states)
        kv_size = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = projected_states.split([self.hidden_size, kv_size, kv_size], dim=-1)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        frequencies = self.get_frequencies(projected_states.device)[past_length : past_length + hidden_states.shape[1]]
        attention_output, present_key_value = self.bounded_attention(
            query_states, key_states, value_states, frequencies, past_key_value
        )
        return self.out_proj(attention_output), present_key_value if use_cache else None


class Lfm2AudioDepthLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2AudioDepthConfig):
        super().__init__()
        self.operator = Lfm2AudioDepthAttention(config)
        self.feed_forward = Lfm2AudioDepthMLP(config)
        self.operator_norm = Lfm2AudioRMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = Lfm2AudioRMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_output, present_key_value = self.operator(
            self.operator_norm(hidden_states), past_key_value=past_key_value, use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, present_key_value


class Lfm2AudioDepthformer(nn.Module):
    def __init__(self, config: Lfm2AudioDepthConfig):
        super().__init__()
        self.layers = nn.ModuleList([Lfm2AudioDepthLayer(config) for _ in range(config.layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_cache = [] if use_cache else None
        for layer, past_key_value in zip(self.layers, past_key_values, strict=True):
            hidden_states, present_key_value = layer(hidden_states, past_key_value=past_key_value, use_cache=use_cache)
            if next_cache is not None:
                next_cache.append(present_key_value)
        return hidden_states, next_cache


@auto_docstring
class Lfm2AudioPreTrainedModel(PreTrainedModel):
    config: Lfm2AudioConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm2AudioDepthLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [r"codebook_offsets$", r"audio_loss_weights$"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False


@auto_docstring
class Lfm2AudioModel(Lfm2AudioPreTrainedModel):
    def __init__(self, config: Lfm2AudioConfig):
        super().__init__(config)
        encoder_config = config.encoder_config
        text_config = config.text_config
        depth_config = config.depth_config
        self._tied_weights_keys = {}
        self.lfm = AutoModel.from_config(text_config)
        # CODEPATH: The released Liquid Audio loader casts the LFM2 rotary buffers together with the model. Preserve
        # that checkpoint behavior when an explicit model dtype is requested so greedy audio codes remain identical.
        if config.dtype is not None:
            # CODEPATH: Native saved checkpoints may deserialize dtype as a string; raw Hub configs use torch.dtype.
            model_dtype = getattr(torch, config.dtype) if isinstance(config.dtype, str) else config.dtype
            self.lfm.rotary_emb.to(dtype=model_dtype)
        self.conformer = AutoModel.from_config(encoder_config)
        # The released FastConformer uses its eager relative-position attention path. Matching it avoids small
        # backend-dependent differences that can be amplified by autoregressive audio sampling.
        self.conformer.set_attn_implementation("eager")
        self.audio_adapter_norm = nn.LayerNorm(encoder_config.hidden_size)
        self.audio_adapter_linear_1 = nn.Linear(encoder_config.hidden_size, text_config.hidden_size)
        self.audio_adapter_linear_2 = nn.Linear(text_config.hidden_size, text_config.hidden_size)

        self.audio_embedding = Lfm2AudioSharedEmbedding(
            text_config.hidden_size,
            config.audio_vocab_size * config.codebooks,
            tie_embeddings=config.tie_audio_embeddings,
        )
        self.depthformer = Lfm2AudioDepthformer(depth_config)
        self.depth_linear = nn.Linear(text_config.hidden_size, depth_config.dim * config.codebooks)
        self.depth_embeddings = nn.ModuleList(
            [
                Lfm2AudioSharedEmbedding(
                    depth_config.dim,
                    config.audio_vocab_size,
                    tie_embeddings=depth_config.tie,
                )
                for _ in range(config.codebooks)
            ]
        )
        # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses untied audio embeddings; custom configs may tie them.
        if config.tie_audio_embeddings:
            self._tied_weights_keys["audio_embedding.to_logits.weight"] = "audio_embedding.embedding.weight"
        # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B ties every depth-codebook input and output embedding.
        if depth_config.tie:
            for codebook_idx in range(config.codebooks):
                self._tied_weights_keys[f"depth_embeddings.{codebook_idx}.to_logits.weight"] = (
                    f"depth_embeddings.{codebook_idx}.embedding.weight"
                )

        self.post_init()

    def get_input_embeddings(self):
        return self.lfm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.lfm.set_input_embeddings(value)

    def get_codebook_offsets(self, device: torch.device) -> torch.LongTensor:
        return torch.arange(self.config.codebooks, device=device) * self.config.audio_vocab_size

    def get_audio_loss_weights(self, device: torch.device) -> torch.FloatTensor:
        # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses logarithmic codebook weights; legacy configs may use linear weights.
        if self.config.codebook_weight == "log":
            return (
                torch.linspace(1, 0, self.config.codebooks, device=device)
                * math.log(self.config.semantic_codebook_factor)
            ).exp()
        weights = torch.ones(self.config.codebooks, device=device)
        weights[0] *= self.config.semantic_codebook_factor
        return weights

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_attention_mask: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor]:
        r"""
        input_features_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask indicating valid log-mel frames before FastConformer subsampling.
        """
        encoder_parameter = next(self.conformer.parameters())
        input_features = input_features.to(device=encoder_parameter.device, dtype=encoder_parameter.dtype)
        # NeMo constructs these frequencies with exp(arange * -log(base) / dim). The algebraically equivalent power
        # form differs by a few bf16 ULPs and can change sampled audio codes, so reproduce the checkpoint formula.
        position_encoder = self.conformer.encode_positions
        legacy_inv_freq = torch.exp(
            torch.arange(
                0,
                self.config.encoder_config.hidden_size,
                2,
                dtype=torch.float32,
                device=encoder_parameter.device,
            )
            * -(math.log(10_000.0) / self.config.encoder_config.hidden_size)
        )
        position_encoder.inv_freq.copy_(legacy_inv_freq)
        if input_features_attention_mask is not None:
            input_features_attention_mask = input_features_attention_mask.to(encoder_parameter.device)
        encoder_outputs = self.conformer(
            input_features=input_features,
            attention_mask=input_features_attention_mask,
            output_attention_mask=True,
            return_dict=True,
        )
        audio_features = self.audio_adapter_norm(encoder_outputs.last_hidden_state)
        audio_features = self.audio_adapter_linear_1(audio_features)
        audio_features = self.audio_adapter_linear_2(F.gelu(audio_features))
        if encoder_outputs.attention_mask is None:
            audio_mask = torch.ones(audio_features.shape[:2], dtype=torch.bool, device=audio_features.device)
        else:
            audio_mask = encoder_outputs.attention_mask.bool()
        return audio_features, audio_mask

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        input_features: torch.FloatTensor | None,
        input_features_attention_mask: torch.Tensor | None,
        modality_ids: torch.LongTensor | None,
        audio_codes: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor | None]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide either `input_ids` or `inputs_embeds`.")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        elif input_ids is not None:
            raise ValueError("You cannot provide both `input_ids` and `inputs_embeds`.")

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        if modality_ids is None:
            modality_ids = torch.full_like(attention_mask, TEXT_MODALITY)
            if input_ids is not None:
                audio_input_mask = (input_ids == self.config.audio_token_id) & attention_mask.bool()
                modality_ids = modality_ids.masked_fill(audio_input_mask, AUDIO_INPUT_MODALITY)
            modality_ids = modality_ids.masked_fill(~attention_mask.bool(), 0)

        audio_hidden_states = None
        audio_input_mask = modality_ids == AUDIO_INPUT_MODALITY
        if input_features is not None:
            audio_features, encoded_audio_mask = self.get_audio_features(
                input_features, input_features_attention_mask=input_features_attention_mask
            )
            audio_hidden_states = audio_features[encoded_audio_mask]
            expected = int(audio_input_mask.sum().item())
            if expected != audio_hidden_states.shape[0]:
                raise ValueError(
                    f"Audio features and placeholder tokens do not match: {audio_hidden_states.shape[0]} features "
                    f"for {expected} placeholders."
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_input_mask.unsqueeze(-1), audio_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        elif audio_input_mask.any():
            raise ValueError("Audio placeholder tokens were provided without `input_features`.")

        audio_output_mask = modality_ids == AUDIO_OUTPUT_MODALITY
        if audio_output_mask.any():
            if audio_codes is None:
                raise ValueError("`audio_codes` are required for audio-output positions in the prompt.")
            # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses exactly 8 Mimi codebooks at every audio-output position.
            if audio_codes.ndim != 3 or audio_codes.shape[1] != self.config.codebooks:
                raise ValueError("`audio_codes` must have shape `(batch_size, codebooks, audio_timesteps)`.")

            output_embeddings = []
            for batch_idx in range(inputs_embeds.shape[0]):
                num_audio_steps = int(audio_output_mask[batch_idx].sum().item())
                codes = audio_codes[batch_idx, :, :num_audio_steps]
                offset_codes = codes + self.get_codebook_offsets(codes.device)[:, None]
                output_embeddings.append(self.audio_embedding(offset_codes).sum(0))
            output_embeddings = torch.cat(output_embeddings, dim=0)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_output_mask.unsqueeze(-1), output_embeddings.to(inputs_embeds.dtype)
            )

        return inputs_embeds, modality_ids, audio_hidden_states

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_attention_mask: torch.Tensor | None = None,
        modality_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Lfm2AudioModelOutputWithPast:
        r"""
        input_features_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask indicating valid log-mel frames before FastConformer subsampling.
        modality_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Modality of each prompt position: 1 for text, 2 for audio input, and 3 for audio output.
        audio_codes (`torch.LongTensor` of shape `(batch_size, codebooks, audio_timesteps)`, *optional*):
            Mimi codebooks used at audio-output positions in the prompt.
        """
        inputs_embeds, _, audio_hidden_states = self._prepare_inputs_embeds(
            input_ids,
            attention_mask,
            input_features,
            input_features_attention_mask,
            modality_ids,
            audio_codes,
            inputs_embeds,
        )
        outputs = self.lfm(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return Lfm2AudioModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_hidden_states,
        )


@auto_docstring
class Lfm2AudioForConditionalGeneration(Lfm2AudioPreTrainedModel, GenerationMixin):
    def __init__(self, config: Lfm2AudioConfig):
        super().__init__(config)
        self.model = Lfm2AudioModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def compute_audio_logits(
        self, hidden_states: torch.FloatTensor, audio_labels: torch.LongTensor
    ) -> torch.FloatTensor:
        """Teacher-force codebooks and return logits with shape `(tokens, codebooks, audio_vocab_size)`."""
        if hidden_states.ndim != 2:
            raise ValueError("`hidden_states` must have shape `(tokens, hidden_size)`.")
        # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B predicts 8 codebooks for every supervised audio position.
        if audio_labels.shape != (hidden_states.shape[0], self.config.codebooks):
            raise ValueError("`audio_labels` must have shape `(tokens, codebooks)`.")

        depth_inputs = self.model.depth_linear(hidden_states).view(
            hidden_states.shape[0], self.config.codebooks, self.config.depth_config.dim
        )
        label_embeddings = torch.stack(
            [embedding(audio_labels[:, idx]) for idx, embedding in enumerate(self.model.depth_embeddings)], dim=1
        )
        previous_embeddings = torch.zeros_like(label_embeddings)
        previous_embeddings[:, 1:] = label_embeddings[:, :-1]
        depth_outputs, _ = self.model.depthformer(depth_inputs + previous_embeddings)
        return torch.stack(
            [embedding.get_logits(depth_outputs[:, idx]) for idx, embedding in enumerate(self.model.depth_embeddings)],
            dim=1,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_attention_mask: torch.Tensor | None = None,
        modality_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Lfm2AudioConditionalGenerationOutput:
        r"""
        input_features_attention_mask (`torch.LongTensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask indicating valid log-mel frames before FastConformer subsampling.
        modality_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Modality of each prompt position: 1 for text, 2 for audio input, and 3 for audio output.
        audio_codes (`torch.LongTensor` of shape `(batch_size, codebooks, audio_timesteps)`, *optional*):
            Mimi codebooks used at audio-output positions in the prompt.
        audio_labels (`torch.LongTensor` of shape `(batch_size, sequence_length, codebooks)`, *optional*):
            Target Mimi codes. Positions containing a negative value are ignored.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            input_features_attention_mask=input_features_attention_mask,
            modality_ids=modality_ids,
            audio_codes=audio_codes,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = F.linear(hidden_states[:, slice_indices], self.get_output_embeddings().weight)

        text_loss = None
        text_tokens = 0
        if labels is not None:
            text_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            target_labels = F.pad(labels, (0, 1), value=-100).roll(shifts=-1, dims=-1)
            target_labels = target_labels.narrow(-1, 0, labels.shape[-1])
            text_tokens = (target_labels[:, slice_indices] != -100).sum()

        audio_loss = None
        audio_logits = None
        audio_tokens = 0
        if audio_labels is not None:
            if audio_labels.ndim != 3 or audio_labels.shape[:2] != hidden_states.shape[:2]:
                raise ValueError("`audio_labels` must have shape `(batch_size, sequence_length, codebooks)`.")
            shifted_audio_labels = audio_labels[:, 1:]
            valid_audio = (shifted_audio_labels >= 0).all(-1)
            selected_labels = shifted_audio_labels[valid_audio]
            audio_tokens = selected_labels.shape[0]
            if audio_tokens:
                audio_logits = self.compute_audio_logits(hidden_states[:, :-1][valid_audio], selected_labels)
                per_codebook_loss = F.cross_entropy(
                    audio_logits.flatten(0, 1), selected_labels.flatten(), reduction="none"
                ).view(-1, self.config.codebooks)
                weights = self.model.get_audio_loss_weights(per_codebook_loss.device).to(per_codebook_loss)
                audio_loss = (per_codebook_loss * weights).sum(-1).mean() / weights.sum()
            else:
                audio_logits = hidden_states.new_empty((0, self.config.codebooks, self.config.audio_vocab_size))
                audio_loss = hidden_states.sum() * 0.0

        loss = None
        if text_loss is not None and audio_loss is not None:
            weighted_tokens = (
                self.config.text_loss_multiplier * text_tokens + self.config.audio_loss_multiplier * audio_tokens
            )
            loss = (
                self.config.text_loss_multiplier * text_loss * text_tokens
                + self.config.audio_loss_multiplier * audio_loss * audio_tokens
            ) / weighted_tokens.clamp_min(1e-6)
        elif text_loss is not None:
            loss = text_loss
        elif audio_loss is not None:
            loss = audio_loss

        return Lfm2AudioConditionalGenerationOutput(
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            logits=logits,
            audio_logits=audio_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float | None, top_k: int | None) -> torch.LongTensor:
        if temperature is None or temperature <= 0 or top_k == 1:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -float("inf"))
        return torch.multinomial(logits.softmax(dim=-1), num_samples=1)

    def _sample_audio_frame(
        self,
        hidden_state: torch.FloatTensor,
        temperature: float | None,
        top_k: int | None,
    ) -> torch.LongTensor:
        depth_inputs = self.model.depth_linear(hidden_state).view(self.config.codebooks, self.config.depth_config.dim)
        previous_embedding = torch.zeros_like(depth_inputs[0])
        past_key_values = None
        output_tokens = []
        for codebook_idx in range(self.config.codebooks):
            current_input = depth_inputs[codebook_idx] + previous_embedding
            depth_output, past_key_values = self.model.depthformer(
                current_input[None, None], past_key_values=past_key_values, use_cache=True
            )
            logits = self.model.depth_embeddings[codebook_idx].get_logits(depth_output[0, -1])
            next_token = self._sample(logits, temperature=temperature, top_k=top_k)
            output_tokens.append(next_token)
            previous_embedding = self.model.depth_embeddings[codebook_idx](next_token).squeeze(0)
        return torch.cat(output_tokens)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_attention_mask: torch.Tensor | None = None,
        modality_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        max_new_tokens: int = 256,
        generation_mode: str = "sequential",
        text_temperature: float | None = None,
        text_top_k: int | None = None,
        audio_temperature: float | None = None,
        audio_top_k: int | None = None,
        **kwargs,
    ) -> Lfm2AudioGenerateOutput:
        """Generate text tokens and 8-codebook audio frames from a single prompt."""
        if generation_mode not in {"sequential", "interleaved"}:
            raise ValueError("`generation_mode` must be either 'sequential' or 'interleaved'.")

        input_embeddings, _, _ = self.model._prepare_inputs_embeds(
            input_ids,
            attention_mask,
            input_features,
            input_features_attention_mask,
            modality_ids,
            audio_codes,
            inputs_embeds,
        )
        if input_embeddings.shape[0] != 1:
            raise ValueError("LFM2-Audio generation currently supports a batch size of 1.")

        current_input = input_embeddings
        current_modality = TEXT_MODALITY
        modality_left = self.config.interleaved_n_text
        text_done = False
        past_key_values = None
        generated_text = []
        generated_audio = []
        generated_modalities = []
        generation_attention_mask = attention_mask
        if generation_attention_mask is not None and bool(generation_attention_mask.bool().all()):
            generation_attention_mask = None

        for _ in range(max_new_tokens):
            outputs = self.model.lfm(
                inputs_embeds=current_input,
                attention_mask=generation_attention_mask if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            hidden_state = outputs.last_hidden_state[0, -1]
            past_key_values = outputs.past_key_values

            if generation_mode == "interleaved":
                modality_left -= 1

            if current_modality == TEXT_MODALITY:
                text_logits = F.linear(hidden_state, self.get_output_embeddings().weight)
                next_token = self._sample(text_logits, temperature=text_temperature, top_k=text_top_k)
                token_id = int(next_token.item())
                # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B treats interleaved EOS as a non-yielded stream terminator.
                if generation_mode == "interleaved" and token_id == self.config.eos_token_id:
                    break
                generated_text.append(next_token)
                generated_modalities.append(TEXT_MODALITY)

                # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 7 (`<|im_end|>`) to finish a response.
                if token_id == self.config.eos_token_id:
                    break
                # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 128 to switch sequential generation to audio.
                if generation_mode == "sequential" and token_id == self.config.audio_start_token_id:
                    current_modality = AUDIO_OUTPUT_MODALITY
                elif generation_mode == "interleaved":
                    # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 130 to mark the end of interleaved text.
                    if token_id == self.config.text_end_token_id:
                        text_done = True
                    if modality_left == 0 or text_done:
                        current_modality = AUDIO_OUTPUT_MODALITY
                        modality_left = self.config.interleaved_n_audio
                current_input = self.get_input_embeddings()(next_token)[None]
            else:
                next_frame = self._sample_audio_frame(hidden_state, temperature=audio_temperature, top_k=audio_top_k)
                # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses Mimi code 2048 to mark the end of generated audio.
                if int(next_frame[0].item()) == self.config.audio_eos_token_id:
                    next_frame = torch.full_like(next_frame, self.config.audio_eos_token_id)
                    current_modality = TEXT_MODALITY
                elif generation_mode == "interleaved" and modality_left == 0 and not text_done:
                    current_modality = TEXT_MODALITY
                    modality_left = self.config.interleaved_n_text

                generated_audio.append(next_frame)
                generated_modalities.append(AUDIO_OUTPUT_MODALITY)
                offset_frame = next_frame + self.model.get_codebook_offsets(next_frame.device)
                current_input = self.model.audio_embedding(offset_frame).sum(0)[None, None]

        if generated_text:
            sequences = torch.cat(generated_text).unsqueeze(0)
        else:
            sequences = torch.empty((1, 0), dtype=torch.long, device=input_embeddings.device)
        if generated_audio:
            output_audio_codes = torch.stack(generated_audio, dim=-1).unsqueeze(0)
        else:
            output_audio_codes = torch.empty(
                (1, self.config.codebooks, 0), dtype=torch.long, device=input_embeddings.device
            )
        modalities = torch.tensor(generated_modalities, dtype=torch.long, device=input_embeddings.device).unsqueeze(0)
        return Lfm2AudioGenerateOutput(
            sequences=sequences,
            audio_codes=output_audio_codes,
            modalities=modalities,
        )


__all__ = [
    "Lfm2AudioConditionalGenerationOutput",
    "Lfm2AudioDetokenizer",
    "Lfm2AudioForConditionalGeneration",
    "Lfm2AudioGenerateOutput",
    "Lfm2AudioModel",
    "Lfm2AudioModelOutputWithPast",
    "Lfm2AudioPreTrainedModel",
]
