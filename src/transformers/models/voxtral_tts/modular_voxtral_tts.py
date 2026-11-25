# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ..auto import AutoModel
from ..mistral.modeling_mistral import MistralAttention, MistralMLP, MistralRMSNorm, eager_attention_forward
from .configuration_voxtral_tts import (
    VoxtralTtsAcousticTransformerConfig,
    VoxtralTtsCodecConfig,
    VoxtralTtsConfig,
)
from .generation_voxtral_tts import VoxtralTtsGenerationMixin


# Special tokens of every audio codebook, predicted by the semantic head
EMPTY_AUDIO_TOKEN = 0
END_AUDIO_TOKEN = 1


@auto_docstring(custom_intro="Output of the Voxtral TTS codec decoder.")
@dataclass
class VoxtralTtsCodecDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Decoded waveform.
    """

    audio_values: torch.FloatTensor | None = None


@auto_docstring(custom_intro="Output of the Voxtral TTS model.")
@dataclass
class VoxtralTtsOutputWithPast(BaseModelOutputWithPast):
    r"""
    cache (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
    semantic_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, padded_semantic_vocab_size)`):
        Logits of the semantic codebook head (the first audio codebook) predicted from every hidden state.
    """

    semantic_logits: torch.FloatTensor | None = None


class VoxtralTtsRMSNorm(MistralRMSNorm):
    pass


class VoxtralTtsMLP(MistralMLP):
    pass


class VoxtralTtsAcousticTransformerAttention(MistralAttention):
    """Bidirectional multi-headed attention without positional embeddings."""

    def __init__(self, config: VoxtralTtsAcousticTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class VoxtralTtsAcousticTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: VoxtralTtsAcousticTransformerConfig, layer_idx: int):
        super().__init__()
        self.self_attn = VoxtralTtsAcousticTransformerAttention(config, layer_idx)
        self.mlp = VoxtralTtsMLP(config)
        self.input_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class VoxtralTtsAcousticTransformerPreTrainedModel(PreTrainedModel):
    config: VoxtralTtsAcousticTransformerConfig
    base_model_prefix = "acoustic_transformer"
    input_modalities = "text"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VoxtralTtsAcousticTransformerLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    Predicts one audio frame (37 codes) from the language model hidden state of the current position:
    the semantic code is the argmax of a linear head, and the 36 acoustic values are sampled with flow matching
    (Euler integration with classifier-free guidance), then quantized to `acoustic_codebook_size` levels.

    The flow-matching transformer sees a 3-token sequence `[noisy acoustic latent, time embedding, LLM state]` and
    reads the velocity off the first token.
    """
)
class VoxtralTtsAcousticTransformer(VoxtralTtsAcousticTransformerPreTrainedModel):
    def __init__(self, config: VoxtralTtsAcousticTransformerConfig):
        super().__init__(config)
        self.semantic_head = nn.Linear(config.llm_hidden_size, config.padded_semantic_vocab_size, bias=False)
        self.llm_projection = nn.Linear(config.llm_hidden_size, config.hidden_size, bias=False)
        self.time_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.input_projection = nn.Linear(config.num_acoustic_codebooks, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [VoxtralTtsAcousticTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.acoustic_head = nn.Linear(config.hidden_size, config.num_acoustic_codebooks, bias=False)
        self.post_init()

    def time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding (cosine first) of flow-matching times in [0, 1], computed in float32."""
        half_dim = self.config.hidden_size // 2
        inv_freq = torch.exp(
            -math.log(self.config.time_embedding_theta)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        freqs = timesteps.float()[:, None] * inv_freq[None, :]
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    def get_semantic_logits(self, llm_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_head(llm_hidden_states)

    @auto_docstring
    def forward(
        self,
        noisy_acoustic_latents: torch.Tensor,
        llm_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        noisy_acoustic_latents (`torch.Tensor` of shape `(batch_size, num_acoustic_codebooks)`):
            Acoustic latents at flow time `timesteps`.
        llm_hidden_states (`torch.Tensor` of shape `(batch_size, llm_hidden_size)`):
            Language model hidden states conditioning the frame (zeros for the unconditional branch).
        timesteps (`torch.Tensor` of shape `(batch_size,)`):
            Flow-matching times in [0, 1].

        Returns:
            `torch.Tensor` of shape `(batch_size, num_acoustic_codebooks)`: the predicted velocity.
        """
        dtype = llm_hidden_states.dtype
        time_states = self.time_projection(self.time_embedding(timesteps).to(dtype))
        llm_states = self.llm_projection(llm_hidden_states)
        return self._predict_velocity(noisy_acoustic_latents, llm_states, time_states, **kwargs)

    def _predict_velocity(self, noisy_acoustic_latents, llm_states, time_states, **kwargs):
        hidden_states = torch.stack(
            [self.input_projection(noisy_acoustic_latents.to(llm_states.dtype)), time_states, llm_states], dim=1
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        hidden_states = self.norm(hidden_states)
        return self.acoustic_head(hidden_states[:, 0])

    @torch.no_grad()
    def generate_frame(
        self,
        llm_hidden_states: torch.Tensor,
        cfg_alpha: float = 1.2,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """
        Generates one audio frame per batch element.

        Args:
            llm_hidden_states (`torch.Tensor` of shape `(batch_size, llm_hidden_size)`):
                Last hidden state of the language model.
            cfg_alpha (`float`, *optional*, defaults to 1.2):
                Classifier-free guidance scale of the flow-matching velocity.
            noise (`torch.Tensor` of shape `(batch_size, num_acoustic_codebooks)`, *optional*):
                Initial acoustic latents. Sampled from a standard normal distribution if not provided.
            generator (`torch.Generator`, *optional*):
                Random generator used to sample `noise`.

        Returns:
            `torch.LongTensor` of shape `(batch_size, 1 + num_acoustic_codebooks)`: the audio codes of the frame, in
            the model vocabulary (i.e. shifted by `num_special_audio_tokens`). A semantic code of
            `END_AUDIO_TOKEN` marks the end of the audio.
        """
        config = self.config
        batch_size, dtype, device = llm_hidden_states.shape[0], llm_hidden_states.dtype, llm_hidden_states.device
        num_special = config.num_special_audio_tokens

        semantic_logits = self.get_semantic_logits(llm_hidden_states).float()
        semantic_logits[:, EMPTY_AUDIO_TOKEN] = -float("inf")
        semantic_logits[:, num_special + config.semantic_codebook_size :] = -float("inf")
        semantic_codes = semantic_logits.argmax(dim=-1)

        if noise is None:
            noise_device = generator.device if generator is not None else device
            noise = torch.randn((batch_size, config.num_acoustic_codebooks), generator=generator, device=noise_device)
        latents = noise.to(device=device, dtype=dtype)

        # The schedule lives in the model dtype, while the time embedding itself is computed in float32
        timesteps = torch.linspace(0, 1, config.num_flow_matching_steps + 1, device=device).to(dtype)
        time_states = self.time_projection(self.time_embedding(timesteps).to(dtype))
        # Conditional and unconditional (zeroed LLM state) branches are batched together
        llm_states = self.llm_projection(torch.cat([llm_hidden_states, torch.zeros_like(llm_hidden_states)]))
        for step in range(config.num_flow_matching_steps):
            step_time_states = time_states[step].expand(2 * batch_size, -1)
            velocity = self._predict_velocity(torch.cat([latents, latents]), llm_states, step_time_states)
            cond_velocity, uncond_velocity = velocity[:batch_size], velocity[batch_size:]
            velocity = cfg_alpha * cond_velocity + (1 - cfg_alpha) * uncond_velocity
            latents = latents + velocity * (timesteps[step + 1] - timesteps[step])

        # Finite scalar quantization of the acoustic latents
        latents = latents.clamp(-1, 1)
        acoustic_codes = (((latents + 1) / 2) * (config.acoustic_codebook_size - 1)).round().long()
        acoustic_codes[semantic_codes == END_AUDIO_TOKEN] = EMPTY_AUDIO_TOKEN
        acoustic_codes = acoustic_codes + num_special
        return torch.cat([semantic_codes[:, None], acoustic_codes], dim=-1)


class VoxtralTtsAudioEmbeddings(nn.Module):
    """Embeds audio frames by summing the embeddings of their codes, all codebooks sharing one (offset) table."""

    def __init__(self, config: VoxtralTtsConfig):
        super().__init__()
        self.codebook_sizes = config.codebook_sizes
        self.embed_audio_tokens = nn.Embedding(config.audio_vocab_size, config.text_config.hidden_size)

    def forward(self, audio_codes: torch.LongTensor) -> torch.Tensor:
        offsets = torch.tensor([0] + self.codebook_sizes[:-1], device=audio_codes.device).cumsum(0)
        return self.embed_audio_tokens(audio_codes + offsets).sum(dim=-2)


class VoxtralTtsCodecCausalConv1d(nn.Conv1d):
    """Causal 1D convolution, left-padded with `pad_mode` (as in the original codec)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="reflect"):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.pad_mode = pad_mode
        self.padding_total = kernel_size - stride

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kernel_size, stride = self.kernel_size[0], self.stride[0]
        length = hidden_states.shape[-1]
        num_frames = math.ceil((length - kernel_size + self.padding_total) / stride + 1)
        extra_padding = (num_frames - 1) * stride + kernel_size - self.padding_total - length
        padding = (self.padding_total, extra_padding)
        if self.pad_mode == "reflect" and length <= max(padding):
            # reflect padding needs inputs longer than the padding: zero-pad on the right first
            extra_zeros = max(padding) - length + 1
            hidden_states = F.pad(hidden_states, (0, extra_zeros))
            hidden_states = F.pad(hidden_states, padding, mode=self.pad_mode)[..., :-extra_zeros]
        else:
            hidden_states = F.pad(hidden_states, padding, mode=self.pad_mode)
        return super().forward(hidden_states)


class VoxtralTtsCodecCausalConvTranspose1d(nn.ConvTranspose1d):
    """Transposed 1D convolution whose `kernel_size - stride` trailing (non-causal) outputs are trimmed."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        trim = self.kernel_size[0] - self.stride[0]
        return hidden_states[..., : hidden_states.shape[-1] - trim]


class VoxtralTtsCodecAttention(MistralAttention):
    """Attention with query/key RMS norms; the ALiBi, causal and sliding-window biases are passed as the mask."""

    def __init__(self, config: VoxtralTtsCodecConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = VoxtralTtsRMSNorm(config.num_attention_heads * self.head_dim, eps=config.qk_norm_eps)
        self.k_norm = VoxtralTtsRMSNorm(config.num_key_value_heads * self.head_dim, eps=config.qk_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # The ALiBi bias is an additive float mask, which only the eager and SDPA paths support
        attention_implementation = self.config._attn_implementation
        if attention_implementation not in ("eager", "sdpa"):
            attention_implementation = "sdpa"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention_implementation, eager_attention_forward
        )
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class VoxtralTtsCodecLayer(GradientCheckpointingLayer):
    def __init__(self, config: VoxtralTtsCodecConfig, layer_idx: int):
        super().__init__()
        self.self_attn = VoxtralTtsCodecAttention(config, layer_idx)
        self.mlp = VoxtralTtsMLP(config)
        self.input_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn_layer_scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_init))
        self.mlp_layer_scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + self.self_attn_layer_scale * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale * hidden_states
        return hidden_states


class VoxtralTtsCodecDecoderStage(nn.Module):
    """An (up)sampling convolution followed by sliding-window transformer layers."""

    def __init__(self, config: VoxtralTtsCodecConfig, stage_idx: int):
        super().__init__()
        kernel_size = config.decoder_kernel_sizes[stage_idx]
        stride = config.decoder_strides[stage_idx]
        if stage_idx == 0:
            latent_dim = config.semantic_dim + config.acoustic_dim
            self.conv = VoxtralTtsCodecCausalConv1d(
                latent_dim, config.hidden_size, kernel_size, stride, pad_mode="replicate"
            )
        elif kernel_size != 1 or stride != 1:
            self.conv = VoxtralTtsCodecCausalConvTranspose1d(
                config.hidden_size, config.hidden_size, kernel_size, stride
            )
        else:
            self.conv = None
        # The window is defined at the patch rate and halved for every 2x rate reduction
        self.sliding_window = config.sliding_window // math.prod(config.decoder_strides[stage_idx + 1 :])
        first_layer_idx = sum(config.decoder_num_layers[:stage_idx])
        self.layers = nn.ModuleList(
            [VoxtralTtsCodecLayer(config, first_layer_idx + i) for i in range(config.decoder_num_layers[stage_idx])]
        )


class VoxtralTtsCodecQuantizer(nn.Module):
    """Decodes semantic (vector-quantized) and acoustic (finite scalar quantized) codes into latents."""

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.acoustic_codebook_size = config.acoustic_codebook_size
        self.cluster_usage = nn.Buffer(torch.ones(config.semantic_codebook_size))
        self.embedding_sum = nn.Buffer(torch.zeros(config.semantic_codebook_size, config.semantic_dim))

    def decode(self, semantic_codes: torch.LongTensor, acoustic_codes: torch.LongTensor, dtype) -> torch.Tensor:
        codebook = self.embedding_sum / self.cluster_usage.clamp(min=1e-5)[:, None]
        semantic_latents = F.embedding(semantic_codes, codebook).to(dtype)
        acoustic_latents = (acoustic_codes * 2 / (self.acoustic_codebook_size - 1) - 1).to(dtype)
        return torch.cat([semantic_latents, acoustic_latents], dim=-1)


@auto_docstring
class VoxtralTtsCodecPreTrainedModel(PreTrainedModel):
    config: VoxtralTtsCodecConfig
    base_model_prefix = "codec_model"
    main_input_name = "audio_codes"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VoxtralTtsCodecLayer"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VoxtralTtsCodecLayer):
            init.constant_(module.self_attn_layer_scale, self.config.layer_scale_init)
            init.constant_(module.mlp_layer_scale, self.config.layer_scale_init)
        elif isinstance(module, VoxtralTtsCodecQuantizer):
            init.ones_(module.cluster_usage)
            init.zeros_(module.embedding_sum)


@auto_docstring(
    custom_intro="""
    The Voxtral TTS audio codec decoder, which turns audio codes (1 semantic and 36 acoustic codebooks at 12.5 Hz)
    into a 24 kHz waveform. The released checkpoint only ships the decoder, so encoding audio is not supported.
    """
)
class VoxtralTtsCodecModel(VoxtralTtsCodecPreTrainedModel):
    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__(config)
        self.quantizer = VoxtralTtsCodecQuantizer(config)
        self.stages = nn.ModuleList(
            [VoxtralTtsCodecDecoderStage(config, stage_idx) for stage_idx in range(len(config.decoder_num_layers))]
        )
        self.output_proj = VoxtralTtsCodecCausalConv1d(
            config.hidden_size, config.patch_size, config.patch_proj_kernel_size
        )
        self.post_init()

    def _alibi_sliding_window_mask(self, sequence_length: int, sliding_window: int, dtype, device) -> torch.Tensor:
        num_heads = self.config.num_attention_heads
        if not math.log2(num_heads).is_integer():
            raise ValueError(f"ALiBi slopes are only implemented for a power-of-two number of heads, got {num_heads}")
        # slopes[h] = r**h with r = 2**(-8 / num_heads)
        slopes = 2.0 ** (-8.0 / num_heads * torch.arange(num_heads, device=device, dtype=torch.float32))
        positions = torch.arange(sequence_length, device=device)
        relative_positions = positions[None, :] - positions[:, None]  # key - query
        mask = slopes.to(dtype)[:, None, None] * relative_positions[None].to(dtype)
        outside = (relative_positions > 0) | (relative_positions < -sliding_window)
        mask = mask.masked_fill(outside[None], torch.finfo(dtype).min)
        return mask[None]

    @can_return_tuple
    @auto_docstring
    def decode(
        self,
        audio_codes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VoxtralTtsCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, 1 + acoustic_dim)`):
            Codec codes: the semantic code followed by the acoustic codes of every frame (without the offset of the
            special audio tokens used by the language model).
        """
        dtype = self.output_proj.weight.dtype
        hidden_states = self.quantizer.decode(audio_codes[..., 0], audio_codes[..., 1:], dtype)
        for stage in self.stages:
            if stage.conv is not None:
                hidden_states = stage.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
            attention_mask = self._alibi_sliding_window_mask(
                hidden_states.shape[1], stage.sliding_window, hidden_states.dtype, hidden_states.device
            )
            for layer in stage.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = self.output_proj(hidden_states.transpose(1, 2))  # (batch, patch_size, num_steps)
        audio_values = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1)
        return VoxtralTtsCodecDecoderOutput(audio_values=audio_values)

    @can_return_tuple
    @auto_docstring
    def forward(
        self, audio_codes: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> VoxtralTtsCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, 1 + acoustic_dim)`):
            Codec codes: the semantic code followed by the acoustic codes of every frame (without the offset of the
            special audio tokens used by the language model).
        """
        return self.decode(audio_codes, **kwargs)


@auto_docstring
class VoxtralTtsPreTrainedModel(PreTrainedModel):
    config: VoxtralTtsConfig
    base_model_prefix = "model"
    input_modalities = "text"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VoxtralTtsAcousticTransformerLayer", "VoxtralTtsCodecLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VoxtralTtsCodecLayer):
            init.constant_(module.self_attn_layer_scale, self.config.codec_config.layer_scale_init)
            init.constant_(module.mlp_layer_scale, self.config.codec_config.layer_scale_init)
        elif isinstance(module, VoxtralTtsCodecQuantizer):
            init.ones_(module.cluster_usage)
            init.zeros_(module.embedding_sum)


@auto_docstring(
    custom_intro="""
    The Voxtral TTS language model backbone, which embeds text tokens, voice embeddings and audio frames and outputs
    the hidden states that condition the generation of every audio frame.
    """
)
class VoxtralTtsModel(VoxtralTtsPreTrainedModel):
    def __init__(self, config: VoxtralTtsConfig):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.audio_embeddings = VoxtralTtsAudioEmbeddings(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor | None, inputs_embeds: torch.Tensor, voice_embeddings: torch.Tensor
    ) -> torch.BoolTensor:
        if input_ids is None:
            special_audio_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = special_audio_mask.all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id
        num_placeholders = int(special_audio_mask.sum())
        if num_placeholders != voice_embeddings.shape[0]:
            raise ValueError(
                f"Got {voice_embeddings.shape[0]} voice embedding vectors for {num_placeholders} audio placeholder "
                "tokens in the prompt."
            )
        return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        voice_embeddings: torch.FloatTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        voice_embeddings (`torch.FloatTensor` of shape `(num_audio_placeholders, hidden_size)`, *optional*):
            Embeddings of the voice preset(s), which replace the `[AUDIO]` placeholder tokens of the prompt(s), in
            order. Returned by [`VoxtralTtsProcessor`].
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, num_codebooks)`, *optional*):
            Audio frames (in the model vocabulary) appended after the prompt: their embeddings are the sum of the
            embeddings of their codes. When `input_ids` / `inputs_embeds` is not provided, only the audio frames are
            embedded (as done at every generation step).
        """
        if (input_ids is None) == (inputs_embeds is None) and audio_codes is None:
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`, or `audio_codes`.")

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if inputs_embeds is not None and voice_embeddings is not None:
            special_audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds, voice_embeddings)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, voice_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
            )
        if audio_codes is not None:
            audio_embeds = self.audio_embeddings(audio_codes)
            if inputs_embeds is None:
                inputs_embeds = audio_embeds
            else:
                inputs_embeds = torch.cat([inputs_embeds, audio_embeds.to(inputs_embeds.dtype)], dim=1)
                # A prompt attention mask (e.g. from the processor) does not cover the appended audio frames
                if (
                    attention_mask is not None
                    and attention_mask.shape[-1] == inputs_embeds.shape[1] - audio_codes.shape[1]
                ):
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones(audio_codes.shape[:2])], dim=-1
                    )
        if attention_mask is not None and attention_mask.ndim == 2 and cache is None:
            if attention_mask.shape[-1] != inputs_embeds.shape[1]:
                raise ValueError(
                    f"`attention_mask` covers {attention_mask.shape[-1]} positions, but the inputs have "
                    f"{inputs_embeds.shape[1]}."
                )
        if position_ids is None and attention_mask is not None and attention_mask.ndim == 2 and cache is None:
            # Left-padded prompts: positions start at the first non-padding token
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    The Voxtral TTS model: a Mistral language model conditioned on a voice preset, a flow-matching acoustic transformer
    that predicts the 37 audio codes of each 80 ms frame, and a codec decoder that turns them into a 24 kHz waveform.
    """
)
class VoxtralTtsForTextToSpeech(VoxtralTtsPreTrainedModel, VoxtralTtsGenerationMixin):
    output_modalities = ("audio",)

    def __init__(self, config: VoxtralTtsConfig):
        super().__init__(config)
        self.model = VoxtralTtsModel(config)
        self.acoustic_transformer = AutoModel.from_config(config.acoustic_transformer_config)
        self.codec_model = AutoModel.from_config(config.codec_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        voice_embeddings: torch.FloatTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VoxtralTtsOutputWithPast:
        r"""
        voice_embeddings (`torch.FloatTensor` of shape `(num_audio_placeholders, hidden_size)`, *optional*):
            Embeddings of the voice preset(s), which replace the `[AUDIO]` placeholder tokens of the prompt(s), in
            order. Returned by [`VoxtralTtsProcessor`].
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, num_codebooks)`, *optional*):
            Audio frames (in the model vocabulary) appended after the prompt.

        Example:

        ```python
        >>> import soundfile as sf
        >>> from transformers import AutoProcessor, VoxtralTtsForTextToSpeech

        >>> processor = AutoProcessor.from_pretrained("mistralai/Voxtral-4B-TTS-2603")
        >>> model = VoxtralTtsForTextToSpeech.from_pretrained("mistralai/Voxtral-4B-TTS-2603", device_map="auto")

        >>> inputs = processor("Paris is a beautiful city!", voice="casual_male").to(model.device)
        >>> outputs = model.generate(**inputs)
        >>> sf.write("paris.wav", outputs.audio[0].float().cpu().numpy(), processor.sampling_rate)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            voice_embeddings=voice_embeddings,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        semantic_logits = self.acoustic_transformer.get_semantic_logits(outputs.last_hidden_state)
        return VoxtralTtsOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            semantic_logits=semantic_logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "VoxtralTtsAcousticTransformerPreTrainedModel",
    "VoxtralTtsAcousticTransformer",
    "VoxtralTtsCodecPreTrainedModel",
    "VoxtralTtsCodecModel",
    "VoxtralTtsPreTrainedModel",
    "VoxtralTtsModel",
    "VoxtralTtsForTextToSpeech",
]
