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

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..lfm3.configuration_lfm3 import Lfm3Config
from ..lfm3.modeling_lfm3 import Lfm3Model, Lfm3PreTrainedModel, Lfm3RotaryEmbedding
from ..xcodec2.modeling_xcodec2 import Xcodec2FiniteScalarQuantization, Xcodec2ISTFTHead


@auto_docstring(checkpoint="LiquidAI/Liquid-Codec")
@strict
class LiquidCodecEncoderConfig(Lfm3Config):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 7):
        L_cache dim in the conv layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.
    use_attention_gate (`bool`, *optional*, defaults to `True`):
        Whether the attention layers scale each head's output by a sigmoid gate computed from the layer input.
    n_fft (`int`, *optional*, defaults to 1280):
        FFT size; the input spectrogram has `3 * (n_fft // 2 + 1)` channels.
    subsampling_factor (`int`, *optional*, defaults to 6):
        Spectrogram frames per codec frame: stride of the convolutional subsampler.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        Channels of the convolutional subsampler before the backbone.
    quantization_dim (`int`, *optional*, defaults to 256):
        Width of the latent given to the quantizer.
    """

    model_type = "liquid_codec_encoder"
    base_config_key = "encoder_config"

    vocab_size = AttributeError()
    pad_token_id = AttributeError()
    bos_token_id = AttributeError()
    eos_token_id = AttributeError()
    tie_word_embeddings = AttributeError()
    base_model_tp_plan = AttributeError()

    hidden_size: int = 512
    intermediate_size: int = 2304
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 131_072
    use_cache: bool = False
    sliding_window: int | None = None
    n_fft: int = 1280
    subsampling_factor: int = 6
    subsampling_conv_channels: int = 256
    quantization_dim: int = 256

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
            }
        if self.layer_types is None:
            self.layer_types = ["conv"] + ["conv", "full_attention"] * ((self.num_hidden_layers - 2) // 2) + ["conv"]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/Liquid-Codec")
@strict
class LiquidCodecDecoderConfig(Lfm3Config):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 7):
        L_cache dim in the conv layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.
    use_attention_gate (`bool`, *optional*, defaults to `True`):
        Whether the attention layers scale each head's output by a sigmoid gate computed from the layer input.
    n_fft (`int`, *optional*, defaults to 1280):
        Synthesis FFT size.
    hop_length (`int`, *optional*, defaults to 320):
        Synthesis hop length.
    upsampling_factor (`int`, *optional*, defaults to 6):
        STFT frames per codec frame: each codec frame is repeated this many times before the backbone.
    quantization_dim (`int`, *optional*, defaults to 256):
        Width of the latent from the quantizer.
    """

    model_type = "liquid_codec_decoder"
    base_config_key = "decoder_config"

    vocab_size = AttributeError()
    pad_token_id = AttributeError()
    bos_token_id = AttributeError()
    eos_token_id = AttributeError()
    tie_word_embeddings = AttributeError()
    base_model_tp_plan = AttributeError()

    hidden_size: int = 512
    intermediate_size: int = 2304
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 131_072
    use_cache: bool = False
    sliding_window: int | None = 30
    n_fft: int = 1280
    hop_length: int = 320
    upsampling_factor: int = 6
    quantization_dim: int = 256

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
            }
        if self.layer_types is None:
            self.layer_types = (
                ["conv"] + ["conv", "sliding_attention"] * ((self.num_hidden_layers - 2) // 2) + ["conv"]
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/Liquid-Codec")
@strict
class LiquidCodecConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict` or [`LiquidCodecEncoderConfig`], *optional*):
        Encoder config. `None`: no encoder (`from_pretrained(..., encoder_config=None)` loads the decoder only).
    decoder_config (`dict` or [`LiquidCodecDecoderConfig`], *optional*):
        Decoder config. `None`: no decoder (`from_pretrained(..., decoder_config=None)` loads the encoder only).
    quantization_dim (`int`, *optional*, defaults to 256):
        Width of the quantizer's latent.
    quantization_levels (`list[int]`, *optional*, defaults to `[8, 7, 6, 6]`):
        Levels per FSQ dimension; their product is the codebook size.
    num_codebooks (`int`, *optional*, defaults to 12):
        Number of parallel FSQ codebooks: the latent is split into this many groups, each quantized separately.
    """

    model_type = "liquid_codec"
    sub_configs_defaults = {
        "encoder_config": SubConfigSpec(config_class=LiquidCodecEncoderConfig, optional=True),
        "decoder_config": SubConfigSpec(config_class=LiquidCodecDecoderConfig, optional=True),
    }

    encoder_config: dict | PreTrainedConfig | None = field(default_factory=dict)
    decoder_config: dict | PreTrainedConfig | None = field(default_factory=dict)
    quantization_dim: int = 256
    quantization_levels: list[int] | tuple[int, ...] = (8, 7, 6, 6)
    num_codebooks: int = 12
    sampling_rate: int = 24_000
    initializer_range: float = 0.02

    def validate_architecture(self):
        if self.encoder_config is None and self.decoder_config is None:
            raise ValueError("The codec needs at least one of `encoder_config` and `decoder_config`.")
        for half in (self.encoder_config, self.decoder_config):
            if isinstance(half, PreTrainedConfig) and half.quantization_dim != self.quantization_dim:
                raise ValueError(
                    f"The halves must read and write the quantizer's `quantization_dim={self.quantization_dim}` "
                    f"latent, got {half.quantization_dim} for {type(half).__name__}."
                )

    @property
    def codebook_size(self) -> int:
        return math.prod(self.quantization_levels)


class LiquidCodecOverlapAddCache:
    """
    Overlap-add cache of [`LiquidCodecISTFTHead`] for streaming decode: holds back the last `n_fft - hop_length`
    samples of a chunk until the next one completes them. [`~LiquidCodecOverlapAddCache.flush`] ends the stream.
    """

    def __init__(self, config: LiquidCodecDecoderConfig):
        self.overlap = config.n_fft - config.hop_length
        self.padding = self.overlap // 2
        self.reset()

    def reset(self):
        self.audio: torch.Tensor | None = None
        self.envelope: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.has_mask = False
        self.num_samples_to_trim = self.padding

    @staticmethod
    def normalize(audio: torch.Tensor, window_envelope: torch.Tensor, samples_mask: torch.Tensor) -> torch.Tensor:
        # Clamp as expected by original: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/vq/codec_decoder_vocos.py#L82
        return audio / window_envelope.clamp(min=1e-11) * samples_mask

    def update(
        self, audio: torch.Tensor, window_envelope: torch.Tensor, samples_mask: torch.Tensor, has_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adds the cached tail, caches the new one; returns the final samples with their envelope and mask."""
        if self.audio is not None:
            audio = torch.cat([audio[:, : self.overlap] + self.audio, audio[:, self.overlap :]], dim=1)
            window_envelope = torch.cat(
                [window_envelope[:, : self.overlap] + self.envelope, window_envelope[:, self.overlap :]], dim=1
            )
            samples_mask = torch.cat(
                [samples_mask[:, : self.overlap] | self.mask, samples_mask[:, self.overlap :]], dim=1
            )

        self.audio, self.envelope = audio[:, -self.overlap :], window_envelope[:, -self.overlap :]
        self.mask = samples_mask[:, -self.overlap :]
        self.has_mask = self.has_mask or has_mask
        audio, window_envelope = audio[:, : -self.overlap], window_envelope[:, : -self.overlap]
        samples_mask = samples_mask[:, : -self.overlap]

        # "same" padding: the first `padding` samples of the stream are dropped
        num_trimmed = min(self.num_samples_to_trim, audio.shape[1])
        self.num_samples_to_trim -= num_trimmed
        return audio[:, num_trimmed:], window_envelope[:, num_trimmed:], samples_mask[:, num_trimmed:]

    def flush(self) -> "LiquidCodecDecoderOutput":
        """Ends the stream: last samples (with their mask if the stream had one), then resets the cache."""
        if self.audio is None:
            raise ValueError("Nothing to flush: no chunk was decoded since the cache was created or last flushed.")

        audio = self.audio[:, self.num_samples_to_trim : -self.padding]
        window_envelope = self.envelope[:, self.num_samples_to_trim : -self.padding]
        samples_mask = self.mask[:, self.num_samples_to_trim : -self.padding]
        has_mask = self.has_mask

        self.reset()
        return LiquidCodecDecoderOutput(
            audio_values=self.normalize(audio, window_envelope, samples_mask),
            audio_values_mask=samples_mask if has_mask else None,
        )


@auto_docstring
@dataclass
class LiquidCodecEncoderModelOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, quantization_dim, num_frames)`):
        Latents given to the quantizer, one per codec frame.
    latents_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*, returned with an
    `input_features_mask`):
        Valid frames of each clip.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each attention layer of the encoder.
    """

    latents: torch.FloatTensor | None = None
    latents_mask: torch.BoolTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class LiquidCodecEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        Codes, per codebook and frame.
    audio_codes_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*, returned with an
    `input_features_mask`):
        Valid frames of each clip.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each attention layer of the encoder.
    """

    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.BoolTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class LiquidCodecDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Decoded audio, float32, at the codec's `sampling_rate`.
    audio_values_mask (`torch.BoolTensor` of shape `(batch_size, num_samples)`, *optional*, returned with an
    `audio_codes_mask`):
        Samples decoded from the valid frames of each clip.
    decoder_cache (`Cache`, *optional*, returned when streaming):
        Decoder cache, to pass with the next chunk.
    overlap_add_cache ([`LiquidCodecOverlapAddCache`], *optional*, returned when streaming):
        Overlap-add cache, to pass with the next chunk; `flush()` ends the stream.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the decoder at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each attention layer of the decoder.
    """

    audio_values: torch.FloatTensor | None = None
    audio_values_mask: torch.BoolTensor | None = None
    decoder_cache: Cache | None = None
    overlap_add_cache: LiquidCodecOverlapAddCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class LiquidCodecOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Reconstructed audio.
    audio_values_mask (`torch.BoolTensor` of shape `(batch_size, num_samples)`, *optional*, returned with an
    `input_features_mask`):
        Samples decoded from the valid frames of each clip.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        Codes, per codebook and frame.
    audio_codes_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`, *optional*, returned with an
    `input_features_mask`):
        Valid frames of each clip.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each attention layer of the encoder.
    """

    audio_values: torch.FloatTensor | None = None
    audio_values_mask: torch.BoolTensor | None = None
    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.BoolTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class LiquidCodecEncoderSubsampler(nn.Module):
    def __init__(self, config: LiquidCodecEncoderConfig):
        super().__init__()
        # 3 features per frequency bin: log-magnitude, cos(phase), sin(phase)
        self.in_proj = nn.Conv1d(3 * (config.n_fft // 2 + 1), config.subsampling_conv_channels, kernel_size=1)
        # causal: left-pad 2, right padding cut
        self.conv = nn.Conv1d(
            config.subsampling_conv_channels, config.subsampling_conv_channels, kernel_size=3, padding=2
        )
        self.out_proj = nn.Conv1d(
            config.subsampling_conv_channels,
            config.hidden_size,
            kernel_size=config.subsampling_factor,
            stride=config.subsampling_factor,
        )
        self.act = ACT2FN["silu"]

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.in_proj(input_features))
        hidden_states = self.act(self.conv(hidden_states)[:, :, : hidden_states.shape[-1]])
        return self.out_proj(hidden_states).transpose(1, 2)


class LiquidCodecDecoderUpsampler(nn.Module):
    def __init__(self, config: LiquidCodecDecoderConfig):
        super().__init__()
        self.upsampling_factor = config.upsampling_factor
        self.proj = nn.Linear(config.quantization_dim, config.hidden_size)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(latents.transpose(1, 2).to(self.proj.weight.dtype))
        hidden_states = F.interpolate(
            hidden_states.transpose(1, 2), scale_factor=self.upsampling_factor, mode="nearest-exact"
        )
        return hidden_states.transpose(1, 2)


class LiquidCodecFiniteScalarQuantization(Xcodec2FiniteScalarQuantization):
    pass


class LiquidCodecQuantizer(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_dim = len(config.quantization_levels)

        self.quantizer = LiquidCodecFiniteScalarQuantization(config)
        self.project_in = nn.Linear(config.quantization_dim, self.num_codebooks * self.codebook_dim)
        self.project_out = nn.Linear(self.num_codebooks * self.codebook_dim, config.quantization_dim)

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.project_in(latents.movedim(1, -1))

        _, indices = self.quantizer(latents.reshape(*latents.shape[:-1], self.num_codebooks, self.codebook_dim))
        return indices.long().transpose(1, 2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # from integer levels: half precision rounds the `codebook` buffer
        latents = self.quantizer._indices_to_codes(codes.transpose(1, 2)).to(self.project_out.weight.dtype)
        latents = latents.reshape(*latents.shape[:-2], self.num_codebooks * self.codebook_dim)

        return self.project_out(latents).movedim(-1, 1)


class LiquidCodecISTFTHead(Xcodec2ISTFTHead):
    def __init__(self, config: LiquidCodecDecoderConfig):
        super().__init__(config)
        window = torch.hann_window(config.n_fft, dtype=torch.float32)
        self.window = nn.Buffer(window)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frames_mask: torch.Tensor | None = None,
        overlap_add_cache: LiquidCodecOverlapAddCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stft_pred = self.linear(hidden_states)
        # Keep the frequency axis last for `torch.polar` compatibility with torch.compile
        magnitude, phase = stft_pred.chunk(2, dim=-1)
        # Cast to float32: complex exponential and irfft are not supported for fp16 (ComplexHalf)
        magnitude = magnitude.float()
        phase = phase.float()
        # no clamp, unlike Xcodec2
        magnitude = torch.exp(magnitude)
        spectrogram_complex = torch.polar(magnitude, phase)

        # Back to audio (ISTFT with manual "same" padding: torch.istft lacks a native same-padding mode,
        # so we use irfft + fold with explicit pre-computed padding to replicate it)
        time_frames = torch.fft.irfft(spectrogram_complex, self.n_fft, dim=-1, norm="backward").transpose(1, 2)
        time_frames = time_frames * self.window[None, :, None]
        num_frames = spectrogram_complex.shape[1]

        # per clip: own frames, own window envelope, as alone
        has_mask = frames_mask is not None
        if frames_mask is None:
            frames_mask = self.window.new_ones((time_frames.shape[0], num_frames))
        frames_mask = frames_mask.to(time_frames)
        # a frame owns the `hop_length` samples at the center of its window
        samples_mask = F.pad(frames_mask.bool().repeat_interleave(self.hop_length, dim=-1), (self.padding,) * 2)
        frames_mask = frames_mask[:, None, :]
        output_size = (num_frames - 1) * self.hop_length + self.n_fft
        audio = F.fold(
            time_frames * frames_mask,
            output_size=(1, output_size),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[:, 0, 0]

        # Normalize
        window_envelope = F.fold(
            self.window.square()[None, :, None] * frames_mask,
            output_size=(1, output_size),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[:, 0, 0]

        if overlap_add_cache is None:
            audio = audio[:, self.padding : -self.padding]
            window_envelope = window_envelope[:, self.padding : -self.padding]
            samples_mask = samples_mask[:, self.padding : -self.padding]
        else:
            audio, window_envelope, samples_mask = overlap_add_cache.update(
                audio, window_envelope, samples_mask, has_mask=has_mask
            )

        return LiquidCodecOverlapAddCache.normalize(audio, window_envelope, samples_mask), samples_mask


class LiquidCodecRotaryEmbedding(Lfm3RotaryEmbedding):
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        # diff -> as the original: cos / sin rounded to the model dtype, rotation in float32
        return cos.to(x.dtype).float(), sin.to(x.dtype).float()


@auto_docstring
class LiquidCodecPreTrainedModel(Lfm3PreTrainedModel):
    config: LiquidCodecConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    # the synthesis window loses precision in half precision
    _keep_in_fp32_modules_strict = ["head.window"]
    _supports_flash_attn = False

    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, LiquidCodecISTFTHead):
            init.copy_(module.window, torch.hann_window(module.n_fft, dtype=torch.float32))
        elif isinstance(module, LiquidCodecFiniteScalarQuantization):
            levels, basis, codebook = module._compute_buffers(device=module.levels.device)
            init.copy_(module.levels, levels)
            init.copy_(module.basis, basis)
            init.copy_(module.codebook, codebook)


@auto_docstring(
    custom_intro="""
    The encoder half of Liquid Codec: embedder, LFM3 layers and output projection, from spectrogram to continuous
    latents.
    """
)
class LiquidCodecEncoderModel(Lfm3Model):
    config: LiquidCodecEncoderConfig
    base_model_prefix = "encoder"
    _keys_to_ignore_on_load_unexpected = [r"^decoder\.", r"^quantizer\."]

    def __init__(self, config: LiquidCodecEncoderConfig):
        super().__init__(config)
        self.subsampler = LiquidCodecEncoderSubsampler(config)
        self.output_proj = nn.Linear(config.hidden_size, config.quantization_dim)

        del self.padding_idx
        del self.vocab_size
        del self.embed_tokens

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecEncoderModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, num_frames)`):
            Spectrogram from [`LiquidCodecFeatureExtractor`], whole codec frames per clip.
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid spectrogram frames of each clip (right-padded); returns `latents_mask`.
        """
        hidden_states = self.subsampler(input_features)

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": hidden_states,
            "attention_mask": None,
            "cache": None,
            "position_ids": position_ids,
        }
        mask_creation_functions = {
            "full_attention": lambda: create_causal_mask(**mask_kwargs),
            "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
            "conv": lambda: create_recurrent_attention_mask(**mask_kwargs),
        }
        causal_mask_mapping = {
            layer_type: mask_creation_functions[layer_type]() for layer_type in set(self.config.layer_types)
        }

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)
        latents = self.output_proj(hidden_states).transpose(1, 2)

        latents_mask = None
        if input_features_mask is not None:
            num_frames = input_features_mask.sum(-1, keepdim=True) // self.config.subsampling_factor
            latents_mask = torch.arange(latents.shape[-1], device=latents.device) < num_frames.to(latents.device)

        return LiquidCodecEncoderModelOutput(latents=latents, latents_mask=latents_mask)


@auto_docstring(
    custom_intro="""
    The decoder half of Liquid Codec: upsampler, LFM3 layers and ISTFT head, from continuous latents to audio.
    """
)
class LiquidCodecDecoderModel(Lfm3Model):
    config: LiquidCodecDecoderConfig
    base_model_prefix = "decoder"
    main_input_name = "latents"
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^quantizer\."]

    def __init__(self, config: LiquidCodecDecoderConfig):
        super().__init__(config)
        self.upsampler = LiquidCodecDecoderUpsampler(config)
        self.head = LiquidCodecISTFTHead(config)

        del self.padding_idx
        del self.vocab_size
        del self.embed_tokens

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        latents: torch.FloatTensor,
        latents_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        overlap_add_cache: LiquidCodecOverlapAddCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecDecoderOutput:
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, quantization_dim, num_frames)`):
            Quantizer latents, one per codec frame.
        latents_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid frames of each clip (right-padded): each clip decoded as alone; returns `audio_values_mask`.
        overlap_add_cache ([`LiquidCodecOverlapAddCache`], *optional*):
            Overlap-add cache from the previous chunk.
        """
        hidden_states = self.upsampler(latents)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": hidden_states,
            "attention_mask": None,
            "cache": cache,
            "position_ids": position_ids,
        }
        mask_creation_functions = {
            "full_attention": lambda: create_causal_mask(**mask_kwargs),
            "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
            "conv": lambda: create_recurrent_attention_mask(**mask_kwargs),
        }
        causal_mask_mapping = {
            layer_type: mask_creation_functions[layer_type]() for layer_type in set(self.config.layer_types)
        }

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        # causal: a clip never sees its right padding
        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                cache=cache,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        frames_mask = None
        if latents_mask is not None:
            frames_mask = latents_mask.repeat_interleave(self.config.upsampling_factor, dim=-1)
        audio_values, audio_values_mask = self.head(
            hidden_states, frames_mask=frames_mask, overlap_add_cache=overlap_add_cache
        )

        return LiquidCodecDecoderOutput(
            audio_values=audio_values,
            audio_values_mask=audio_values_mask if latents_mask is not None else None,
            decoder_cache=cache,
            overlap_add_cache=overlap_add_cache,
        )


@auto_docstring(custom_intro="The Liquid Codec audio tokenizer.")
class LiquidCodecModel(LiquidCodecPreTrainedModel, PreTrainedAudioTokenizerBase):
    # full checkpoint loads with a half dropped (`encoder_config=None` / `decoder_config=None`)
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^decoder\."]

    def __init__(self, config: LiquidCodecConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config) if config.encoder_config is not None else None
        self.quantizer = LiquidCodecQuantizer(config)
        self.decoder = AutoModel.from_config(config.decoder_config) if config.decoder_config is not None else None

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecEncoderOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, num_frames)`):
            Spectrogram from [`LiquidCodecFeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid spectrogram frames of each clip (right-padded); returns `audio_codes_mask`.

        Encodes the spectrogram into discrete codes of shape `(batch_size, num_codebooks, num_frames)`: the latents of
        [`LiquidCodecEncoderModel.forward`], quantized.
        """
        if self.encoder is None:
            raise ValueError(
                "Audio encoding was requested, but the model was initialized without an encoder (`encoder_config=None`)."
            )

        encoder_outputs = self.encoder(
            input_features, input_features_mask=input_features_mask, return_dict=True, **kwargs
        )
        audio_codes = self.quantizer.encode(encoder_outputs.latents)

        audio_codes_mask = encoder_outputs.latents_mask
        if audio_codes_mask is not None:
            audio_codes_mask = audio_codes_mask.to(audio_codes.device)

        return LiquidCodecEncoderOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def decode(
        self,
        audio_codes: torch.LongTensor,
        audio_codes_mask: torch.Tensor | None = None,
        decoder_cache: Cache | None = None,
        overlap_add_cache: LiquidCodecOverlapAddCache | None = None,
        use_streaming: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            Codes to decode; `hop_length * upsampling_factor` samples per frame.
        audio_codes_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid frames of each clip (right-padded): each clip decoded as alone; returns `audio_values_mask`.
            When streaming: the chunk's frames.
        decoder_cache (`Cache`, *optional*):
            Decoder cache from the previous chunk.
        overlap_add_cache ([`LiquidCodecOverlapAddCache`], *optional*):
            Overlap-add cache from the previous chunk.
        use_streaming (`bool`, *optional*, defaults to `False`):
            Decode chunk by chunk: caches created on the first chunk, returned with each. Trailing samples stay
            in `overlap_add_cache` until `overlap_add_cache.flush()`.

        Decodes the codes into audio: the codes are dequantized into the latents of [`LiquidCodecDecoderModel.forward`].
        """
        if self.decoder is None:
            raise ValueError(
                "Audio decoding was requested, but the model was initialized without a decoder (`decoder_config=None`)."
            )

        if use_streaming and decoder_cache is None:
            decoder_cache = DynamicCache(config=self.config.decoder_config)
        if use_streaming and overlap_add_cache is None:
            overlap_add_cache = LiquidCodecOverlapAddCache(config=self.config.decoder_config)

        latents = self.quantizer.decode(audio_codes)

        return self.decoder(
            latents,
            latents_mask=audio_codes_mask,
            cache=decoder_cache,
            overlap_add_cache=overlap_add_cache,
            return_dict=True,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, num_frames)`):
            Spectrogram from [`LiquidCodecFeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid spectrogram frames of each clip (right-padded): each clip decoded as alone.
        """
        encoded = self.encode(input_features, input_features_mask, return_dict=True, **kwargs)
        decoded = self.decode(encoded.audio_codes, encoded.audio_codes_mask, return_dict=True)

        return LiquidCodecOutput(
            audio_values=decoded.audio_values,
            audio_values_mask=decoded.audio_values_mask,
            audio_codes=encoded.audio_codes,
            audio_codes_mask=encoded.audio_codes_mask,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
        )


__all__ = [
    "LiquidCodecConfig",
    "LiquidCodecDecoderConfig",
    "LiquidCodecDecoderModel",
    "LiquidCodecEncoderConfig",
    "LiquidCodecEncoderModel",
    "LiquidCodecModel",
    "LiquidCodecOverlapAddCache",
    "LiquidCodecPreTrainedModel",
]
