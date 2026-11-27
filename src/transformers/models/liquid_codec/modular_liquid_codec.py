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
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling
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
        Width of the bottleneck between the backbone and the quantizer projection.
    num_codebooks (`int`, *optional*, defaults to 12):
        Number of FSQ codebooks the output latent is split into.
    codebook_dim (`int`, *optional*, defaults to 4):
        FSQ dimensions per codebook: the output latent has `num_codebooks * codebook_dim` channels.
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
    num_codebooks: int = 12
    codebook_dim: int = 4

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
        Width of the bottleneck between the quantizer projection and the backbone.
    num_codebooks (`int`, *optional*, defaults to 12):
        Number of FSQ codebooks the input latent is made of.
    codebook_dim (`int`, *optional*, defaults to 4):
        FSQ dimensions per codebook: the input latent has `num_codebooks * codebook_dim` channels.
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
    num_codebooks: int = 12
    codebook_dim: int = 4

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
    quantization_levels: list[int] | tuple[int, ...] = (8, 7, 6, 6)
    num_codebooks: int = 12
    sampling_rate: int = 24_000
    initializer_range: float = 0.02

    def validate_architecture(self):
        if self.encoder_config is None and self.decoder_config is None:
            raise ValueError("The codec needs at least one of `encoder_config` and `decoder_config`.")
        codebook_dim = len(self.quantization_levels)
        for half in (self.encoder_config, self.decoder_config):
            if isinstance(half, PreTrainedConfig) and (half.num_codebooks, half.codebook_dim) != (
                self.num_codebooks,
                codebook_dim,
            ):
                raise ValueError(
                    f"The halves must match the quantizer's `num_codebooks={self.num_codebooks}` and "
                    f"`codebook_dim={codebook_dim}`, got {half.num_codebooks} and {half.codebook_dim} for "
                    f"{type(half).__name__}."
                )

    @property
    def codebook_size(self) -> int:
        return math.prod(self.quantization_levels)


@auto_docstring
@dataclass
class LiquidCodecEncoderModelOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_codebooks * codebook_dim, num_frames)`):
        Latents given to the FSQ quantizer, one per codec frame.
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
    istft_cache ([`LiquidCodecISTFTCache`], *optional*, returned when streaming):
        ISTFT cache, to pass with the next chunk; `flush()` ends the stream.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the decoder at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each attention layer of the decoder.
    """

    audio_values: torch.FloatTensor | None = None
    audio_values_mask: torch.BoolTensor | None = None
    decoder_cache: Cache | None = None
    istft_cache: "LiquidCodecISTFTCache | None" = None
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
        hidden_states = self.in_proj(input_features)
        hidden_states = self.act(hidden_states)

        # causal: drop the right padding
        num_frames = hidden_states.shape[-1]
        hidden_states = self.conv(hidden_states)[:, :, :num_frames]
        hidden_states = self.act(hidden_states)

        hidden_states = self.out_proj(hidden_states)
        return hidden_states.transpose(1, 2)


class LiquidCodecDecoderUpsampler(nn.Module):
    def __init__(self, config: LiquidCodecDecoderConfig):
        super().__init__()
        self.upsampling_factor = config.upsampling_factor
        self.proj = nn.Linear(config.quantization_dim, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, scale_factor=self.upsampling_factor, mode="nearest-exact")
        return hidden_states.transpose(1, 2)


class LiquidCodecFiniteScalarQuantization(Xcodec2FiniteScalarQuantization):
    pass


class LiquidCodecQuantizer(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_dim = len(config.quantization_levels)

        self.quantizer = LiquidCodecFiniteScalarQuantization(config)

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames = latents.shape
        latents = latents.transpose(1, 2).reshape(batch_size, num_frames, self.num_codebooks, self.codebook_dim)
        _, audio_codes = self.quantizer(latents)
        return audio_codes.transpose(1, 2).long()

    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames = audio_codes.shape
        # from the integer levels, not the `codebook` buffer: half precision rounds it
        latents = self.quantizer._indices_to_codes(audio_codes.transpose(1, 2))
        latents = latents.reshape(batch_size, num_frames, self.num_codebooks * self.codebook_dim)
        return latents.transpose(1, 2)


class LiquidCodecISTFTCache:
    """
    Streaming cache of [`LiquidCodecISTFTHead`]: the last `n_fft - hop_length` samples of a chunk, which the next
    chunk's frames still overlap. [`~LiquidCodecISTFTCache.flush`] ends the stream.

    Each chunk is normalized by its own window envelope. Over the overlap, the stream is the crossfade of both chunks,
    weighted by their share of the full envelope: it depends only on the window and the hop.
    """

    def __init__(self, config: LiquidCodecDecoderConfig):
        self.num_overlap_samples = config.n_fft - config.hop_length
        self.num_padding_samples = self.num_overlap_samples // 2

        # window envelope of the frames reaching into the overlap: a chunk's first ones cover its head, its last ones
        # its tail
        squared_window = torch.hann_window(config.n_fft).square()
        num_frames = math.ceil(self.num_overlap_samples / config.hop_length)
        window_envelope = F.fold(
            squared_window[None, :, None].expand(1, -1, num_frames),
            output_size=(1, (num_frames - 1) * config.hop_length + config.n_fft),
            kernel_size=(1, config.n_fft),
            stride=(1, config.hop_length),
        )[0, 0, 0]
        head_envelope = window_envelope[: self.num_overlap_samples]
        tail_envelope = window_envelope[-self.num_overlap_samples :]
        # each from its own envelope: `1 - fade_out` loses its smallest values to float32 rounding
        self.fade_out = tail_envelope / (tail_envelope + head_envelope)
        self.fade_in = head_envelope / (tail_envelope + head_envelope)

        self.cache: torch.Tensor | None = None
        self.is_initialized = False

    def lazy_initialization(self, audio: torch.Tensor):
        self.cache = audio.new_zeros(audio.shape[0], self.num_overlap_samples)
        self.fade_out = self.fade_out.to(audio)
        self.fade_in = self.fade_in.to(audio)

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cache)

        self.is_initialized = True

    def update(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Crossfades the cached samples with the start of a chunk, then caches its last ones. Returns the samples no
        later chunk overlaps.
        """
        if self.is_initialized:
            start = audio[:, : self.num_overlap_samples] * self.fade_in + self.cache * self.fade_out
            audio = torch.cat([start, audio[:, self.num_overlap_samples :]], dim=1)
        else:
            self.lazy_initialization(audio)
            # frames centred on hops: the stream starts `num_padding_samples` in
            audio = audio[:, self.num_padding_samples :]

        self.cache.copy_(audio[:, -self.num_overlap_samples :])
        return audio[:, : -self.num_overlap_samples]

    def flush(self) -> "LiquidCodecDecoderOutput":
        """Ends the stream: last samples, up to the padding, then resets the cache."""
        if not self.is_initialized:
            raise ValueError("Nothing to flush: no chunk was decoded since the cache was created or last flushed.")

        audio = self.cache[:, : self.num_padding_samples]

        self.reset()
        return LiquidCodecDecoderOutput(audio_values=audio)

    def reset(self):
        self.cache = None
        self.is_initialized = False


class LiquidCodecISTFTHead(Xcodec2ISTFTHead):
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        istft_cache: LiquidCodecISTFTCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # float32: polar/irfft don't support ComplexHalf. No clamp, unlike Xcodec2
        magnitude, phase = self.linear(hidden_states).float().chunk(2, dim=-1)
        spectrogram = torch.polar(magnitude.exp(), phase)

        # per clip: own frames, own window envelope, as alone. Streaming: the cache trims the stream's edges
        audio = self._istft(spectrogram, padding=istft_cache is None, mask=mask)
        if istft_cache is not None:
            audio = istft_cache.update(audio)

        if mask is None:
            return audio, None

        # a frame owns the `hop_length` samples at the center of its window
        samples_mask = mask.bool().repeat_interleave(self.hop_length, dim=-1)
        return audio * samples_mask, samples_mask

    def _istft(
        self, spectrogram: torch.Tensor, padding: bool = True, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Inverse STFT of `spectrogram` (`(batch_size, num_frames, n_fft // 2 + 1)`). With `padding`, frames are
        centred on hops: `num_frames * hop_length` samples; without, the edges are kept, for a stream chunk. With
        `mask`, masked frames are left out: each signal as if alone.
        """
        time_frames = torch.fft.irfft(spectrogram, self.n_fft, dim=-1, norm="backward") * self.window
        window_envelope = self.window.square().expand_as(time_frames)
        if mask is not None:
            mask = mask[..., None].to(time_frames.dtype)
            time_frames = time_frames * mask
            window_envelope = window_envelope * mask

        frames = torch.cat([time_frames, window_envelope]).transpose(1, 2)
        output_size = (spectrogram.shape[1] - 1) * self.hop_length + self.n_fft
        overlap_added = F.fold(
            frames, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length)
        )[:, 0, 0]
        # frames centred on hops: trim `(n_fft - hop_length) // 2` samples at each end, `num_frames * hop_length` left
        num_padding_samples = (self.n_fft - self.hop_length) // 2 if padding else 0
        overlap_added = overlap_added[:, num_padding_samples : output_size - num_padding_samples]
        audio, window_envelope = overlap_added.chunk(2)

        # Clamp as in Vocos: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L78
        return audio / window_envelope.clamp(min=1e-11)


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
    The encoder half of Liquid Codec: embedder, LFM3 layers and output projections, from spectrogram to the continuous
    latents of the FSQ quantizer.
    """
)
class LiquidCodecEncoderModel(Lfm3Model):
    config: LiquidCodecEncoderConfig

    def __init__(self, config: LiquidCodecEncoderConfig):
        super().__init__(config)
        self.subsampler = LiquidCodecEncoderSubsampler(config)
        self.output_proj = nn.Linear(config.hidden_size, config.quantization_dim)
        self.quantizer_in_proj = nn.Linear(config.quantization_dim, config.num_codebooks * config.codebook_dim)

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
        hidden_states = self.output_proj(hidden_states)
        latents = self.quantizer_in_proj(hidden_states)
        latents = latents.transpose(1, 2)

        latents_mask = None
        if input_features_mask is not None:
            num_frames = input_features_mask.sum(-1, keepdim=True) // self.config.subsampling_factor
            latents_mask = torch.arange(latents.shape[-1], device=latents.device) < num_frames.to(latents.device)

        return LiquidCodecEncoderModelOutput(latents=latents, latents_mask=latents_mask)


@auto_docstring(
    custom_intro="""
    The decoder half of Liquid Codec: input projection, upsampler, LFM3 layers and ISTFT head, from FSQ values to audio.
    """
)
class LiquidCodecDecoderModel(Lfm3Model):
    config: LiquidCodecDecoderConfig
    main_input_name = "latents"

    def __init__(self, config: LiquidCodecDecoderConfig):
        super().__init__(config)
        self.quantizer_out_proj = nn.Linear(config.num_codebooks * config.codebook_dim, config.quantization_dim)
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
        istft_cache: LiquidCodecISTFTCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecDecoderOutput:
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, num_codebooks * codebook_dim, num_frames)`):
            FSQ values from the quantizer, one per codec frame.
        latents_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid frames of each clip (right-padded): each clip decoded as alone; returns `audio_values_mask`.
        istft_cache ([`LiquidCodecISTFTCache`], *optional*):
            ISTFT cache from the previous chunk.
        """
        if istft_cache is not None and latents_mask is not None:
            raise ValueError(
                "Streaming decodes one stream per batch row: a frames mask (`audio_codes_mask`) is not supported."
            )

        latents = latents.transpose(1, 2).to(self.quantizer_out_proj.weight.dtype)
        hidden_states = self.quantizer_out_proj(latents)
        hidden_states = self.upsampler(hidden_states)

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

        mask = None
        if latents_mask is not None:
            mask = latents_mask.repeat_interleave(self.config.upsampling_factor, dim=-1)
        audio_values, audio_values_mask = self.head(hidden_states, mask=mask, istft_cache=istft_cache)

        return LiquidCodecDecoderOutput(
            audio_values=audio_values,
            audio_values_mask=audio_values_mask,
            decoder_cache=cache,
            istft_cache=istft_cache,
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
        istft_cache: LiquidCodecISTFTCache | None = None,
        use_streaming: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LiquidCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            Codes to decode; `hop_length * upsampling_factor` samples per frame.
        audio_codes_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid frames of each clip (right-padded): each clip decoded as alone; returns `audio_values_mask`. Not
            supported when streaming, which decodes one stream per batch row.
        decoder_cache (`Cache`, *optional*):
            Decoder cache from the previous chunk.
        istft_cache ([`LiquidCodecISTFTCache`], *optional*):
            ISTFT cache from the previous chunk.
        use_streaming (`bool`, *optional*, defaults to `False`):
            Decode chunk by chunk: caches created on the first chunk, returned with each. Trailing samples stay
            in `istft_cache` until `istft_cache.flush()`.

        Decodes the codes into audio: the codes are dequantized into the latents of [`LiquidCodecDecoderModel.forward`].
        """
        if self.decoder is None:
            raise ValueError(
                "Audio decoding was requested, but the model was initialized without a decoder (`decoder_config=None`)."
            )

        if use_streaming and decoder_cache is None:
            decoder_cache = DynamicCache(config=self.config.decoder_config)
        if use_streaming and istft_cache is None:
            istft_cache = LiquidCodecISTFTCache(config=self.config.decoder_config)

        latents = self.quantizer.decode(audio_codes)

        return self.decoder(
            latents,
            latents_mask=audio_codes_mask,
            cache=decoder_cache,
            istft_cache=istft_cache,
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
    "LiquidCodecISTFTCache",
    "LiquidCodecPreTrainedModel",
]
