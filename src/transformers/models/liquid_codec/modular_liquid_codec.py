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
"""Liquid Codec: a 24 kHz speech codec with finite scalar quantization into 12 codebooks at 12.5 frames per second."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import ModelOutput, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.import_utils import is_tracing
from ..auto import AutoConfig, AutoModel
from ..xcodec2.modeling_xcodec2 import Xcodec2FiniteScalarQuantization


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B-Preview")
@strict
class LiquidCodecConfig(PreTrainedConfig):
    r"""
    Configuration of the [`LiquidCodecModel`] bundled with LFM3-Audio-2B and LFM3-Omni-2B in their
    `audio_tokenizer/` directory.

    encoder_config (`dict` or [`Lfm3Config`], *optional*):
        The causal LFM3 model over the stacked STFT frames; its `hidden_size` is the width of the codec.
    decoder_config (`dict` or [`Lfm3Config`], *optional*):
        The causal LFM3 model that synthesizes the STFT frames from the codes; its `sliding_window` is the decoder
        context in 75 Hz frames.
    encoder_conv_dim (`int`, *optional*, defaults to 256):
        The number of channels of the STFT front end.
    stft_n_fft (`int`, *optional*, defaults to 1280):
        The FFT size of the analysis and synthesis transforms.
    stft_hop_length (`int`, *optional*, defaults to 320):
        The hop of the analysis and synthesis transforms.
    stft_win_length (`int`, *optional*, defaults to 1280):
        The window length of the analysis and synthesis transforms.
    stft_frame_stack (`int`, *optional*, defaults to 6):
        The number of STFT frames per codec frame.
    quantizer_dim (`int`, *optional*, defaults to 256):
        The width of the latent handed to the quantizer.
    fsq_levels (`list[int]`, *optional*, defaults to `[8, 7, 6, 6]`):
        The number of levels of each finite-scalar-quantization dimension; their product is the codebook size.
    n_q (`int`, *optional*, defaults to 12):
        The number of codebooks.
    sampling_rate (`int`, *optional*, defaults to 24000):
        The sampling rate of the audio.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal initializer of the weights.
    """

    model_type = "liquid_codec"
    attribute_map = {"quantization_levels": "fsq_levels"}
    _backbone_kwargs = {
        "vocab_size": 1,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "hidden_size": 512,
        "intermediate_size": 2304,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 131_072,
        "use_cache": False,
        "tie_word_embeddings": False,
    }
    sub_configs_defaults = {
        "encoder_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="lfm3",
            init_kwargs={
                **_backbone_kwargs,
                "layer_types": [
                    "conv",
                    "conv",
                    "full_attention",
                    "conv",
                    "full_attention",
                    "conv",
                    "full_attention",
                    "conv",
                ],
                "sliding_window": None,
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
                },
            },
        ),
        "decoder_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="lfm3",
            init_kwargs={
                **_backbone_kwargs,
                "layer_types": [
                    "conv",
                    "conv",
                    "sliding_attention",
                    "conv",
                    "sliding_attention",
                    "conv",
                    "sliding_attention",
                    "conv",
                ],
                "sliding_window": 30,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 1.0,
                    }
                },
            },
        ),
    }

    encoder_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    encoder_conv_dim: int = 256
    stft_n_fft: int = 1280
    stft_hop_length: int = 320
    stft_win_length: int = 1280
    stft_frame_stack: int = 6
    quantizer_dim: int = 256
    fsq_levels: list[int] | None = None
    n_q: int = 12
    sampling_rate: int = 24_000
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.fsq_levels is None:
            self.fsq_levels = [8, 7, 6, 6]
        self.fsq_levels = list(self.fsq_levels)
        super().__post_init__(**kwargs)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def frame_size(self) -> int:
        return self.stft_hop_length * self.stft_frame_stack


class LiquidCodecOverlapAddCache:
    """The unfinished tail of a streamed synthesis: the overlap-add numerator and the window-envelope denominator of
    the last `win_length - hop_length` samples, which the frames of the next chunk still add to. Both are `None`
    before the first frame and after the final flush."""

    def __init__(self):
        self.audio: torch.Tensor | None = None
        self.envelope: torch.Tensor | None = None


@auto_docstring
@dataclass
class LiquidCodecEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        The code of every codebook of every frame.
    """

    audio_codes: torch.LongTensor = None


@auto_docstring
@dataclass
class LiquidCodecDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        The decoded 24 kHz audio in float32.
    decoder_past_key_values (`Cache`, *optional*, returned when streaming):
        The decoder LFM3 cache to pass to the next chunk of the stream.
    overlap_cache ([`LiquidCodecOverlapAddCache`], *optional*, returned when streaming):
        The unfinished overlap-add tail to pass to the next chunk of the stream.
    """

    audio_values: torch.FloatTensor = None
    decoder_past_key_values: Cache | None = None
    overlap_cache: LiquidCodecOverlapAddCache | None = None


@auto_docstring
@dataclass
class LiquidCodecOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        The reconstructed audio, cut to the input length.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        The code of every codebook of every frame.
    """

    audio_values: torch.FloatTensor = None
    audio_codes: torch.LongTensor = None


class LiquidCodecSTFT(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.win_length = config.stft_win_length
        self.window = nn.Buffer(torch.hann_window(self.win_length, dtype=torch.float32), persistent=False)
        self.num_features = 3 * (self.n_fft // 2 + 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        total_pad = self.win_length - self.hop_length
        audio = F.pad(audio.float(), (total_pad // 2, total_pad - total_pad // 2), mode="reflect")
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.float(),
            center=False,
            return_complex=True,
        )
        magnitude = spec.abs()
        phase_denominator = magnitude.clamp_min(1e-5)
        return torch.cat((torch.log1p(magnitude), spec.real / phase_denominator, spec.imag / phase_denominator), dim=1)


class LiquidCodecFrontEnd(nn.Module):
    def __init__(self, config: LiquidCodecConfig, num_features: int):
        super().__init__()
        self.frame_stack = config.stft_frame_stack
        self.in_proj = nn.Conv1d(num_features, config.encoder_conv_dim, kernel_size=1)
        self.conv = nn.Conv1d(config.encoder_conv_dim, config.encoder_conv_dim, kernel_size=3, padding=2)
        self.out = nn.Conv1d(
            config.encoder_conv_dim,
            config.encoder_config.hidden_size,
            kernel_size=self.frame_stack,
            stride=self.frame_stack,
        )
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.in_proj(features.to(self.in_proj.weight.dtype)))
        hidden_states = self.act(self.conv(hidden_states)[:, :, : hidden_states.shape[-1]])
        return self.out(hidden_states)


class LiquidCodecFiniteScalarQuantization(Xcodec2FiniteScalarQuantization):
    """Xcodec2's finite scalar quantization over `n_q` codebooks, with the latent projected down to the codebook
    dimensions and back."""

    def __init__(self, config: LiquidCodecConfig):
        super().__init__(config)
        self.num_codebooks = config.n_q
        self.codebook_dim = len(self.quantization_levels)
        self.project_in = nn.Linear(config.quantizer_dim, self.num_codebooks * self.codebook_dim)
        self.project_out = nn.Linear(self.num_codebooks * self.codebook_dim, config.quantizer_dim)

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.project_in(latents.movedim(1, -1).contiguous()).float()
        if not is_tracing(latents):
            # finite audio far above full scale overflows the float32 STFT, and NaN latents round to INT64_MIN codes
            torch_compilable_check(
                torch.isfinite(latents),
                "The audio overflows the codec (non-finite latents); expected samples in [-1, 1]",
            )
        _, indices = self(latents.reshape(*latents.shape[:-1], self.num_codebooks, self.codebook_dim))
        return indices.long().transpose(1, 2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        latents = self._indices_to_codes(codes.transpose(1, 2))
        latents = latents.reshape(*latents.shape[:-2], self.num_codebooks * self.codebook_dim)
        return self.project_out(latents.to(self.project_out.weight.dtype)).movedim(-1, 1)


class LiquidCodecISTFT(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.win_length = config.stft_win_length
        self.window = nn.Buffer(torch.hann_window(self.win_length, dtype=torch.float32))
        # samples a frame shares with its neighbours: trimmed at both ends of a clip, held back between chunks
        self.overlap = self.win_length - self.hop_length

    def forward(
        self, spec: torch.Tensor, overlap_cache: LiquidCodecOverlapAddCache | None = None, finalize: bool = True
    ) -> torch.Tensor:
        batch_size, num_frames = spec.shape[0], spec.shape[-1]
        trim = self.overlap // 2
        if num_frames:
            frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
            output_size = (num_frames - 1) * self.hop_length + self.win_length
            fold = {
                "output_size": (1, output_size),
                "kernel_size": (1, self.win_length),
                "stride": (1, self.hop_length),
            }
            audio = F.fold(frames, **fold)[:, 0, 0]
            envelope = F.fold(self.window.square().expand(1, num_frames, -1).transpose(1, 2), **fold)[0, 0, 0]
        elif overlap_cache is None or overlap_cache.audio is None or not finalize:
            return self.window.new_zeros((batch_size, 0))
        else:
            # the final flush of a stream: only the held-back tail is left to release
            audio, envelope = self.window.new_zeros((batch_size, self.overlap)), self.window.new_zeros(self.overlap)
        if overlap_cache is None:
            audio, envelope = audio[:, trim:-trim], envelope[trim:-trim]
        else:
            if overlap_cache.audio is None:
                audio, envelope = audio[:, trim:], envelope[trim:]
            else:
                audio = torch.cat([audio[:, : self.overlap] + overlap_cache.audio, audio[:, self.overlap :]], dim=1)
                envelope = torch.cat([envelope[: self.overlap] + overlap_cache.envelope, envelope[self.overlap :]])
            if finalize:
                audio, envelope = audio[:, :-trim], envelope[:-trim]
                overlap_cache.audio, overlap_cache.envelope = None, None
            else:
                overlap_cache.audio, overlap_cache.envelope = audio[:, -self.overlap :], envelope[-self.overlap :]
                audio, envelope = audio[:, : -self.overlap], envelope[: -self.overlap]
        return (audio / envelope).float()


class LiquidCodecDecoder(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.frame_stack = config.stft_frame_stack
        self.lfm = AutoModel.from_config(config.decoder_config)
        self.lin = nn.Linear(config.decoder_config.hidden_size, 2 * (config.stft_n_fft // 2 + 1))
        self.istft = LiquidCodecISTFT(config)

    def forward(
        self,
        embeddings: torch.Tensor,
        past_key_values: Cache | None = None,
        overlap_cache: LiquidCodecOverlapAddCache | None = None,
        finalize: bool = True,
    ) -> torch.Tensor:
        if embeddings.shape[1]:
            hidden_states = self.lfm(
                inputs_embeds=F.interpolate(
                    embeddings.mT, embeddings.shape[1] * self.frame_stack, mode="nearest-exact"
                ).mT,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            ).last_hidden_state
            log_magnitude, angle = torch.chunk(self.lin(hidden_states).mT.contiguous(), 2, 1)
            spec = torch.polar(log_magnitude.float().exp(), angle.float())
        else:
            spec = embeddings.new_zeros((embeddings.shape[0], self.lin.out_features // 2, 0), dtype=torch.complex64)
        return self.istft(spec, overlap_cache, finalize)


@auto_docstring(
    custom_intro="The Liquid Codec neural audio codec: 24 kHz audio to 12 codebooks at 12.5 frames per second and back."
)
class LiquidCodecModel(PreTrainedAudioTokenizerBase):
    config: LiquidCodecConfig
    base_model_prefix = "codec"
    main_input_name = "input_values"
    # the analysis window is a non-persistent buffer cast in the forward; only the synthesis window is a weight
    _keep_in_fp32_modules_strict = ["istft"]
    _supports_flash_attn = False
    _supports_sdpa = True

    def __init__(self, config: LiquidCodecConfig):
        super().__init__(config)
        self.stft = LiquidCodecSTFT(config)
        self.frontend = LiquidCodecFrontEnd(config, self.stft.num_features)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.to_quantizer = nn.Linear(config.encoder_config.hidden_size, config.quantizer_dim)
        self.quantizer = LiquidCodecFiniteScalarQuantization(config)
        self.decoder = LiquidCodecDecoder(config)
        self.from_quantizer = nn.Linear(config.quantizer_dim, config.decoder_config.hidden_size)
        # the backbones only ever see `inputs_embeds`: their token embedding is not a codec weight
        del self.encoder.embed_tokens, self.decoder.lfm.embed_tokens
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (LiquidCodecSTFT, LiquidCodecISTFT)):
            init.copy_(module.window, torch.hann_window(module.win_length, dtype=torch.float32))
        elif isinstance(module, LiquidCodecFiniteScalarQuantization):
            levels, basis, codebook = module._compute_buffers(device=module.levels.device)
            init.copy_(module.levels, levels)
            init.copy_(module.basis, basis)
            init.copy_(module.codebook, codebook)

    @can_return_tuple
    def encode(self, input_values: torch.FloatTensor, *, return_dict: bool | None = None) -> LiquidCodecEncoderOutput:
        r"""
        Non-finite audio, or audio that overflows the codec, is refused eagerly; a compiled `encode` skips the check.

        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            24 kHz audio. The tail is zero-padded to a whole codec frame of 1920 samples.
        return_dict (`bool`, *optional*):
            Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if input_values.ndim != 2 or input_values.shape[-1] == 0:
            raise ValueError(
                f"Expected non-empty audio of shape (batch_size, num_samples), got {tuple(input_values.shape)}"
            )
        if not is_tracing(input_values):
            # a non-finite sample would round to an INT64_MIN code that only fails far downstream; the tensor-valued
            # check would break a `fullgraph` compile or a CUDA-graph capture
            torch_compilable_check(torch.isfinite(input_values), "`input_values` must be finite")
        frame_size = self.config.frame_size
        num_samples = -(input_values.shape[-1] // -frame_size) * frame_size
        input_values = F.pad(input_values, (0, num_samples - input_values.shape[-1]))
        hidden_states = self.frontend(self.stft(input_values)).mT
        latents = self.to_quantizer(self.encoder(inputs_embeds=hidden_states).last_hidden_state).mT
        return LiquidCodecEncoderOutput(audio_codes=self.quantizer.encode(latents))

    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.LongTensor,
        decoder_past_key_values: Cache | None = None,
        overlap_cache: LiquidCodecOverlapAddCache | None = None,
        use_streaming: bool = False,
        finalize: bool = True,
        *,
        return_dict: bool | None = None,
    ) -> LiquidCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            The codes to decode; every frame becomes 1920 samples.
        decoder_past_key_values (`Cache`, *optional*):
            The decoder LFM3 cache returned by the previous chunk of the stream.
        overlap_cache ([`LiquidCodecOverlapAddCache`], *optional*):
            The overlap-add tail returned by the previous chunk of the stream.
        use_streaming (`bool`, *optional*, defaults to `False`):
            Whether to decode chunk by chunk: the caches are created on the first chunk and returned with every
            chunk, which decode to the same audio as the concatenated codes.
        finalize (`bool`, *optional*, defaults to `True`):
            Whether this is the last chunk of the stream. Until then the trailing samples that later frames still
            contribute to are held back in `overlap_cache`.
        return_dict (`bool`, *optional*):
            Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if audio_codes.ndim != 3 or audio_codes.shape[1] != self.config.n_q:
            raise ValueError(
                f"Expected codes of shape (batch_size, {self.config.n_q}, num_frames), got {tuple(audio_codes.shape)}"
            )
        torch_compilable_check(
            (audio_codes >= 0) & (audio_codes < self.config.codebook_size),
            f"Codes must be in [0, {self.config.codebook_size})",
        )
        if use_streaming and decoder_past_key_values is None:
            decoder_past_key_values = DynamicCache(config=self.config.decoder_config)
        if use_streaming and overlap_cache is None:
            overlap_cache = LiquidCodecOverlapAddCache()
        embeddings = self.from_quantizer(self.quantizer.decode(audio_codes).mT)
        audio = self.decoder(embeddings, decoder_past_key_values, overlap_cache, finalize)
        return LiquidCodecDecoderOutput(
            audio_values=audio, decoder_past_key_values=decoder_past_key_values, overlap_cache=overlap_cache
        )

    @can_return_tuple
    @auto_docstring
    def forward(self, input_values: torch.FloatTensor) -> LiquidCodecOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            24 kHz audio.
        """
        audio_codes = self.encode(input_values, return_dict=True).audio_codes
        audio = self.decode(audio_codes, return_dict=True).audio_values
        return LiquidCodecOutput(audio_values=audio[..., : input_values.shape[-1]], audio_codes=audio_codes)


__all__ = ["LiquidCodecConfig", "LiquidCodecModel", "LiquidCodecOverlapAddCache"]
