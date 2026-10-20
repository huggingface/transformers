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

import math
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedAudioTokenizerBase, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.type_validators import activation_fn_key, interval, positive_int_field
from ..auto import AutoModel
from ..mimi.configuration_mimi import MimiConfig, MimiDecoderConfig, MimiEncoderConfig
from ..mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiDecoderModel,
    MimiDecoderOutput,
    MimiEncoderModel,
    MimiLayerScale,
    MimiPreTrainedModel,
)


# TODO: @eustlb seems #49735 broke conversion, this is a workaround
mono = interval(min=1, max=1)


@auto_docstring(checkpoint="eustlb/pocket-tts-audio-tokenizer")
@strict
class PocketTTSAudioTokenizerConfig(MimiConfig):
    r"""
    encoder_config (`None`, *optional*):
        Set to `None` to drop the encoder, e.g. to only decode latents. Otherwise it is derived from the other fields,
        which the encoder and the decoder share.
    decoder_config (`None`, *optional*):
        Set to `None` to drop the decoder, e.g. to only encode audio. Otherwise it is derived from the other fields.
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data; the released checkpoints, and this implementation, are mono.
    upsampling_ratios (`list[int]`, *optional*):
        Kernel size and stride ratios of the decoder's transposed convolutions; the encoder uses them in reverse order.
        If not specified, defaults to `[6, 5, 4]`.
    num_filters (`int`, *optional*, defaults to 64):
        Number of convolution kernels of the first `PocketTTSAudioTokenizerConv1d` layer of the encoder.
    num_residual_layers (`int`, *optional*, defaults to 1):
        Number of residual layers.
    last_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the last convolution layer.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each layer.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"constant"`):
        Padding mode for the convolutions.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
        equal to 1.0, it means that all the trimming is done at the right.
    use_conv_shortcut (`bool`, *optional*, defaults to `False`):
        Whether to use a convolution as the skip connection of the SEANet residual blocks; the released checkpoints do
        not.
    upsample_groups (`int`, *optional*, defaults to 512):
        Number of groups of the transposed convolution that upsamples the latents to the SEANet rate.
    latent_stride (`int`, *optional*, defaults to 16):
        Number of SEANet frames per latent frame: the stride of the encoder's convolution that downsamples the SEANet
        output to the latents, and of the decoder's transposed one that upsamples them back. A latent frame covers
        `prod(upsampling_ratios) * latent_stride` audio samples (`frame_size`, 1920 for the released checkpoints).
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous latents. Mimi's residual vector quantizer is replaced by this bottleneck: the
        encoder's downsampling convolution outputs `latent_dim` channels, and the decoder projects them back to
        `hidden_size` before upsampling.
    use_streaming (`bool`, *optional*, defaults to `False`):
        Whether the encoder streams by default: it then returns the padding cache and key-value cache to pass to the
        next call.

    Example:

    ```python
    >>> from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel

    >>> # a full codec, and its decoder alone
    >>> model = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig())
    >>> decoder_only = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig(encoder_config=None))
    ```
    """

    model_type = "pocket_tts_audio_tokenizer"

    latent_stride: int = positive_int_field(default=16)
    latent_dim: int = positive_int_field(default=32)
    audio_channels: int = mono(default=1)
    num_hidden_layers: int = positive_int_field(default=2)
    hidden_act: str = activation_fn_key(default="gelu_pytorch_tanh")

    # Mimi's residual vector quantizer is replaced by the `latent_dim` bottleneck
    codebook_size = AttributeError()
    codebook_dim = AttributeError()
    num_quantizers = AttributeError()
    num_semantic_quantizers = AttributeError()
    vector_quantization_hidden_dimension = AttributeError()
    tie_word_embeddings = AttributeError()

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.upsampling_ratios if self.upsampling_ratios else [6, 5, 4]
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        # set to `False` by `encoder_config=None` / `decoder_config=None`, see `encoder_config`
        self._has_encoder = self._has_decoder = True
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        raise AttributeError("Not needed for the Pocket TTS audio tokenizer")

    @property
    def num_codebooks(self):
        raise AttributeError("The Pocket TTS audio tokenizer has continuous latents, not codebooks.")

    @property
    def frame_size(self) -> int:
        """Number of audio samples per latent frame."""
        return math.prod(self.upsampling_ratios) * self.latent_stride

    @property
    def frame_rate(self) -> float:
        """Number of latent frames per second."""
        return self.sampling_rate / self.frame_size


# trf-ignore: TRF057
class PocketTTSAudioTokenizerEncoderConfig(MimiEncoderConfig):
    pass


# trf-ignore: TRF057
class PocketTTSAudioTokenizerDecoderConfig(MimiDecoderConfig):
    pass


class PocketTTSAudioTokenizerConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


@auto_docstring
@dataclass
class PocketTTSAudioTokenizerEncoderOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio, one frame per `config.frame_size` audio samples.
    latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
        Mask of the latent frames computed from audio samples (`1`) rather than padding (`0`).
    encoder_past_key_values (`Cache`, *optional*):
        Key-value cache of the encoder transformer, returned when streaming.
    padding_cache (`PocketTTSAudioTokenizerConv1dPaddingCache`, *optional*):
        Padding cache of the encoder's causal convolutions, returned when streaming.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder transformer at the output of each layer plus its input, at the SEANet frame rate.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each encoder transformer layer.
    """

    latents: torch.FloatTensor | None = None
    latents_mask: torch.LongTensor | None = None
    encoder_past_key_values: Cache | None = None
    padding_cache: PocketTTSAudioTokenizerConv1dPaddingCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class PocketTTSAudioTokenizerDecoderOutput(MimiDecoderOutput):
    pass


@auto_docstring
@dataclass
class PocketTTSAudioTokenizerOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio.
    latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
        Mask of the latent frames computed from audio samples (`1`) rather than padding (`0`).
    audio_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
        Audio waveform reconstructed from the latents.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder transformer at the output of each layer plus its input.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each encoder transformer layer.
    """

    latents: torch.FloatTensor | None = None
    latents_mask: torch.LongTensor | None = None
    audio_values: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class PocketTTSAudioTokenizerConv1d(MimiConv1d):
    pass


class PocketTTSAudioTokenizerLayerScale(MimiLayerScale):
    pass


@auto_docstring
class PocketTTSAudioTokenizerPreTrainedModel(MimiPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        # Mimi's, without its residual vector quantizer's codebooks
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, PocketTTSAudioTokenizerLayerScale):
            init.constant_(module.scale, self.config.layer_scale_initial_scale)
        elif isinstance(module, PocketTTSAudioTokenizerConv1d):
            kernel_size = module.conv.kernel_size[0]
            stride = module.conv.stride[0]
            dilation = module.conv.dilation[0]
            kernel_size = (kernel_size - 1) * dilation + 1
            init.constant_(module.stride, stride)
            init.constant_(module.kernel_size, kernel_size)
            init.constant_(module.padding_total, kernel_size - stride)


class PocketTTSAudioTokenizerEncoderModel(MimiEncoderModel):
    _keys_to_ignore_on_load_unexpected = [r"^decoder\.", r"^latent_proj\."]


class PocketTTSAudioTokenizerDecoderModel(MimiDecoderModel):
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^latent_proj\."]


@auto_docstring(
    custom_intro="""
    The Pocket TTS audio tokenizer: a Mimi-architecture codec with a continuous 32-dimensional latent at 12.5 Hz (the
    VAE-GAN of Continuous Audio Language Models), turning 24 kHz audio into latents and back.
    """
)
class PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerPreTrainedModel, PreTrainedAudioTokenizerBase):
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^decoder\."]

    def __init__(self, config: PocketTTSAudioTokenizerConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config) if config.encoder_config is not None else None
        self.decoder = AutoModel.from_config(config.decoder_config) if config.decoder_config is not None else None
        self.latent_proj = nn.Conv1d(config.latent_dim, config.hidden_size, kernel_size=1, bias=False)
        self.post_init()

    @can_return_tuple
    def encode(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.Tensor | None = None,
        encoder_past_key_values: Cache | None = None,
        padding_cache: PocketTTSAudioTokenizerConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PocketTTSAudioTokenizerEncoderOutput:
        """
        Encodes audio into continuous latents of shape `(batch_size, num_frames, latent_dim)`: the latents of
        [`PocketTTSAudioTokenizerEncoderModel.forward`], frames first.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
                Audio waveform, as returned by [`EncodecFeatureExtractor`], which zero-pads it to a multiple of
                `config.frame_size` samples as the original does.
            padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
                Mask of the audio samples (`1`) as opposed to padding (`0`), from which `latents_mask` is computed.
            encoder_past_key_values (`Cache`, *optional*):
                Key-value cache of the encoder transformer, returned by the previous call when streaming.
            padding_cache (`PocketTTSAudioTokenizerConv1dPaddingCache`, *optional*):
                Padding cache of the encoder's causal convolutions, returned by the previous call when streaming.
            use_streaming (`bool`, *optional*):
                Whether to encode in streaming mode, returning the caches to pass to the next call.
        """
        if self.encoder is None:
            raise ValueError(
                "Audio encoding was requested, but the model was initialized without an encoder (`encoder_config=None`)."
            )
        encoder_outputs = self.encoder(
            input_values,
            past_key_values=encoder_past_key_values,
            padding_cache=padding_cache,
            use_streaming=use_streaming,
            return_dict=True,
            **kwargs,
        )
        latents = encoder_outputs.latents.transpose(1, 2)

        latents_mask = None
        if padding_mask is not None:
            num_frames = self.encoder.get_encoded_length(padding_mask.sum(-1)).to(latents.device)
            latents_mask = (torch.arange(latents.shape[1], device=latents.device) < num_frames[:, None]).long()

        return PocketTTSAudioTokenizerEncoderOutput(
            latents=latents,
            latents_mask=latents_mask,
            encoder_past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @can_return_tuple
    def decode(
        self,
        latents: torch.FloatTensor,
        decoder_past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PocketTTSAudioTokenizerDecoderOutput:
        """
        Decodes continuous latents into audio: the latents are projected to `hidden_size`, then decoded by
        [`PocketTTSAudioTokenizerDecoderModel.forward`].

        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
                Continuous latents to decode, one frame per `config.frame_size` audio samples.
            decoder_past_key_values (`Cache`, *optional*):
                Key-value cache of the decoder transformer, returned by the previous call when decoding in chunks.
        """
        if self.decoder is None:
            raise ValueError(
                "Audio decoding was requested, but the model was initialized without a decoder (`decoder_config=None`)."
            )
        return self.decoder(
            self.latent_proj(latents.transpose(1, 2)),
            past_key_values=decoder_past_key_values,
            return_dict=True,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PocketTTSAudioTokenizerOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
            Audio waveform, as returned by [`EncodecFeatureExtractor`].
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask of the audio samples (`1`) as opposed to padding (`0`).
        """
        # the recorded hidden states and attentions are the encoder's, as for an encoder-decoder model's encoder outputs
        encoder_outputs = self.encode(input_values, padding_mask=padding_mask, return_dict=True, **kwargs)
        audio_values = self.decode(encoder_outputs.latents, return_dict=True).audio_values
        return PocketTTSAudioTokenizerOutput(
            latents=encoder_outputs.latents,
            latents_mask=encoder_outputs.latents_mask,
            audio_values=audio_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "PocketTTSAudioTokenizerConfig",
    "PocketTTSAudioTokenizerEncoderConfig",
    "PocketTTSAudioTokenizerDecoderConfig",
    "PocketTTSAudioTokenizerPreTrainedModel",
    "PocketTTSAudioTokenizerEncoderModel",
    "PocketTTSAudioTokenizerDecoderModel",
    "PocketTTSAudioTokenizerModel",
]
