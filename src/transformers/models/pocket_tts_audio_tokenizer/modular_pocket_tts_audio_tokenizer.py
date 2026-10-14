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
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedAudioTokenizerBase, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..mimi.modeling_mimi import (
    MimiAttention,
    MimiConv1d,
    MimiConvTranspose1d,
    MimiDecoder,
    MimiEncoder,
    MimiLayerScale,
    MimiMLP,
    MimiPreTrainedModel,
    MimiRotaryEmbedding,
    MimiTransformerLayer,
    MimiTransformerModel,
)
from .configuration_pocket_tts_audio_tokenizer import (
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerDecoderConfig,
    PocketTTSAudioTokenizerEncoderConfig,
)


@auto_docstring
@dataclass
class PocketTTSAudioTokenizerEncoderOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
        Continuous latents of the audio, one frame per `config.frame_size` audio samples.
    latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
        Mask of the latent frames computed from audio samples (`1`) rather than padding (`0`).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the encoder transformer at the output of each layer plus its input, at the SEANet frame rate.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each encoder transformer layer.
    """

    latents: torch.FloatTensor | None = None
    latents_mask: torch.LongTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class PocketTTSAudioTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
        Decoded audio waveform.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Hidden states of the decoder transformer at the output of each layer plus its input, at the SEANet frame rate.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Attention weights of each decoder transformer layer.
    """

    audio_values: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


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


class PocketTTSAudioTokenizerConvTranspose1d(MimiConvTranspose1d):
    pass


class PocketTTSAudioTokenizerResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by Mimi. Difference to Mimi: no optional convolutional shortcut, the released
    checkpoints have none.
    """

    def __init__(self, config: PocketTTSAudioTokenizerEncoderConfig, dim: int, dilations: list[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [PocketTTSAudioTokenizerConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

    def forward(self, hidden_states, padding_cache=None):
        residual = hidden_states
        for layer in self.block:
            if isinstance(layer, PocketTTSAudioTokenizerConv1d):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return residual + hidden_states


class PocketTTSAudioTokenizerSeanetEncoder(MimiEncoder):
    """SEANet encoder as used by Mimi."""

    # Difference to Mimi: none in the layers; `__init__` is rewritten only because the inherited one names its list of
    # convolution layers after the class prefix, which the modular converter turns into an unreadable identifier.
    def __init__(self, config: PocketTTSAudioTokenizerEncoderConfig):
        nn.Module.__init__(self)
        model = [PocketTTSAudioTokenizerConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # names of the convolution submodules, which index the padding cache
        conv_layer_names = ["layers.0"]

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                conv_layer_names.extend([f"layers.{len(model)}.block.1", f"layers.{len(model)}.block.3"])
                model += [
                    PocketTTSAudioTokenizerResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])
                ]
            # Add downsampling layers
            model += [nn.ELU()]
            conv_layer_names.append(f"layers.{len(model)}")
            model += [
                PocketTTSAudioTokenizerConv1d(
                    config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio
                )
            ]
            scaling *= 2

        model += [nn.ELU()]
        conv_layer_names.append(f"layers.{len(model)}")
        model += [
            PocketTTSAudioTokenizerConv1d(
                config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size
            )
        ]

        self.layers = nn.ModuleList(model)
        for layer_idx, layer_name in enumerate(conv_layer_names):
            self.get_submodule(layer_name).layer_idx = layer_idx


class PocketTTSAudioTokenizerSeanetDecoder(MimiDecoder):
    """SEANet decoder as used by Mimi."""


class PocketTTSAudioTokenizerLayerScale(MimiLayerScale):
    pass


class PocketTTSAudioTokenizerRotaryEmbedding(MimiRotaryEmbedding):
    pass


class PocketTTSAudioTokenizerMLP(MimiMLP):
    pass


class PocketTTSAudioTokenizerAttention(MimiAttention):
    pass


class PocketTTSAudioTokenizerTransformerLayer(MimiTransformerLayer):
    pass


class PocketTTSAudioTokenizerTransformerModel(MimiTransformerModel):
    pass


@auto_docstring
class PocketTTSAudioTokenizerPreTrainedModel(MimiPreTrainedModel):
    config: PocketTTSAudioTokenizerConfig
    base_model_prefix = "pocket_tts_audio_tokenizer"
    main_input_name = "input_values"
    _no_split_modules = ["PocketTTSAudioTokenizerTransformerLayer"]
    _can_record_outputs = {
        "hidden_states": PocketTTSAudioTokenizerTransformerLayer,
        "attentions": PocketTTSAudioTokenizerAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
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


@auto_docstring(
    custom_intro="""
    The encoder half of the Pocket TTS audio tokenizer: SEANet encoder, encoder transformer and a strided convolution
    down to the continuous latents (Mimi's encoder with the quantizer replaced by a `latent_dim` bottleneck).
    """
)
class PocketTTSAudioTokenizerEncoderModel(PocketTTSAudioTokenizerPreTrainedModel):
    config: PocketTTSAudioTokenizerEncoderConfig
    # loaded from a full codec checkpoint: its `encoder.` prefix is stripped, and the decoder's weights are not this half's
    base_model_prefix = "encoder"
    _keys_to_ignore_on_load_unexpected = [r"^decoder\."]

    def __init__(self, config: PocketTTSAudioTokenizerEncoderConfig):
        super().__init__(config)
        self.seanet_encoder = PocketTTSAudioTokenizerSeanetEncoder(config)
        self.encoder_transformer = PocketTTSAudioTokenizerTransformerModel(config)
        # Difference to Mimi: the stride is the full SEANet-to-latent ratio (16 for the released checkpoints), not 2, and
        # the convolution outputs the latents directly.
        self.downsample = PocketTTSAudioTokenizerConv1d(
            config,
            config.hidden_size,
            config.latent_dim,
            kernel_size=2 * config.latent_stride,
            stride=config.latent_stride,
            bias=False,
            pad_mode="replicate",
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PocketTTSAudioTokenizerEncoderOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`):
            Audio waveform, as returned by [`EncodecFeatureExtractor`]. It is zero-padded to a multiple of
            `config.frame_size` samples, as the original does.
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask of the audio samples (`1`) as opposed to padding (`0`). The convolutions and the transformer are causal,
            so right padding does not change the frames computed from audio; the mask only gives `latents_mask`.
        """
        input_values = nn.functional.pad(input_values, (0, -input_values.shape[-1] % self.config.frame_size))
        hidden_states = self.seanet_encoder(input_values)
        hidden_states = self.encoder_transformer(
            hidden_states.transpose(1, 2), return_dict=True, **kwargs
        ).last_hidden_state
        latents = self.downsample(hidden_states.transpose(1, 2)).transpose(1, 2)

        latents_mask = None
        if padding_mask is not None:
            num_frames = torch.ceil(padding_mask.sum(-1) / self.config.frame_size).to(latents.device)
            latents_mask = (torch.arange(latents.shape[1], device=latents.device) < num_frames[:, None]).long()
        return PocketTTSAudioTokenizerEncoderOutput(latents=latents, latents_mask=latents_mask)


@auto_docstring(
    custom_intro="""
    The decoder half of the Pocket TTS audio tokenizer: projection of the continuous latents, transposed convolution up
    to the SEANet rate, decoder transformer and SEANet decoder.
    """
)
class PocketTTSAudioTokenizerDecoderModel(PocketTTSAudioTokenizerPreTrainedModel):
    config: PocketTTSAudioTokenizerDecoderConfig
    main_input_name = "latents"
    # loaded from a full codec checkpoint: its `decoder.` prefix is stripped, and the encoder's weights are not this half's
    base_model_prefix = "decoder"
    _keys_to_ignore_on_load_unexpected = [r"^encoder\."]

    def __init__(self, config: PocketTTSAudioTokenizerDecoderConfig):
        super().__init__(config)
        # Mimi's residual vector quantizer output projection, applied directly to the continuous latents.
        self.latent_proj = nn.Conv1d(config.latent_dim, config.hidden_size, kernel_size=1, bias=False)
        # Difference to Mimi: the stride is the full latent-to-SEANet ratio (16 for the released checkpoints), not 2.
        self.upsample = PocketTTSAudioTokenizerConvTranspose1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * config.latent_stride,
            stride=config.latent_stride,
            bias=False,
            groups=config.upsample_groups,
        )
        self.decoder_transformer = PocketTTSAudioTokenizerTransformerModel(config)
        self.seanet_decoder = PocketTTSAudioTokenizerSeanetDecoder(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, latents: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> PocketTTSAudioTokenizerDecoderOutput:
        r"""
        latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
            Continuous latents to decode, one frame per `config.frame_size` audio samples.
        """
        hidden_states = self.upsample(self.latent_proj(latents.transpose(1, 2)))
        hidden_states = self.decoder_transformer(
            hidden_states.transpose(1, 2), return_dict=True, **kwargs
        ).last_hidden_state
        audio_values = self.seanet_decoder(hidden_states.transpose(1, 2))
        return PocketTTSAudioTokenizerDecoderOutput(audio_values=audio_values)


@auto_docstring(
    custom_intro="""
    The Pocket TTS audio tokenizer: a Mimi-architecture codec with a continuous 32-dimensional latent at 12.5 Hz (the
    VAE-GAN of Continuous Audio Language Models), turning 24 kHz audio into latents and back.
    """
)
class PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerPreTrainedModel, PreTrainedAudioTokenizerBase):
    def __init__(self, config: PocketTTSAudioTokenizerConfig):
        super().__init__(config)
        # A half whose config is `None` is not built (a decoder-only checkpoint, or one half of a full checkpoint loaded
        # alone): its weights in a full checkpoint are ignored.
        self.encoder = AutoModel.from_config(config.encoder_config) if config.encoder_config is not None else None
        self.decoder = AutoModel.from_config(config.decoder_config) if config.decoder_config is not None else None
        absent = [name for name in ("encoder", "decoder") if getattr(self, name) is None]
        if absent:
            patterns = {rf"^{name}\." for name in absent}
            self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or []) | patterns
        self.post_init()

    def encode(
        self, input_values: torch.FloatTensor, padding_mask: torch.Tensor | None = None, **kwargs
    ) -> PocketTTSAudioTokenizerEncoderOutput:
        """Encodes audio into continuous latents, see [`PocketTTSAudioTokenizerEncoderModel.forward`]."""
        if self.encoder is None:
            raise ValueError(
                "This audio tokenizer has no encoder (`config.encoder_config` is `None`): it can only decode."
            )
        return self.encoder(input_values, padding_mask=padding_mask, **kwargs)

    def decode(self, latents: torch.FloatTensor, **kwargs) -> PocketTTSAudioTokenizerDecoderOutput:
        """Decodes continuous latents into audio, see [`PocketTTSAudioTokenizerDecoderModel.forward`]."""
        if self.decoder is None:
            raise ValueError(
                "This audio tokenizer has no decoder (`config.decoder_config` is `None`): it can only encode."
            )
        return self.decoder(latents, **kwargs)

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
    "PocketTTSAudioTokenizerPreTrainedModel",
    "PocketTTSAudioTokenizerEncoderModel",
    "PocketTTSAudioTokenizerDecoderModel",
    "PocketTTSAudioTokenizerModel",
]
