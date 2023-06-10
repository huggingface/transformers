# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EnCodec model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    logging,
)
from .configuration_encodec import EncodecConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "EncodecConfig"


ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Matthijs/encodec_24khz",
    "Matthijs/encodec_48khz",
    # See all EnCodec models at https://huggingface.co/models?filter=encodec
]


@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        # TODO

    """

    audio_codes: torch.FloatTensor = None
    audio_values: torch.FloatTensor = None


@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        # TODO

    """

    audio_codes: torch.FloatTensor = None
    audio_scales: Optional[torch.Tensor] = None


@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        # TODO
    """

    audio_values: Optional[torch.FloatTensor] = None


class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        pad_mode: str = "reflect",
    ):
        super().__init__()

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "EncodecConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.norm_type = norm
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        self.causal = causal
        self.pad_mode = pad_mode

    @staticmethod
    def _get_extra_padding_for_conv1d(
        hidden_states: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
    ) -> int:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happen.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            extra_pad = 0
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
            padded = nn.functional.pad(hidden_states, paddings, mode, value)
            end = padded.shape[-1] - extra_pad
            return padded[..., :end]
        else:
            return nn.functional.pad(hidden_states, paddings, mode, value)

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride

        ehidden_statestra_padding = self._get_extra_padding_for_conv1d(
            hidden_states, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(hidden_states, (padding_total, ehidden_statestra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states, (padding_left, padding_right + ehidden_statestra_padding), mode=self.pad_mode
            )

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()

        self.norm_type = norm
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        if norm == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        self.causal = causal
        self.trim_right_ratio = trim_right_ratio

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2

        padding_left = padding_total - padding_right

        # unpad
        end = hidden_states.shape[-1] - padding_right
        hidden_states = hidden_states[..., padding_left:end]
        return hidden_states


class EncodecLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data. Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(2, 0, 1)
        hidden_states = self.lstm(hidden_states)[0] + hidden_states
        hidden_states = hidden_states.permute(1, 2, 0)
        return hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.

    Args:
        config:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
    """

    def __init__(self, config: EncodecConfig, dim: int, kernel_sizes: List[int], dilations: List[int]):
        super().__init__()

        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                nn.ELU(),
                EncodecConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=config.norm,
                    causal=config.causal,
                    pad_mode=config.pad_mode,
                ),
            ]
        self.block = nn.ModuleList(block)

        self.shortcut = EncodecConv1d(
            dim,
            dim,
            kernel_size=1,
            norm=config.norm,
            causal=config.causal,
            pad_mode=config.pad_mode,
        )

    def forward(self, hidden_states):
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        mult = 1
        model = [
            EncodecConv1d(
                config.audio_channels,
                mult * config.num_filters,
                config.kernel_size,
                norm=config.norm,
                causal=config.causal,
                pad_mode=config.pad_mode,
            )
        ]

        # Downsample to raw audio scale
        for ratio in reversed(config.ratios):
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config,
                        mult * config.num_filters,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base**j, 1],
                    )
                ]
            # Add downsampling layers
            model += [
                nn.ELU(),
                EncodecConv1d(
                    mult * config.num_filters,
                    mult * config.num_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=config.norm,
                    causal=config.causal,
                    pad_mode=config.pad_mode,
                ),
            ]
            mult *= 2

        model += [EncodecLSTM(mult * config.num_filters, config.num_lstm_layers)]

        model += [
            nn.ELU(),
            EncodecConv1d(
                mult * config.num_filters,
                config.dimension,
                config.last_kernel_size,
                norm=config.norm,
                causal=config.causal,
                pad_mode=config.pad_mode,
            ),
        ]

        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        mult = int(2 ** len(config.ratios))
        model = [
            EncodecConv1d(
                config.dimension,
                mult * config.num_filters,
                config.kernel_size,
                norm=config.norm,
                causal=config.causal,
                pad_mode=config.pad_mode,
            )
        ]

        model += [EncodecLSTM(mult * config.num_filters, config.num_lstm_layers)]

        # Upsample to raw audio scale
        for ratio in config.ratios:
            # Add upsampling layers
            model += [
                nn.ELU(),
                EncodecConvTranspose1d(
                    mult * config.num_filters,
                    mult * config.num_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=config.norm,
                    causal=config.causal,
                    trim_right_ratio=config.trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config,
                        mult * config.num_filters // 2,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base**j, 1],
                    )
                ]
            mult //= 2

        # Add final layers
        model += [
            nn.ELU(),
            EncodecConv1d(
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
                norm=config.norm,
                causal=config.causal,
                pad_mode=config.pad_mode,
            ),
        ]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: EncodecConfig, dim: int):
        super().__init__()
        embed = torch.zeros(config.bins, dim)

        self.codebook_size = config.bins

        self.register_buffer("inited", torch.Tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(config.bins))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def quantize(self, hidden_states):
        embed = self.embed.t()
        dist = -(
            hidden_states.pow(2).sum(1, keepdim=True) - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize


class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()

        codebook_dim = config.codebook_dim if config.codebook_dim is not None else config.dimension

        self.requires_projection = codebook_dim != config.dimension
        self.codebook = EncodecEuclideanCodebook(config, codebook_dim)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class EncodecResidualVectorQuantization(nn.Module):
    """
    Residual vector quantization implementation. Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, config: EncodecConfig, num_quantizers: int):
        super().__init__()
        self.layers = nn.ModuleList([EncodecVectorQuantization(config) for _ in range(num_quantizers)])

    def encode(self, hidden_states: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        residual = hidden_states
        all_indices = []
        num_quantizers = num_quantizers or len(self.layers)
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: EncodecConfig, num_quantizers: int):
        super().__init__()
        self.config = config
        self.num_quantizers = num_quantizers
        self.vector_quantization = EncodecResidualVectorQuantization(config, num_quantizers)

    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: Optional[float] = None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = math.log2(self.config.bins) * frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: torch.Tensor, frame_rate: int, bandwidth: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vector_quantization.encode(embeddings, num_quantizers=num_quantizers)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        return self.vector_quantization.decode(codes)


class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EncodecConfig
    base_model_prefix = "encodec"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncodecEncoder, EncodecDecoder)):
            module.gradient_checkpointing = value


# TODO
ENCODEC_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`EncodecEncoderWithSpeechPrenet`] or [`EncodecEncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`EncodecEncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`EncodecDecoderWithSpeechPrenet`] or [`EncodecDecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`EncodecDecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.
"""


# TODO
ENCODEC_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# TODO
ENCODEC_INPUTS_DOCSTRING = r"""
    Args:
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.

            </Tip>

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read [`EncodecDecoder._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

        head_mask (`torch.FloatTensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_values` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_values` of shape `(batch_size, sequence_length)`. decoder_inputs_embeds (`torch.FloatTensor`
            of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
            `decoder_input_values` you can choose to directly pass an embedded representation. If `past_key_values` is
            used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is
            useful if you want more control over how to convert `decoder_input_values` indices into associated vectors
            than the model's internal embedding lookup matrix.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The EnCodec neural audio codec model.",
    ENCODEC_BASE_START_DOCSTRING,
)
class EncodecModel(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        self.config = config

        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        hop_length = np.prod(config.ratios)
        self.frame_rate = math.ceil(config.sampling_rate / hop_length)

        num_quantizers = int(1000 * config.target_bandwidths[-1] // (self.frame_rate * 10))
        self.quantizer = EncodecResidualVectorQuantizer(config, num_quantizers)

        self.bits_per_codebook = int(math.log2(self.config.bins))
        if 2**self.bits_per_codebook != self.config.bins:
            raise ValueError("Number of quantizer bins must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], EncodecEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[B, K, T]`, with `K` the number of codebooks, `T` frames.
        """
        return_dict = return_dict or self.config.return_dict

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunched decoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - stride + 1, stride):
            frame = (
                input_values[:, :, offset : offset + chunk_length]
                * padding_mask[..., offset : offset + chunk_length].bool()
            )
            encoded_frame, scale = self._encode_frame(frame, bandwidth, None)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = torch.stack(encoded_frames)

        if return_dict:
            return EncodecEncoderOutput(encoded_frames, scales)

        return (encoded_frames, scales)

    def _encode_frame(
        self, input_values: torch.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate

        if self.config.chunk_in_sec is not None and duration > 1e-5 + self.config.chunk_in_sec:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_in_sec}")

        scale = None
        if self.config.normalize:
            # input_values = input_values * padding_mask
            mono = input_values.mean(dim=1, keepdim=True)
            scale = mono.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, self.frame_rate, bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    @staticmethod
    def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
        # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
        # e.g., more than 2 frames per position.
        # The core idea is to use a weight function that is a triangle,
        # with a maximum value at the middle of the chunk.
        # We use this weighting when summing the frames, and divide by the sum of weights
        # for each positions at the end. Thus:
        #   - if a frame is the only one to cover a position, the weighting is a no-op.
        #   - if 2 frames cover a position:
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       , i.e. S offset of second frame starts, T end of first frame.
        # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
        # After the final normalization, the weight of the second frame at position `t` is
        # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
        #
        #   - if more than 2 frames overlap at a given point, we hope that by induction
        #      something sensible happens.
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

        frame_length = frames[0].shape[-1]
        t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (t - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0

        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        if sum_weight.min() == 0:
            raise ValueError(f"`sum_weight` minimum element must be bigger than zero: {sum_weight}`")

        return out / sum_weight

    def decode(
        self,
        audio_codes,
        audio_scales,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.
        """
        return_dict = return_dict or self.config.return_dict

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []

            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if return_dict:
            return EncodecDecoderOutput(audio_values)
        return (audio_values,)

    def _decode_frame(
        self,
        codes: Union[List[Tuple[torch.Tensor, Optional[torch.Tensor]]], EncodecEncoderOutput],
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = codes.transpose(0, 1)
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    # TODO @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    # TODO @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecOutput]:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.

        bandwidth (`float`, *optional*):
            The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
            bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            bandwidth == 6.0

        Returns:
        """
        return_dict = return_dict or self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, return_dict).to_tuple()
        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]

        if return_dict:
            return EncodecOutput(audio_codes, audio_values)

        return (audio_codes, audio_values)
