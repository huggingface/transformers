# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Transformers Xcodec model."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import ModelOutput, auto_docstring
from ..auto import AutoModel
from .configuration_xcodec import XcodecConfig


@dataclass
class XcodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discrete code indices computed using `model.encode`.
        audio_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`, *optional*)
            Decoded audio values obtained using the decoder part of Xcodec.
    """

    audio_codes: Optional[torch.LongTensor] = None
    audio_values: Optional[torch.FloatTensor] = None


@dataclass
class XcodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discrete code indices computed using `model.encode`.
    """

    audio_codes: Optional[torch.LongTensor] = None


@dataclass
class XcodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, channels, num_samples)`, *optional*):
            Decoded audio values obtained using the decoder part of Xcodec.
    """

    audio_values: Optional[torch.FloatTensor] = None


class ResidualUnit(nn.Module):
    """Residual block for SemanticEncoder and SemanticDecoder used in Xcodec."""

    def __init__(self, config: XcodecConfig, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.activation = nn.ELU()
        padding = ((config.unit_kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            config.unit_kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output_tensor = self.activation(hidden_state)
        output_tensor = self.conv1(output_tensor)
        output_tensor = self.activation(output_tensor)
        output_tensor = self.conv2(output_tensor)
        return hidden_state + output_tensor


class SemanticEncoderBlock(nn.Module):
    def __init__(self, config: XcodecConfig, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.res_units = nn.ModuleList(
            [ResidualUnit(config, in_channels, in_channels, dilation) for dilation in config.block_dilations]
        )

        # special case: stride=1, do not use kernel=2
        kernel = 3 if stride == 1 else (2 * stride)
        padding = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for unit in self.res_units:
            hidden_state = unit(hidden_state)
        hidden_state = self.conv(hidden_state)
        return hidden_state


class SemanticEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if len(config.strides) != len(config.channel_ratios):
            raise ValueError("Number of strides must match the number of channel_ratios.")
        self.conv = nn.Conv1d(
            config.semantic_hidden_size,
            config.semantic_hidden_size,
            config.kernel_size,
            1,
            config.kernel_size // 2,
            bias=False,
        )

        in_channels = config.semantic_hidden_size
        conv_blocks = []
        for i, stride in enumerate(config.strides):
            out_channels = int(config.semantic_hidden_size * config.channel_ratios[i])
            conv_blocks += [SemanticEncoderBlock(config, in_channels, out_channels, stride)]
            in_channels = out_channels

        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv(hidden_state)
        for block in self.conv_blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class SemanticDecoderBlock(nn.Module):
    def __init__(self, config: XcodecConfig, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if stride == 1:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        else:
            kernel_size = 2 * stride
            padding = (stride + 1) // 2
            output_padding = 1 if stride % 2 == 1 else 0
            self.conv = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False
            )

        self.res_units = nn.ModuleList(
            [ResidualUnit(config, out_channels, out_channels, dilation) for dilation in config.block_dilations]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv(hidden_state)
        for unit in self.res_units:
            hidden_state = unit(hidden_state)
        return hidden_state


class SemanticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=config.semantic_hidden_size,
            out_channels=int(config.semantic_hidden_size * config.channel_ratios[0]),
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=False,
        )
        conv_blocks = []
        for i, stride in enumerate(config.strides):
            in_channels = int(config.semantic_hidden_size * config.channel_ratios[i])

            if i < (len(config.channel_ratios) - 1):
                out_channels = int(config.semantic_hidden_size * config.channel_ratios[i + 1])
            else:
                out_channels = config.semantic_hidden_size

            conv_blocks += [SemanticDecoderBlock(config, in_channels, out_channels, stride)]

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.conv2 = nn.Conv1d(
            config.semantic_hidden_size,
            config.semantic_hidden_size,
            config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=False,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for block in self.conv_blocks:
            hidden_state = block(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class XcodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)
        self.codebook_size = config.codebook_size
        self.register_buffer("inited", torch.Tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(config.codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.quantize
    def quantize(self, hidden_states):
        embed = self.embed.t()
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        quantized = F.embedding(embed_ind, self.embed)
        return quantized


class XcodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: XcodecConfig):
        super().__init__()
        self.codebook = XcodecEuclideanCodebook(config)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization.encode
    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    # Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization.decode
    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class XcodecResidualVectorQuantization(nn.Module):
    """
    Residual vector quantization implementation. Follows Algorithm 1 in https://huggingface.co/papers/2107.03312
    """

    def __init__(self, config: XcodecConfig):
        super().__init__()
        self.quantizers = nn.ModuleList([XcodecVectorQuantization(config) for _ in range(config.num_quantizers)])
        self.frame_rate = config.frame_rate
        self.codebook_size = config.codebook_size
        self.num_quantizers = config.num_quantizers

    def get_bandwidth_per_quantizer(self):
        """Return bandwidth per quantizer."""
        return math.log2(self.codebook_size) * self.frame_rate / 1000

    def get_num_quantizers_for_bandwidth(self, bandwidth=None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = self.get_bandwidth_per_quantizer()
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: torch.Tensor, bandwidth=None) -> torch.Tensor:
        """
        Encode the input tensor into discrete indices using RVQ, with the number of quantizers selected based on the given bandwidth.
        Each quantizer /codebook residually quantizes the input and returns the nearest indices in terms of Euclidian distance.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for quantizer in self.quantizers[:num_quantizers]:
            indices = quantizer.encode(residual)
            quantized = quantizer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to their quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            quantizer = self.quantizers[i]
            quantized = quantizer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


@auto_docstring
class XcodecPreTrainedModel(PreTrainedAudioTokenizerBase):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XcodecConfig
    base_model_prefix = "xcodec"
    main_input_name = "input_values"

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

    def apply_weight_norm(self):
        """Apply weight norm in the acoustic encoder and decoder because the original checkpoint has weight norm applied."""
        weight_norm = torch.nn.utils.weight_norm
        if hasattr(torch.nn.utils.parametrizations, "weight_norm"):
            weight_norm = torch.nn.utils.parametrizations.weight_norm

        weight_norm(self.acoustic_encoder.conv1)
        weight_norm(self.acoustic_encoder.conv2)

        for block in self.acoustic_encoder.block:
            weight_norm(block.conv1)
            for res_unit in (block.res_unit1, block.res_unit2, block.res_unit3):
                weight_norm(res_unit.conv1)
                weight_norm(res_unit.conv2)

        weight_norm(self.acoustic_decoder.conv1, name="weight")
        weight_norm(self.acoustic_decoder.conv2, name="weight")

        for block in self.acoustic_decoder.block:
            weight_norm(block.conv_t1, name="weight")
            for res_unit in (block.res_unit1, block.res_unit2, block.res_unit3):
                weight_norm(res_unit.conv1, name="weight")
                weight_norm(res_unit.conv2, name="weight")

    def remove_weight_norm(self):
        """Remove the weight norm from the acoustic encoder and decoder."""
        for module in (self.acoustic_encoder, self.acoustic_decoder):
            for m in module.modules():
                try:
                    torch.nn.utils.remove_weight_norm(m, name="weight")
                except (ValueError, AttributeError):
                    pass
                if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
                    torch.nn.utils.parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


@auto_docstring(custom_intro="""The Xcodec neural audio codec model.""")
class XcodecModel(XcodecPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pad = config.hop_length // 2
        acoustic_model = AutoModel.from_config(config.acoustic_model_config)
        self.acoustic_encoder = acoustic_model.encoder
        self.acoustic_decoder = acoustic_model.decoder
        self._adjust_dac_decoder(self.acoustic_decoder)
        self.encoder_semantic = SemanticEncoder(config)
        self.decoder_semantic = SemanticDecoder(config)
        self.semantic_model = AutoModel.from_config(config.semantic_model_config).eval()
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.semantic_model_config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.acoustic_model_config.hidden_size)
        self.quantizer = XcodecResidualVectorQuantization(config)

    @staticmethod
    def _adjust_dac_decoder(decoder: nn.Module):
        r"""
        DAC implemented in Xcodec is slightly different from the HF version.
        DAC in Xcodec adjusts the output padding in every ConvTranspose1d in the decoder and removes
        the final `nn.Tanh` activation function.
        """
        for module in decoder.modules():
            if isinstance(module, nn.ConvTranspose1d):
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                module.output_padding = (stride % 2,)
        if hasattr(decoder, "tanh") and isinstance(decoder.tanh, nn.Tanh):
            decoder.tanh = nn.Identity()

    def _extract_semantic_features(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        input_values = input_values[:, 0, :]
        input_values = F.pad(input_values, (self.pad, self.pad))
        with torch.no_grad():
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        stacked = torch.stack(hidden_states, dim=1)
        return stacked.mean(dim=1)

    @auto_docstring
    def encode(
        self,
        input_values: torch.Tensor,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, XcodecEncoderOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            Float values of the input audio waveform.
        bandwidth (`float`, *optional*):
            The target bandwidth in (kbps) supports only values in `config.target_bandwidths`.
            Defaults to the highest available bandwidth `4.0` kbps.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`].

        Returns:
            `torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)` containing the discrete encoded audio codes.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        channels = input_values.shape[1]
        if channels != 1:
            raise ValueError(f"Audio must be mono, but got {channels}")

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[-1]
        elif bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. Select one of {self.config.target_bandwidths}."
            )

        e_semantic_input = self._extract_semantic_features(input_values).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.acoustic_encoder(input_values)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            # make sure they line up if frames don't match
            e_acoustic = self.acoustic_encoder(F.pad(input_values[:, 0, :], (self.pad, self.pad)).unsqueeze(1))

        embeddings = torch.cat([e_acoustic, e_semantic], dim=1)
        embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        audio_codes = self.quantizer.encode(embeddings, bandwidth)
        audio_codes = audio_codes.transpose(0, 1)

        if not return_dict:
            return audio_codes

        return XcodecEncoderOutput(audio_codes)

    @auto_docstring
    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, XcodecDecoderOutput]:
        r"""
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`):
            Discrete code indices computed using `model.encode`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`]

        Returns:
            Decoded audio values of shape `(batch_size, channels, num_samples)` obtained using the decoder part of
            Xcodec.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_codes = audio_codes.transpose(0, 1)
        quantized = self.quantizer.decode(audio_codes)
        quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        audio_values = self.acoustic_decoder(quantized_acoustic)

        if not return_dict:
            return audio_values

        return XcodecDecoderOutput(audio_values)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        audio_codes: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], XcodecOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            The raw float values of the input audio waveform.
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`:
            Discrete code indices computed using `model.encode`.
        bandwidth (`float`, *optional*):
            Target bandwidth in kbps. Must be one of `config.target_bandwidths`. Defaults to the highest available bandwidth.
        bandwidth (`float`, *optional*):
            Target bandwidth in kbps. Must be one of `config.target_bandwidths`. Defaults to the highest available bandwidth.
        return_dict (`bool`, *optional*):
            Whether to return a [`XcodecOutput`] instead of a plain tuple.

        Returns:
            `XcodecOutput` or tuple `(audio_codes, audio_values)`:
            - `audio_codes` of shape `(batch_size, num_quantizers, codes_length)`: the quantized discrete codes.
            - `audio_values` of shape `(batch_size, channels, num_samples)`: the reconstructed audio waveform given the codes.

        Example:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, XcodecModel

        >>> model_id = "hf-audio/xcodec-hubert-librispeech"
        >>> model = XcodecModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        >>> audio_sample = dataset[0]['audio']['array']

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        length = input_values.shape[-1]

        if audio_codes is None:
            audio_codes = self.encode(input_values, bandwidth, return_dict=False)

        audio_values = self.decode(audio_codes, return_dict=return_dict)[0][..., :length]

        if not return_dict:
            return (audio_codes, audio_values)

        return XcodecOutput(audio_codes=audio_codes, audio_values=audio_values)


__all__ = ["XcodecModel", "XcodecPreTrainedModel"]
