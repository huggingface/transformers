# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""HiggsAudioTokenizer model."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ...utils import ModelOutput, auto_docstring
from ..auto.modeling_auto import AutoModel
from ..dac.modeling_dac import DacDecoder, DacEncoder
from ..xcodec.modeling_xcodec import (
    SemanticDecoder,
    SemanticEncoder,
    XcodecEuclideanCodebook,
    XcodecPreTrainedModel,
)
from .configuration_higgs_audio_tokenizer import HiggsAudioTokenizerConfig


@dataclass
@auto_docstring
class HiggsAudioTokenizerOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, time_steps)`):
        Codebook indices for each codebook (quantized discrete representation of input).
    audio_values (`torch.Tensor` of shape `(batch_size, input_length)`):
        Reconstructed audio data.
    """

    audio_codes: Optional[torch.LongTensor] = None
    audio_values: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring
class HiggsAudioTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`, *optional*):
        Codebook indices for each codebook (quantized discrete representation of input).
    """

    audio_codes: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring
class HiggsAudioTokenizerDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, input_length)`, *optional*):
        Decoded audio values, obtained using the decoder part of HiggsAudioTokenizer.
    """

    audio_values: Optional[torch.FloatTensor] = None


class HiggsAudioTokenizerEncoder(DacEncoder):
    pass


class HiggsAudioTokenizerDecoder(DacDecoder):
    pass


class HiggsAudioTokenizerSemanticEncoder(SemanticEncoder):
    pass


class HiggsAudioTokenizerSemanticDecoder(SemanticDecoder):
    pass


class HiggsAudioTokenizerEuclideanCodebook(XcodecEuclideanCodebook):
    pass


class HiggsAudioTokenizerVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__()
        self.codebook = HiggsAudioTokenizerEuclideanCodebook(config)
        codebook_dim = config.codebook_dim
        dim = config.hidden_size
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.project_in(hidden_states)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class HiggsAudioTokenizerResidualVectorQuantization(nn.Module):
    """
    Residual vector quantization implementation. Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [HiggsAudioTokenizerVectorQuantization(config) for _ in range(config.num_quantizers)]
        )
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
class HiggsAudioTokenizerPreTrainedModel(XcodecPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HiggsAudioTokenizerConfig
    base_model_prefix = "higgs_audio_tokenizer"


@auto_docstring(custom_intro="""The Higgs Audio neural audio codec model.""")
class HiggsAudioTokenizer(HiggsAudioTokenizerPreTrainedModel):
    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__(config)
        self.config = config
        self.pad = config.pad
        self.acoustic_encoder = HiggsAudioTokenizerEncoder(config.acoustic_model_config)
        self.acoustic_decoder = HiggsAudioTokenizerDecoder(config.acoustic_model_config)
        self._adjust_dac_decoder(self.acoustic_decoder)
        self.encoder_semantic = HiggsAudioTokenizerSemanticEncoder(config)
        self.decoder_semantic = HiggsAudioTokenizerSemanticDecoder(config)
        self.semantic_model = AutoModel.from_config(config.semantic_model_config)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.semantic_model_config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.acoustic_model_config.hidden_size)
        self.quantizer = HiggsAudioTokenizerResidualVectorQuantization(config)

        self.downsample_mode = config.downsample_mode
        self.semantic_downsample_factor = int(
            config.hop_length / (config.sample_rate / config.semantic_sample_rate) / config.downsample_factor
        )
        self.sampling_rate = config.sample_rate
        self.semantic_sample_rate = config.semantic_sample_rate
        if self.downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=config.semantic_downsample_factor, stride=config.semantic_downsample_factor
            )

    @staticmethod
    def _adjust_dac_decoder(decoder: nn.Module):
        r"""
        DAC implemented in HiggsAudioTokenizer is slightly different from the HF version.
        DAC in HiggsAudioTokenizer adjusts the output padding in every ConvTranspose1d in the decoder and removes
        the final `nn.Tanh` activation function.
        """
        for module in decoder.modules():
            if isinstance(module, nn.ConvTranspose1d):
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                module.output_padding = (stride % 2,)
        if hasattr(decoder, "tanh") and isinstance(decoder.tanh, nn.Tanh):
            decoder.tanh = nn.Identity()

    def _extract_semantic_features(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        input_values = torchaudio.functional.resample(input_values, self.sampling_rate, self.semantic_sample_rate)
        input_values = input_values[:, 0, :]
        input_values = F.pad(input_values, (self.pad, self.pad))
        with torch.no_grad():
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        stacked = torch.stack(hidden_states, dim=1)
        semantic_features = stacked.mean(dim=1)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                semantic_features = semantic_features[:, :: self.semantic_downsample_factor, :]
        elif self.downsample_mode == "avg":
            semantic_features = self.semantic_pooling(semantic_features.transpose(1, 2)).transpose(1, 2)

        return semantic_features

    @auto_docstring
    def encode(
        self,
        input_values: torch.Tensor,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, HiggsAudioTokenizerEncoderOutput]:
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

        if input_values.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch_size, channels, num_samples), but got shape {input_values.shape}"
            )

        _, channels, _ = input_values.shape

        if channels not in (1, 2):
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

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

        return HiggsAudioTokenizerEncoderOutput(audio_codes)

    @auto_docstring
    def decode(
        self, audio_codes: torch.Tensor, return_dict: Optional[bool] = None, **kwargs
    ) -> Union[torch.Tensor, HiggsAudioTokenizerDecoderOutput]:
        r"""
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`):
            Discrete code indices computed using `model.encode`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`]

        Returns:
            Decoded audio values of shape `(batch_size, channels, num_samples)` obtained using the decoder part of HiggsAudioTokenizer.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_codes = audio_codes.transpose(0, 1)
        quantized = self.quantizer.decode(audio_codes)
        quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        audio_values = self.acoustic_decoder(quantized_acoustic)

        if not return_dict:
            return audio_values

        return HiggsAudioTokenizerDecoderOutput(audio_values)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        audio_codes: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], HiggsAudioTokenizerOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            The raw float values of the input audio waveform.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discrete code indices computed using `model.encode`.
        bandwidth (`float`, *optional*):
            Target bandwidth in kbps. Must be one of `config.target_bandwidths`.
            Defaults to the highest available bandwidth.
        return_dict (`bool`, *optional*):
            Whether to return a [`HiggsAudioTokenizerOutput`] instead of a plain tuple.

        Returns:
            `HiggsAudioTokenizerOutput` or tuple `(audio_codes, audio_values)`:
            - `audio_codes` of shape `(batch_size, num_quantizers, codes_length)`: the quantized discrete codes.
            - `audio_values` of shape `(batch_size, channels, num_samples)`: the reconstructed audio waveform given the codes.

        Example:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, HiggsAudioTokenizer

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "bosonai/higgs-audio-v2-tokenizer"
        >>> model = HiggsAudioTokenizer.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if audio_codes is None:
            audio_codes = self.encode(input_values, bandwidth, return_dict=False)

        audio_values = self.decode(audio_codes, return_dict=return_dict)[0]

        if not return_dict:
            return (audio_codes, audio_values)

        return HiggsAudioTokenizerOutput(audio_codes=audio_codes, audio_values=audio_values)


__all__ = ["HiggsAudioTokenizer", "HiggsAudioTokenizerPreTrainedModel"]
