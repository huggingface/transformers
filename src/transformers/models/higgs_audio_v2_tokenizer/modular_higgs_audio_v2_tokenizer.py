# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..xcodec.configuration_xcodec import XcodecConfig
from ..xcodec.modeling_xcodec import XcodecEuclideanCodebook, XcodecModel, XcodecPreTrainedModel


class HiggsAudioV2TokenizerConfig(XcodecConfig):
    r"""
    This is the configuration class to store the configuration of an [`HiggsAudioV2TokenizerModel`]. It is used to instantiate a
    HiggsAudioV2Tokenizer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [`Higgs Audio v2 Tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer).
    e.g. [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            target_bandwidths (`List[float]`, *optional*, defaults to `[0.5, 1, 1.5, 2]`):
                The range of different bandwidths (in kbps) the model can encode audio with.
            sample_rate (`int`, *optional*, defaults to 24000):
                The sampling rate at which the audio waveform should be digitalized, in hertz (Hz).
            kernel_size (`int`, *optional*, defaults to 3):
                Kernel size for the initial semantic convolution.
            channel_ratios (`List[float]`, *optional*, defaults to `[1, 1]`):
                Expansion factors for the number of output channels in each semantic block.
            strides (`List[int]`, *optional*, defaults to `[1, 1]`):
                Strides for each semantic encoder block.
            block_dilations (`List[int]`, *optional*, defaults to `[1, 1]`):
                Dilation factors for the residual units in semantic blocks.
            unit_kernel_size (`int`, *optional*, defaults to 3):
                Kernel size inside each ResidualUnit in semantic blocks.
            codebook_size (`int`, *optional*, defaults to 1024):
                Number of entries in each residual quantizer's codebook.
            codebook_dim (`int`, *optional*, defaults to 64):
                Dimensionality of each codebook vector.
            initializer_range (`float`, *optional*, defaults to 0.02):
                Standard deviation of the truncated normal initializer for all weight matrices.
            acoustic_model_config (`Union[Dict, AutoConfig]`, *optional*):
                An instance of the configuration for the acoustic (DAC) model.
            semantic_model_config (`Union[Dict, AutoConfig]`, *optional*):
                An instance of the configuration object for the semantic (HuBERT) model.
            semantic_sample_rate (`int`, *optional*, defaults to 16000):
                The sampling rate at which the semantic model expects audio input, in hertz (Hz).
            downsample_factor (`int`, *optional*, defaults to 320):
                Downsampling factor for the semantic features.

    Example:

    ```python
    >>> from transformers import HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerConfig

    >>> # Initializing configuration
    >>> configuration = HiggsAudioV2TokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = HiggsAudioV2TokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    _default_semantic_model_config_kwargs = {
        "mask_time_prob": 0.0,
    }

    def __init__(
        self,
        target_bandwidths=[0.5, 1, 1.5, 2],
        sample_rate=24000,
        kernel_size=3,
        channel_ratios=[1, 1],
        strides=[1, 1],
        block_dilations=[1, 1],
        unit_kernel_size=3,
        codebook_size=1024,
        codebook_dim=64,
        initializer_range=0.02,
        acoustic_model_config=None,
        semantic_model_config=None,
        semantic_sample_rate=16000,
        downsample_factor=320,
        **kwargs,
    ):
        super().__init__(
            target_bandwidths=target_bandwidths,
            sample_rate=sample_rate,
            kernel_size=kernel_size,
            channel_ratios=channel_ratios,
            strides=strides,
            block_dilations=block_dilations,
            unit_kernel_size=unit_kernel_size,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            initializer_range=initializer_range,
            acoustic_model_config=acoustic_model_config,
            semantic_model_config=semantic_model_config,
            **kwargs,
        )

        self.semantic_sample_rate = semantic_sample_rate
        self.downsample_factor = downsample_factor

    @property
    def semantic_downsample_factor(self):
        return int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / self.downsample_factor)


class HiggsAudioV2TokenizerPreTrainedModel(XcodecPreTrainedModel):
    _no_split_modules = ["HiggsAudioV2TokenizerResidualVectorQuantization", "DacResidualUnit"]
    _keys_to_ignore_on_load_unexpected = ["semantic_model.masked_spec_embed"]


class HiggsAudioV2TokenizerEuclideanCodebook(XcodecEuclideanCodebook): ...


class HiggsAudioV2TokenizerVectorQuantization(nn.Module):
    def __init__(self, config: HiggsAudioV2TokenizerConfig):
        super().__init__()
        self.codebook = HiggsAudioV2TokenizerEuclideanCodebook(config)
        self.project_in = nn.Linear(config.hidden_size, config.codebook_dim)
        self.project_out = nn.Linear(config.codebook_dim, config.hidden_size)

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


class HiggsAudioV2TokenizerModel(XcodecModel):
    def _extract_semantic_features(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.sample_rate != self.config.semantic_sample_rate:
            input_values = torchaudio.functional.resample(
                input_values, self.config.sample_rate, self.config.semantic_sample_rate
            )

        input_values = input_values[:, 0, :]
        # TODO: there is a diff here with original codebase https://github.com/boson-ai/higgs-audio/blob/f644b62b855ba2b938896436221e01efadcc76ca/boson_multimodal/audio_processing/higgs_audio_v2_tokenizer.py#L173-L174
        # input_values = F.pad(input_values, (self.pad, self.pad))
        input_values = F.pad(input_values, (160, 160))
        with torch.no_grad():
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        stacked = torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)
        semantic_features = stacked.mean(dim=1)

        if self.config.semantic_downsample_factor > 1:
            semantic_features = semantic_features[:, :: self.config.semantic_downsample_factor, :]

        return semantic_features


__all__ = ["HiggsAudioV2TokenizerConfig", "HiggsAudioV2TokenizerPreTrainedModel", "HiggsAudioV2TokenizerModel"]
