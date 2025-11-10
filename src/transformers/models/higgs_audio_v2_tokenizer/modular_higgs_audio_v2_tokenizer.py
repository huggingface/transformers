# coding=utf-8
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
from ..xcodec.modeling_xcodec import XcodecEuclideanCodebook, XcodecModel


class HiggsAudioV2TokenizerConfig(XcodecConfig):
    def __init__(
        self,
        target_bandwidths=None,
        sample_rate=24000,
        kernel_size=3,
        channel_ratios=[1, 1],
        strides=[1, 1],
        block_dilations=[1, 1],
        unit_kernel_size=3,
        codebook_size=1024,
        codebook_dim=None,
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


class HiggsAudioV2TokenizerEuclideanCodebook(XcodecEuclideanCodebook): ...


class HiggsAudioV2TokenizerVectorQuantization(nn.Module):
    def __init__(self, config: HiggsAudioV2TokenizerConfig):
        super().__init__()
        self.codebook = HiggsAudioV2TokenizerEuclideanCodebook(config)
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

        stacked = torch.stack(hidden_states, dim=1)
        semantic_features = stacked.mean(dim=1)

        semantic_downsample_factor = int(
            self.config.hop_length
            / (self.config.sample_rate / self.config.semantic_sample_rate)
            / self.config.downsample_factor
        )
        if semantic_downsample_factor > 1:
            semantic_features = semantic_features[:, ::semantic_downsample_factor, :]

        return semantic_features


__all__ = ["HiggsAudioV2TokenizerConfig", "HiggsAudioV2TokenizerModel"]
