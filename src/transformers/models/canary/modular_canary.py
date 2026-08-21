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
"""PyTorch Canary model."""

import math

import numpy as np
import torch
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..cohere_asr.modeling_cohere_asr import (
    CohereAsrDecoder,
    CohereAsrForConditionalGeneration,
    CohereAsrModel,
    CohereAsrPreTrainedModel,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import SinusoidsPositionEmbedding
from .configuration_canary import CanaryConfig, CanaryDecoderConfig


logger = logging.get_logger(__name__)


class CanaryPositionalEmbedding(SinusoidsPositionEmbedding):
    """
    Identical to [`SinusoidsPositionEmbedding`] except that the timescales and the `1 / sqrt(channels)` scaling match
    NeMo's `FixedPositionalEncoding`, and it is indexed by `position_ids`.
    """

    def __init__(self, length: int, channels: int):
        max_timescale = 10000 ** ((channels - 2) / channels)
        super().__init__(length, channels, max_timescale)

    def compute_default_singular_positional_embedding(self) -> torch.Tensor:
        log_timescale_increment = np.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.channels // 2).float())
        scaled_time = torch.arange(self.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1) / math.sqrt(self.channels)
        return emb.to(torch.get_default_dtype())

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.positional_embedding[position_ids]


@auto_docstring
class CanaryPreTrainedModel(CohereAsrPreTrainedModel):
    config: CanaryConfig
    _no_split_modules = ["ParakeetEncoderBlock", "CanaryDecoderLayer"]

    def _get_feat_extract_output_lengths(self):
        raise AttributeError("Not needed for Canary")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, CanaryPositionalEmbedding):
            init.copy_(module.positional_embedding, module.compute_default_singular_positional_embedding())


class CanaryDecoder(CohereAsrDecoder):
    config: CanaryDecoderConfig

    def __init__(self, config: CanaryDecoderConfig):
        super().__init__(config)
        self.pos_emb = CanaryPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.proj = nn.Identity()


@auto_docstring(
    custom_intro="""
    The bare Canary model (FastConformer encoder + Transformer decoder) outputting raw hidden-states without any
    specific head on top.
    """
)
class CanaryModel(CohereAsrModel):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.decoder = CanaryDecoder(config.decoder_config)


@auto_docstring(
    custom_intro="""
    The Canary model with a language modeling head. Can be used for multilingual automatic speech recognition and
    speech-to-text translation.
    """
)
class CanaryForConditionalGeneration(CohereAsrForConditionalGeneration):
    def __init__(self, config: CanaryConfig):
        super().__init__(config)
        self.proj_out = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=True)


__all__ = ["CanaryForConditionalGeneration", "CanaryModel", "CanaryPreTrainedModel"]
