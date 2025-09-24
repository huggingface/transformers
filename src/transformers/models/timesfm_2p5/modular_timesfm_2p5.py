# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from ..timesfm.configuration_timesfm import TimesFmConfig
from ..timesfm.modeling_timesfm import (
    TimesFmAttention,
    TimesFmDecoderLayer,
    TimesFmMLP,
    TimesFmModel,
    TimesFmModelForPrediction,
    TimesFmOutput,
    TimesFmOutputForPrediction,
    TimesFmPositionalEmbedding,
    TimesFmPreTrainedModel,
    TimesFmResidualBlock,
    TimesFmRMSNorm,
)


class Timesfm2P5Config(TimesFmConfig):
    pass


class Timesfm2P5Output(TimesFmOutput):
    pass


class Timesfm2P5OutputForPrediction(TimesFmOutputForPrediction):
    pass


class Timesfm2P5MLP(TimesFmMLP):
    pass


class Timesfm2P5ResidualBlock(TimesFmResidualBlock):
    pass


class Timesfm2P5RMSNorm(TimesFmRMSNorm):
    pass


class Timesfm2P5PositionalEmbedding(TimesFmPositionalEmbedding):
    pass


class Timesfm2P5Attention(TimesFmAttention):
    pass


class Timesfm2P5DecoderLayer(TimesFmDecoderLayer):
    pass


class Timesfm2P5PreTrainedModel(TimesFmPreTrainedModel):
    pass


class Timesfm2P5Model(TimesFmModel):
    pass


class Timesfm2P5ModelForPrediction(TimesFmModelForPrediction):
    pass


__all__ = [
    "Timesfm2P5Config",
    "Timesfm2P5ModelForPrediction",
    "Timesfm2P5PreTrainedModel",
    "Timesfm2P5Model",
]
