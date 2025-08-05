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

from ..t5.configuration_t5 import T5Config, T5OnnxConfig
from ..t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5ClassificationHead,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5Model,
    T5PreTrainedModel,
    T5Stack,
)
from ...modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg


class T5LaConfig(T5Config):
    pass


class T5LaOnnxConfig(T5OnnxConfig):
    pass


class T5LaLayerNorm(T5LayerNorm):
    pass


class T5LaDenseActDense(T5DenseActDense):
    pass


class T5LaDenseGatedActDense(T5DenseGatedActDense):
    pass


class T5LaLayerFF(T5LayerFF):
    pass


class T5LaAttention(T5Attention):
    pass


class T5LaLayerSelfAttention(T5LayerSelfAttention):
    pass


class T5LaLayerCrossAttention(T5LayerCrossAttention):
    pass


class T5LaBlock(T5Block):
    pass


class T5LaClassificationHead(T5ClassificationHead):
    pass


class T5LaPreTrainedModel(T5PreTrainedModel):
    pass


class T5LaStack(T5Stack):
    pass


class LookAheadHeads(nn.Module):
    def __init__(self, config: T5LaConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                # K heads for LA positions:
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.lookahead_size)
            ]
        )

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        # Apply each head to the shared features
        logits = [head(x) for head in self.heads]

        # Stack logits along a new dimension to create a tensor of shape [batch_size, num_heads, output_size]
        logits = torch.stack(logits, dim=1)
        return logits


@auto_docstring
@dataclass
class Seq2SeqLMOutputLA(Seq2SeqLMOutput):
    lookahead_logits: torch.FloatTensor = None
    lookahead_loss: Optional[torch.FloatTensor] = None


class T5LaModel(T5Model):
    pass


class T5LaForConditionalGeneration(T5ForConditionalGeneration):
    pass


class T5LaEncoderModel(T5EncoderModel):
    pass


class T5LaForSequenceClassification(T5ForSequenceClassification):
    pass


class T5LaForTokenClassification(T5ForTokenClassification):
    pass


class T5LaForQuestionAnswering(T5ForQuestionAnswering):
    pass


__all__ = [
    "T5LaConfig",
    "T5LaOnnxConfig",
    "T5LaEncoderModel",
    "T5LaForConditionalGeneration",
    "T5LaModel",
    "T5LaPreTrainedModel",
    "load_tf_weights_in_t5",
    "T5LaForQuestionAnswering",
    "T5LaForSequenceClassification",
    "T5LaForTokenClassification",
]
