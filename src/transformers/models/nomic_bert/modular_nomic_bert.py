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

from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertAttention,
    BertCrossAttention,
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertIntermediate,
    BertLayer,
    BertLMHeadModel,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertOutput,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertPreTrainingHeads,
    BertSelfAttention,
    BertSelfOutput,
)


class NomicBertConfig(BertConfig):
    pass


class NomicBertEmbeddings(BertEmbeddings):
    pass


class NomicBertSelfAttention(BertSelfAttention):
    pass


class NomicBertCrossAttention(BertCrossAttention):
    pass


class NomicBertSelfOutput(BertSelfOutput):
    pass


class NomicBertAttention(BertAttention):
    pass


class NomicBertIntermediate(BertIntermediate):
    pass


class NomicBertOutput(BertOutput):
    pass


class NomicBertLayer(BertLayer):
    pass


class NomicBertEncoder(BertEncoder):
    pass


class NomicBertPooler(BertPooler):
    pass


class NomicBertPredictionHeadTransform(BertPredictionHeadTransform):
    pass


class NomicBertLMPredictionHead(BertLMPredictionHead):
    pass


class NomicBertOnlyMLMHead(BertOnlyMLMHead):
    pass


class NomicBertOnlyNSPHead(BertOnlyNSPHead):
    pass


class NomicBertPreTrainingHeads(BertPreTrainingHeads):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    pass


class NomicBertForPreTrainingOutput(BertForPreTrainingOutput):
    pass


class NomicBertModel(BertModel):
    pass


class NomicBertForPreTraining(BertForPreTraining):
    pass


class NomicBertLMHeadModel(BertLMHeadModel):
    pass


class NomicBertForMaskedLM(BertForMaskedLM):
    pass


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    pass


class NomicBertForSequenceClassification(BertForSequenceClassification):
    pass


class NomicBertForMultipleChoice(BertForMultipleChoice):
    pass


class NomicBertForTokenClassification(BertForTokenClassification):
    pass


class NomicBertForQuestionAnswering(BertForQuestionAnswering):
    pass


__all__ = [
    "NomicBertConfig",
    "NomicBertForMaskedLM",
    "NomicBertForMultipleChoice",
    "NomicBertForNextSentencePrediction",
    "NomicBertForPreTraining",
    "NomicBertForQuestionAnswering",
    "NomicBertForSequenceClassification",
    "NomicBertForTokenClassification",
    "NomicBertLayer",
    "NomicBertLMHeadModel",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
