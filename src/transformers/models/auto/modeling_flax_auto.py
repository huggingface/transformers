# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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
""" Auto Model class. """


from collections import OrderedDict

from ...utils import logging
from ..bart.modeling_flax_bart import (
    FlaxBartForConditionalGeneration,
    FlaxBartForQuestionAnswering,
    FlaxBartForSequenceClassification,
    FlaxBartModel,
)
from ..bert.modeling_flax_bert import (
    FlaxBertForMaskedLM,
    FlaxBertForMultipleChoice,
    FlaxBertForNextSentencePrediction,
    FlaxBertForPreTraining,
    FlaxBertForQuestionAnswering,
    FlaxBertForSequenceClassification,
    FlaxBertForTokenClassification,
    FlaxBertModel,
)
from ..big_bird.modeling_flax_big_bird import (
    FlaxBigBirdForMaskedLM,
    FlaxBigBirdForMultipleChoice,
    FlaxBigBirdForPreTraining,
    FlaxBigBirdForQuestionAnswering,
    FlaxBigBirdForSequenceClassification,
    FlaxBigBirdForTokenClassification,
    FlaxBigBirdModel,
)
from ..clip.modeling_flax_clip import FlaxCLIPModel
from ..electra.modeling_flax_electra import (
    FlaxElectraForMaskedLM,
    FlaxElectraForMultipleChoice,
    FlaxElectraForPreTraining,
    FlaxElectraForQuestionAnswering,
    FlaxElectraForSequenceClassification,
    FlaxElectraForTokenClassification,
    FlaxElectraModel,
)
from ..gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model
from ..roberta.modeling_flax_roberta import (
    FlaxRobertaForMaskedLM,
    FlaxRobertaForMultipleChoice,
    FlaxRobertaForQuestionAnswering,
    FlaxRobertaForSequenceClassification,
    FlaxRobertaForTokenClassification,
    FlaxRobertaModel,
)
from ..t5.modeling_flax_t5 import FlaxT5ForConditionalGeneration, FlaxT5Model
from ..vit.modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel
from .auto_factory import auto_class_factory
from .configuration_auto import (
    BartConfig,
    BertConfig,
    BigBirdConfig,
    CLIPConfig,
    ElectraConfig,
    GPT2Config,
    RobertaConfig,
    T5Config,
    ViTConfig,
)


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (RobertaConfig, FlaxRobertaModel),
        (BertConfig, FlaxBertModel),
        (BigBirdConfig, FlaxBigBirdModel),
        (BartConfig, FlaxBartModel),
        (GPT2Config, FlaxGPT2Model),
        (ElectraConfig, FlaxElectraModel),
        (CLIPConfig, FlaxCLIPModel),
        (ViTConfig, FlaxViTModel),
        (T5Config, FlaxT5Model),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (RobertaConfig, FlaxRobertaForMaskedLM),
        (BertConfig, FlaxBertForPreTraining),
        (BigBirdConfig, FlaxBigBirdForPreTraining),
        (BartConfig, FlaxBartForConditionalGeneration),
        (ElectraConfig, FlaxElectraForPreTraining),
        (T5Config, FlaxT5ForConditionalGeneration),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (RobertaConfig, FlaxRobertaForMaskedLM),
        (BertConfig, FlaxBertForMaskedLM),
        (BigBirdConfig, FlaxBigBirdForMaskedLM),
        (BartConfig, FlaxBartForConditionalGeneration),
        (ElectraConfig, FlaxElectraForMaskedLM),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (BartConfig, FlaxBartForConditionalGeneration),
        (T5Config, FlaxT5ForConditionalGeneration),
    ]
)

FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image-classsification
        (ViTConfig, FlaxViTForImageClassification),
    ]
)

FLAX_MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (GPT2Config, FlaxGPT2LMHeadModel)
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (BartConfig, FlaxBartForConditionalGeneration)
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (RobertaConfig, FlaxRobertaForSequenceClassification),
        (BertConfig, FlaxBertForSequenceClassification),
        (BigBirdConfig, FlaxBigBirdForSequenceClassification),
        (BartConfig, FlaxBartForSequenceClassification),
        (ElectraConfig, FlaxElectraForSequenceClassification),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (RobertaConfig, FlaxRobertaForQuestionAnswering),
        (BertConfig, FlaxBertForQuestionAnswering),
        (BigBirdConfig, FlaxBigBirdForQuestionAnswering),
        (BartConfig, FlaxBartForQuestionAnswering),
        (ElectraConfig, FlaxElectraForQuestionAnswering),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (RobertaConfig, FlaxRobertaForTokenClassification),
        (BertConfig, FlaxBertForTokenClassification),
        (BigBirdConfig, FlaxBigBirdForTokenClassification),
        (ElectraConfig, FlaxElectraForTokenClassification),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (RobertaConfig, FlaxRobertaForMultipleChoice),
        (BertConfig, FlaxBertForMultipleChoice),
        (BigBirdConfig, FlaxBigBirdForMultipleChoice),
        (ElectraConfig, FlaxElectraForMultipleChoice),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, FlaxBertForNextSentencePrediction),
    ]
)

FlaxAutoModel = auto_class_factory("FlaxAutoModel", FLAX_MODEL_MAPPING)

FlaxAutoModelForImageClassification = auto_class_factory(
    "FlaxAutoModelForImageClassification",
    FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    head_doc="image classification modeling",
)

FlaxAutoModelForCausalLM = auto_class_factory(
    "FlaxAutoModelForCausalLM", FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, head_doc="causal language modeling"
)

FlaxAutoModelForPreTraining = auto_class_factory(
    "FlaxAutoModelForPreTraining", FLAX_MODEL_FOR_PRETRAINING_MAPPING, head_doc="pretraining"
)

FlaxAutoModelForMaskedLM = auto_class_factory(
    "FlaxAutoModelForMaskedLM", FLAX_MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked language modeling"
)


FlaxAutoModelForSeq2SeqLM = auto_class_factory(
    "FlaxAutoModelForSeq2SeqLM",
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    head_doc="sequence-to-sequence language modeling",
)

FlaxAutoModelForSequenceClassification = auto_class_factory(
    "FlaxAutoModelForSequenceClassification",
    FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    head_doc="sequence classification",
)

FlaxAutoModelForQuestionAnswering = auto_class_factory(
    "FlaxAutoModelForQuestionAnswering", FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

FlaxAutoModelForTokenClassification = auto_class_factory(
    "FlaxAutoModelForTokenClassification", FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

FlaxAutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)

FlaxAutoModelForNextSentencePrediction = auto_class_factory(
    "FlaxAutoModelForNextSentencePrediction",
    FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    head_doc="next sentence prediction",
)

FlaxAutoModelForSeq2SeqLM = auto_class_factory(
    "FlaxAutoModelForSeq2SeqLM",
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    head_doc="sequence-to-sequence language modeling",
)
