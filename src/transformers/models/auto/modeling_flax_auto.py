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
from .auto_factory import _BaseAutoModelClass, LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (RobertaConfig, FlaxRobertaModel),
        (BertConfig, FlaxBertModel),
        (BigBirdConfig, FlaxBigBirdModel),
        (BartConfig, FlaxBartModel),
        (GPT2Config, FlaxGPT2Model),
        (GPTNeoConfig, FlaxGPTNeoModel),
        (ElectraConfig, FlaxElectraModel),
        (CLIPConfig, FlaxCLIPModel),
        (ViTConfig, FlaxViTModel),
        (MBartConfig, FlaxMBartModel),
        (T5Config, FlaxT5Model),
        (MT5Config, FlaxMT5Model),
        (Wav2Vec2Config, FlaxWav2Vec2Model),
        (MarianConfig, FlaxMarianModel),
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
        (MBartConfig, FlaxMBartForConditionalGeneration),
        (T5Config, FlaxT5ForConditionalGeneration),
        (MT5Config, FlaxMT5ForConditionalGeneration),
        (Wav2Vec2Config, FlaxWav2Vec2ForPreTraining),
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
        (MBartConfig, FlaxMBartForConditionalGeneration),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (BartConfig, FlaxBartForConditionalGeneration),
        (T5Config, FlaxT5ForConditionalGeneration),
        (MT5Config, FlaxMT5ForConditionalGeneration),
        (MarianConfig, FlaxMarianMTModel),
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
        (GPT2Config, FlaxGPT2LMHeadModel),
        (GPTNeoConfig, FlaxGPTNeoForCausalLM),
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
        (MBartConfig, FlaxMBartForSequenceClassification),
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
        (MBartConfig, FlaxMBartForQuestionAnswering),
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


class FlaxAutoModel(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_MAPPING


FlaxAutoModel = auto_class_update(FlaxAutoModel)


class FlaxAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_PRETRAINING_MAPPING


FlaxAutoModelForPreTraining = auto_class_update(FlaxAutoModelForPreTraining, head_doc="pretraining")


class FlaxAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING


FlaxAutoModelForCausalLM = auto_class_update(FlaxAutoModelForCausalLM, head_doc="causal language modeling")


class FlaxAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_MASKED_LM_MAPPING


FlaxAutoModelForMaskedLM = auto_class_update(FlaxAutoModelForMaskedLM, head_doc="masked language modeling")


class FlaxAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


FlaxAutoModelForSeq2SeqLM = auto_class_update(
    FlaxAutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class FlaxAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


FlaxAutoModelForSequenceClassification = auto_class_update(
    FlaxAutoModelForSequenceClassification, head_doc="sequence classification"
)


class FlaxAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING


FlaxAutoModelForQuestionAnswering = auto_class_update(FlaxAutoModelForQuestionAnswering, head_doc="question answering")


class FlaxAutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


FlaxAutoModelForTokenClassification = auto_class_update(
    FlaxAutoModelForTokenClassification, head_doc="token classification"
)


class FlaxAutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


FlaxAutoModelForMultipleChoice = auto_class_update(FlaxAutoModelForMultipleChoice, head_doc="multiple choice")


class FlaxAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


FlaxAutoModelForNextSentencePrediction = auto_class_update(
    FlaxAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class FlaxAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


FlaxAutoModelForImageClassification = auto_class_update(
    FlaxAutoModelForImageClassification, head_doc="image classification"
)
