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
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("pegasus", "FlaxPegasusModel"),
        ("distilbert", "FlaxDistilBertModel"),
        ("albert", "FlaxAlbertModel"),
        ("roberta", "FlaxRobertaModel"),
        ("bert", "FlaxBertModel"),
        ("beit", "FlaxBeitModel"),
        ("big_bird", "FlaxBigBirdModel"),
        ("bart", "FlaxBartModel"),
        ("gpt2", "FlaxGPT2Model"),
        ("gpt_neo", "FlaxGPTNeoModel"),
        ("electra", "FlaxElectraModel"),
        ("clip", "FlaxCLIPModel"),
        ("vit", "FlaxViTModel"),
        ("mbart", "FlaxMBartModel"),
        ("t5", "FlaxT5Model"),
        ("mt5", "FlaxMT5Model"),
        ("wav2vec2", "FlaxWav2Vec2Model"),
        ("marian", "FlaxMarianModel"),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "FlaxAlbertForPreTraining"),
        ("roberta", "FlaxRobertaForMaskedLM"),
        ("bert", "FlaxBertForPreTraining"),
        ("big_bird", "FlaxBigBirdForPreTraining"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("electra", "FlaxElectraForPreTraining"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("t5", "FlaxT5ForConditionalGeneration"),
        ("mt5", "FlaxMT5ForConditionalGeneration"),
        ("wav2vec2", "FlaxWav2Vec2ForPreTraining"),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("distilbert", "FlaxDistilBertForMaskedLM"),
        ("albert", "FlaxAlbertForMaskedLM"),
        ("roberta", "FlaxRobertaForMaskedLM"),
        ("bert", "FlaxBertForMaskedLM"),
        ("big_bird", "FlaxBigBirdForMaskedLM"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("electra", "FlaxElectraForMaskedLM"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("pegasus", "FlaxPegasusForConditionalGeneration"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("t5", "FlaxT5ForConditionalGeneration"),
        ("mt5", "FlaxMT5ForConditionalGeneration"),
        ("marian", "FlaxMarianMTModel"),
        ("encoder-decoder", "FlaxEncoderDecoderModel"),
    ]
)

FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("vit", "FlaxViTForImageClassification"),
        ("beit", "FlaxBeitForImageClassification"),
    ]
)

FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("gpt2", "FlaxGPT2LMHeadModel"),
        ("gpt_neo", "FlaxGPTNeoForCausalLM"),
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("distilbert", "FlaxDistilBertForSequenceClassification"),
        ("albert", "FlaxAlbertForSequenceClassification"),
        ("roberta", "FlaxRobertaForSequenceClassification"),
        ("bert", "FlaxBertForSequenceClassification"),
        ("big_bird", "FlaxBigBirdForSequenceClassification"),
        ("bart", "FlaxBartForSequenceClassification"),
        ("electra", "FlaxElectraForSequenceClassification"),
        ("mbart", "FlaxMBartForSequenceClassification"),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("distilbert", "FlaxDistilBertForQuestionAnswering"),
        ("albert", "FlaxAlbertForQuestionAnswering"),
        ("roberta", "FlaxRobertaForQuestionAnswering"),
        ("bert", "FlaxBertForQuestionAnswering"),
        ("big_bird", "FlaxBigBirdForQuestionAnswering"),
        ("bart", "FlaxBartForQuestionAnswering"),
        ("electra", "FlaxElectraForQuestionAnswering"),
        ("mbart", "FlaxMBartForQuestionAnswering"),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("distilbert", "FlaxDistilBertForTokenClassification"),
        ("albert", "FlaxAlbertForTokenClassification"),
        ("roberta", "FlaxRobertaForTokenClassification"),
        ("bert", "FlaxBertForTokenClassification"),
        ("big_bird", "FlaxBigBirdForTokenClassification"),
        ("electra", "FlaxElectraForTokenClassification"),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("distilbert", "FlaxDistilBertForMultipleChoice"),
        ("albert", "FlaxAlbertForMultipleChoice"),
        ("roberta", "FlaxRobertaForMultipleChoice"),
        ("bert", "FlaxBertForMultipleChoice"),
        ("big_bird", "FlaxBigBirdForMultipleChoice"),
        ("electra", "FlaxElectraForMultipleChoice"),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "FlaxBertForNextSentencePrediction"),
    ]
)


FLAX_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_MAPPING_NAMES)
FLAX_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
FLAX_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
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
