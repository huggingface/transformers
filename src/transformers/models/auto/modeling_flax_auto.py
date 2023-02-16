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
""" Auto Model class."""


from collections import OrderedDict

from ...utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "FlaxAlbertModel"),
        ("bart", "FlaxBartModel"),
        ("beit", "FlaxBeitModel"),
        ("bert", "FlaxBertModel"),
        ("big_bird", "FlaxBigBirdModel"),
        ("blenderbot", "FlaxBlenderbotModel"),
        ("blenderbot-small", "FlaxBlenderbotSmallModel"),
        ("clip", "FlaxCLIPModel"),
        ("distilbert", "FlaxDistilBertModel"),
        ("electra", "FlaxElectraModel"),
        ("gpt-sw3", "FlaxGPT2Model"),
        ("gpt2", "FlaxGPT2Model"),
        ("gpt_neo", "FlaxGPTNeoModel"),
        ("gptj", "FlaxGPTJModel"),
        ("longt5", "FlaxLongT5Model"),
        ("marian", "FlaxMarianModel"),
        ("mbart", "FlaxMBartModel"),
        ("mt5", "FlaxMT5Model"),
        ("opt", "FlaxOPTModel"),
        ("pegasus", "FlaxPegasusModel"),
        ("roberta", "FlaxRobertaModel"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormModel"),
        ("roformer", "FlaxRoFormerModel"),
        ("t5", "FlaxT5Model"),
        ("vision-text-dual-encoder", "FlaxVisionTextDualEncoderModel"),
        ("vit", "FlaxViTModel"),
        ("wav2vec2", "FlaxWav2Vec2Model"),
        ("xglm", "FlaxXGLMModel"),
        ("xlm-roberta", "FlaxXLMRobertaModel"),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "FlaxAlbertForPreTraining"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("bert", "FlaxBertForPreTraining"),
        ("big_bird", "FlaxBigBirdForPreTraining"),
        ("electra", "FlaxElectraForPreTraining"),
        ("longt5", "FlaxLongT5ForConditionalGeneration"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("mt5", "FlaxMT5ForConditionalGeneration"),
        ("roberta", "FlaxRobertaForMaskedLM"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM"),
        ("roformer", "FlaxRoFormerForMaskedLM"),
        ("t5", "FlaxT5ForConditionalGeneration"),
        ("wav2vec2", "FlaxWav2Vec2ForPreTraining"),
        ("xlm-roberta", "FlaxXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "FlaxAlbertForMaskedLM"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("bert", "FlaxBertForMaskedLM"),
        ("big_bird", "FlaxBigBirdForMaskedLM"),
        ("distilbert", "FlaxDistilBertForMaskedLM"),
        ("electra", "FlaxElectraForMaskedLM"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("roberta", "FlaxRobertaForMaskedLM"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM"),
        ("roformer", "FlaxRoFormerForMaskedLM"),
        ("xlm-roberta", "FlaxXLMRobertaForMaskedLM"),
    ]
)

FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "FlaxBartForConditionalGeneration"),
        ("blenderbot", "FlaxBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "FlaxBlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "FlaxEncoderDecoderModel"),
        ("longt5", "FlaxLongT5ForConditionalGeneration"),
        ("marian", "FlaxMarianMTModel"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("mt5", "FlaxMT5ForConditionalGeneration"),
        ("pegasus", "FlaxPegasusForConditionalGeneration"),
        ("t5", "FlaxT5ForConditionalGeneration"),
    ]
)

FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("beit", "FlaxBeitForImageClassification"),
        ("vit", "FlaxViTForImageClassification"),
    ]
)

FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "FlaxVisionEncoderDecoderModel"),
    ]
)

FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "FlaxBartForCausalLM"),
        ("bert", "FlaxBertForCausalLM"),
        ("big_bird", "FlaxBigBirdForCausalLM"),
        ("electra", "FlaxElectraForCausalLM"),
        ("gpt-sw3", "FlaxGPT2LMHeadModel"),
        ("gpt2", "FlaxGPT2LMHeadModel"),
        ("gpt_neo", "FlaxGPTNeoForCausalLM"),
        ("gptj", "FlaxGPTJForCausalLM"),
        ("opt", "FlaxOPTForCausalLM"),
        ("roberta", "FlaxRobertaForCausalLM"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForCausalLM"),
        ("xglm", "FlaxXGLMForCausalLM"),
        ("xlm-roberta", "FlaxXLMRobertaForCausalLM"),
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "FlaxAlbertForSequenceClassification"),
        ("bart", "FlaxBartForSequenceClassification"),
        ("bert", "FlaxBertForSequenceClassification"),
        ("big_bird", "FlaxBigBirdForSequenceClassification"),
        ("distilbert", "FlaxDistilBertForSequenceClassification"),
        ("electra", "FlaxElectraForSequenceClassification"),
        ("mbart", "FlaxMBartForSequenceClassification"),
        ("roberta", "FlaxRobertaForSequenceClassification"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForSequenceClassification"),
        ("roformer", "FlaxRoFormerForSequenceClassification"),
        ("xlm-roberta", "FlaxXLMRobertaForSequenceClassification"),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "FlaxAlbertForQuestionAnswering"),
        ("bart", "FlaxBartForQuestionAnswering"),
        ("bert", "FlaxBertForQuestionAnswering"),
        ("big_bird", "FlaxBigBirdForQuestionAnswering"),
        ("distilbert", "FlaxDistilBertForQuestionAnswering"),
        ("electra", "FlaxElectraForQuestionAnswering"),
        ("mbart", "FlaxMBartForQuestionAnswering"),
        ("roberta", "FlaxRobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForQuestionAnswering"),
        ("roformer", "FlaxRoFormerForQuestionAnswering"),
        ("xlm-roberta", "FlaxXLMRobertaForQuestionAnswering"),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "FlaxAlbertForTokenClassification"),
        ("bert", "FlaxBertForTokenClassification"),
        ("big_bird", "FlaxBigBirdForTokenClassification"),
        ("distilbert", "FlaxDistilBertForTokenClassification"),
        ("electra", "FlaxElectraForTokenClassification"),
        ("roberta", "FlaxRobertaForTokenClassification"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForTokenClassification"),
        ("roformer", "FlaxRoFormerForTokenClassification"),
        ("xlm-roberta", "FlaxXLMRobertaForTokenClassification"),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "FlaxAlbertForMultipleChoice"),
        ("bert", "FlaxBertForMultipleChoice"),
        ("big_bird", "FlaxBigBirdForMultipleChoice"),
        ("distilbert", "FlaxDistilBertForMultipleChoice"),
        ("electra", "FlaxElectraForMultipleChoice"),
        ("roberta", "FlaxRobertaForMultipleChoice"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMultipleChoice"),
        ("roformer", "FlaxRoFormerForMultipleChoice"),
        ("xlm-roberta", "FlaxXLMRobertaForMultipleChoice"),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "FlaxBertForNextSentencePrediction"),
    ]
)

FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech-encoder-decoder", "FlaxSpeechEncoderDecoderModel"),
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
FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
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
FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
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


class FlaxAutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING


FlaxAutoModelForVision2Seq = auto_class_update(FlaxAutoModelForVision2Seq, head_doc="vision-to-text modeling")


class FlaxAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


FlaxAutoModelForSpeechSeq2Seq = auto_class_update(
    FlaxAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)
