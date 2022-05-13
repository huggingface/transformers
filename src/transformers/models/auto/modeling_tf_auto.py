# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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


import warnings
from collections import OrderedDict

from ...utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


TF_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("speech_to_text", "TFSpeech2TextModel"),
        ("clip", "TFCLIPModel"),
        ("deberta-v2", "TFDebertaV2Model"),
        ("deberta", "TFDebertaModel"),
        ("rembert", "TFRemBertModel"),
        ("roformer", "TFRoFormerModel"),
        ("convbert", "TFConvBertModel"),
        ("convnext", "TFConvNextModel"),
        ("data2vec-vision", "TFData2VecVisionModel"),
        ("led", "TFLEDModel"),
        ("lxmert", "TFLxmertModel"),
        ("mt5", "TFMT5Model"),
        ("t5", "TFT5Model"),
        ("distilbert", "TFDistilBertModel"),
        ("albert", "TFAlbertModel"),
        ("bart", "TFBartModel"),
        ("camembert", "TFCamembertModel"),
        ("xlm-roberta", "TFXLMRobertaModel"),
        ("longformer", "TFLongformerModel"),
        ("roberta", "TFRobertaModel"),
        ("layoutlm", "TFLayoutLMModel"),
        ("bert", "TFBertModel"),
        ("openai-gpt", "TFOpenAIGPTModel"),
        ("gpt2", "TFGPT2Model"),
        ("gptj", "TFGPTJModel"),
        ("mobilebert", "TFMobileBertModel"),
        ("transfo-xl", "TFTransfoXLModel"),
        ("xlnet", "TFXLNetModel"),
        ("flaubert", "TFFlaubertModel"),
        ("xlm", "TFXLMModel"),
        ("ctrl", "TFCTRLModel"),
        ("electra", "TFElectraModel"),
        ("funnel", ("TFFunnelModel", "TFFunnelBaseModel")),
        ("dpr", "TFDPRQuestionEncoder"),
        ("mpnet", "TFMPNetModel"),
        ("tapas", "TFTapasModel"),
        ("mbart", "TFMBartModel"),
        ("marian", "TFMarianModel"),
        ("pegasus", "TFPegasusModel"),
        ("blenderbot", "TFBlenderbotModel"),
        ("blenderbot-small", "TFBlenderbotSmallModel"),
        ("vit", "TFViTModel"),
        ("vit_mae", "TFViTMAEModel"),
        ("wav2vec2", "TFWav2Vec2Model"),
        ("hubert", "TFHubertModel"),
    ]
)

TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("lxmert", "TFLxmertForPreTraining"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("albert", "TFAlbertForPreTraining"),
        ("bart", "TFBartForConditionalGeneration"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("bert", "TFBertForPreTraining"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("mobilebert", "TFMobileBertForPreTraining"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("xlnet", "TFXLNetLMHeadModel"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("electra", "TFElectraForPreTraining"),
        ("tapas", "TFTapasForMaskedLM"),
        ("funnel", "TFFunnelForPreTraining"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("vit_mae", "TFViTMAEForPreTraining"),
    ]
)

TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),
        ("rembert", "TFRemBertForMaskedLM"),
        ("roformer", "TFRoFormerForMaskedLM"),
        ("convbert", "TFConvBertForMaskedLM"),
        ("led", "TFLEDForConditionalGeneration"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("albert", "TFAlbertForMaskedLM"),
        ("marian", "TFMarianMTModel"),
        ("bart", "TFBartForConditionalGeneration"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("longformer", "TFLongformerForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("bert", "TFBertForMaskedLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("gptj", "TFGPTJForCausalLM"),
        ("mobilebert", "TFMobileBertForMaskedLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("xlnet", "TFXLNetLMHeadModel"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("electra", "TFElectraForMaskedLM"),
        ("tapas", "TFTapasForMaskedLM"),
        ("funnel", "TFFunnelForMaskedLM"),
        ("mpnet", "TFMPNetForMaskedLM"),
    ]
)

TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("camembert", "TFCamembertForCausalLM"),
        ("rembert", "TFRemBertForCausalLM"),
        ("roformer", "TFRoFormerForCausalLM"),
        ("roberta", "TFRobertaForCausalLM"),
        ("bert", "TFBertLMHeadModel"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("gptj", "TFGPTJForCausalLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("xlnet", "TFXLNetLMHeadModel"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("ctrl", "TFCTRLLMHeadModel"),
    ]
)

TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("vit", "TFViTForImageClassification"),
        ("convnext", "TFConvNextForImageClassification"),
        ("data2vec-vision", "TFData2VecVisionForImageClassification"),
    ]
)

TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "TFVisionEncoderDecoderModel"),
    ]
)

TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("deberta-v2", "TFDebertaV2ForMaskedLM"),
        ("deberta", "TFDebertaForMaskedLM"),
        ("rembert", "TFRemBertForMaskedLM"),
        ("roformer", "TFRoFormerForMaskedLM"),
        ("convbert", "TFConvBertForMaskedLM"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("albert", "TFAlbertForMaskedLM"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("longformer", "TFLongformerForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("bert", "TFBertForMaskedLM"),
        ("mobilebert", "TFMobileBertForMaskedLM"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("electra", "TFElectraForMaskedLM"),
        ("tapas", "TFTapasForMaskedLM"),
        ("funnel", "TFFunnelForMaskedLM"),
        ("mpnet", "TFMPNetForMaskedLM"),
    ]
)

TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("led", "TFLEDForConditionalGeneration"),
        ("mt5", "TFMT5ForConditionalGeneration"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("marian", "TFMarianMTModel"),
        ("mbart", "TFMBartForConditionalGeneration"),
        ("pegasus", "TFPegasusForConditionalGeneration"),
        ("blenderbot", "TFBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "TFBlenderbotSmallForConditionalGeneration"),
        ("bart", "TFBartForConditionalGeneration"),
        ("encoder-decoder", "TFEncoderDecoderModel"),
    ]
)

TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),
    ]
)

TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("deberta-v2", "TFDebertaV2ForSequenceClassification"),
        ("deberta", "TFDebertaForSequenceClassification"),
        ("rembert", "TFRemBertForSequenceClassification"),
        ("roformer", "TFRoFormerForSequenceClassification"),
        ("convbert", "TFConvBertForSequenceClassification"),
        ("distilbert", "TFDistilBertForSequenceClassification"),
        ("albert", "TFAlbertForSequenceClassification"),
        ("camembert", "TFCamembertForSequenceClassification"),
        ("xlm-roberta", "TFXLMRobertaForSequenceClassification"),
        ("longformer", "TFLongformerForSequenceClassification"),
        ("roberta", "TFRobertaForSequenceClassification"),
        ("layoutlm", "TFLayoutLMForSequenceClassification"),
        ("bert", "TFBertForSequenceClassification"),
        ("xlnet", "TFXLNetForSequenceClassification"),
        ("mobilebert", "TFMobileBertForSequenceClassification"),
        ("flaubert", "TFFlaubertForSequenceClassification"),
        ("xlm", "TFXLMForSequenceClassification"),
        ("electra", "TFElectraForSequenceClassification"),
        ("tapas", "TFTapasForSequenceClassification"),
        ("funnel", "TFFunnelForSequenceClassification"),
        ("gpt2", "TFGPT2ForSequenceClassification"),
        ("gptj", "TFGPTJForSequenceClassification"),
        ("mpnet", "TFMPNetForSequenceClassification"),
        ("openai-gpt", "TFOpenAIGPTForSequenceClassification"),
        ("transfo-xl", "TFTransfoXLForSequenceClassification"),
        ("ctrl", "TFCTRLForSequenceClassification"),
    ]
)

TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("deberta-v2", "TFDebertaV2ForQuestionAnswering"),
        ("deberta", "TFDebertaForQuestionAnswering"),
        ("rembert", "TFRemBertForQuestionAnswering"),
        ("roformer", "TFRoFormerForQuestionAnswering"),
        ("convbert", "TFConvBertForQuestionAnswering"),
        ("distilbert", "TFDistilBertForQuestionAnswering"),
        ("albert", "TFAlbertForQuestionAnswering"),
        ("camembert", "TFCamembertForQuestionAnswering"),
        ("xlm-roberta", "TFXLMRobertaForQuestionAnswering"),
        ("longformer", "TFLongformerForQuestionAnswering"),
        ("roberta", "TFRobertaForQuestionAnswering"),
        ("bert", "TFBertForQuestionAnswering"),
        ("xlnet", "TFXLNetForQuestionAnsweringSimple"),
        ("mobilebert", "TFMobileBertForQuestionAnswering"),
        ("flaubert", "TFFlaubertForQuestionAnsweringSimple"),
        ("xlm", "TFXLMForQuestionAnsweringSimple"),
        ("electra", "TFElectraForQuestionAnswering"),
        ("funnel", "TFFunnelForQuestionAnswering"),
        ("gptj", "TFGPTJForQuestionAnswering"),
        ("mpnet", "TFMPNetForQuestionAnswering"),
    ]
)

TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TFTapasForQuestionAnswering"),
    ]
)


TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("deberta-v2", "TFDebertaV2ForTokenClassification"),
        ("deberta", "TFDebertaForTokenClassification"),
        ("rembert", "TFRemBertForTokenClassification"),
        ("roformer", "TFRoFormerForTokenClassification"),
        ("convbert", "TFConvBertForTokenClassification"),
        ("distilbert", "TFDistilBertForTokenClassification"),
        ("albert", "TFAlbertForTokenClassification"),
        ("camembert", "TFCamembertForTokenClassification"),
        ("flaubert", "TFFlaubertForTokenClassification"),
        ("xlm", "TFXLMForTokenClassification"),
        ("xlm-roberta", "TFXLMRobertaForTokenClassification"),
        ("longformer", "TFLongformerForTokenClassification"),
        ("roberta", "TFRobertaForTokenClassification"),
        ("layoutlm", "TFLayoutLMForTokenClassification"),
        ("bert", "TFBertForTokenClassification"),
        ("mobilebert", "TFMobileBertForTokenClassification"),
        ("xlnet", "TFXLNetForTokenClassification"),
        ("electra", "TFElectraForTokenClassification"),
        ("funnel", "TFFunnelForTokenClassification"),
        ("mpnet", "TFMPNetForTokenClassification"),
    ]
)

TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("rembert", "TFRemBertForMultipleChoice"),
        ("roformer", "TFRoFormerForMultipleChoice"),
        ("convbert", "TFConvBertForMultipleChoice"),
        ("camembert", "TFCamembertForMultipleChoice"),
        ("xlm", "TFXLMForMultipleChoice"),
        ("xlm-roberta", "TFXLMRobertaForMultipleChoice"),
        ("longformer", "TFLongformerForMultipleChoice"),
        ("roberta", "TFRobertaForMultipleChoice"),
        ("bert", "TFBertForMultipleChoice"),
        ("distilbert", "TFDistilBertForMultipleChoice"),
        ("mobilebert", "TFMobileBertForMultipleChoice"),
        ("xlnet", "TFXLNetForMultipleChoice"),
        ("flaubert", "TFFlaubertForMultipleChoice"),
        ("albert", "TFAlbertForMultipleChoice"),
        ("electra", "TFElectraForMultipleChoice"),
        ("funnel", "TFFunnelForMultipleChoice"),
        ("mpnet", "TFMPNetForMultipleChoice"),
    ]
)

TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "TFBertForNextSentencePrediction"),
        ("mobilebert", "TFMobileBertForNextSentencePrediction"),
    ]
)


TF_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_MAPPING_NAMES)
TF_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
TF_MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES)
TF_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
TF_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)


class TFAutoModel(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_MAPPING


TFAutoModel = auto_class_update(TFAutoModel)


class TFAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_PRETRAINING_MAPPING


TFAutoModelForPreTraining = auto_class_update(TFAutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _TFAutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_WITH_LM_HEAD_MAPPING


_TFAutoModelWithLMHead = auto_class_update(_TFAutoModelWithLMHead, head_doc="language modeling")


class TFAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING


TFAutoModelForCausalLM = auto_class_update(TFAutoModelForCausalLM, head_doc="causal language modeling")


class TFAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


TFAutoModelForImageClassification = auto_class_update(
    TFAutoModelForImageClassification, head_doc="image classification"
)


class TFAutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_VISION_2_SEQ_MAPPING


TFAutoModelForVision2Seq = auto_class_update(TFAutoModelForVision2Seq, head_doc="vision-to-text modeling")


class TFAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MASKED_LM_MAPPING


TFAutoModelForMaskedLM = auto_class_update(TFAutoModelForMaskedLM, head_doc="masked language modeling")


class TFAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


TFAutoModelForSeq2SeqLM = auto_class_update(
    TFAutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


TFAutoModelForSequenceClassification = auto_class_update(
    TFAutoModelForSequenceClassification, head_doc="sequence classification"
)


class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING


TFAutoModelForQuestionAnswering = auto_class_update(TFAutoModelForQuestionAnswering, head_doc="question answering")


class TFAutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


TFAutoModelForTableQuestionAnswering = auto_class_update(
    TFAutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class TFAutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


TFAutoModelForTokenClassification = auto_class_update(
    TFAutoModelForTokenClassification, head_doc="token classification"
)


class TFAutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


TFAutoModelForMultipleChoice = auto_class_update(TFAutoModelForMultipleChoice, head_doc="multiple choice")


class TFAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


TFAutoModelForNextSentencePrediction = auto_class_update(
    TFAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class TFAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


TFAutoModelForSpeechSeq2Seq = auto_class_update(
    TFAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
