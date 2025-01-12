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
"""Auto Model class."""

import warnings
from collections import OrderedDict

from ...utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


TF_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "TFAlbertModel"),
        ("bart", "TFBartModel"),
        ("bert", "TFBertModel"),
        ("brand-new-bert", "TFBrandNewBERTModel"),
        ("blenderbot", "TFBlenderbotModel"),
        ("blenderbot-small", "TFBlenderbotSmallModel"),
        ("blip", "TFBlipModel"),
        ("camembert", "TFCamembertModel"),
        ("clip", "TFCLIPModel"),
        ("convbert", "TFConvBertModel"),
        ("convnext", "TFConvNextModel"),
        ("convnextv2", "TFConvNextV2Model"),
        ("ctrl", "TFCTRLModel"),
        ("cvt", "TFCvtModel"),
        ("data2vec-vision", "TFData2VecVisionModel"),
        ("deberta", "TFDebertaModel"),
        ("deberta-v2", "TFDebertaV2Model"),
        ("deit", "TFDeiTModel"),
        ("distilbert", "TFDistilBertModel"),
        ("dpr", "TFDPRQuestionEncoder"),
        ("efficientformer", "TFEfficientFormerModel"),
        ("electra", "TFElectraModel"),
        ("esm", "TFEsmModel"),
        ("flaubert", "TFFlaubertModel"),
        ("funnel", ("TFFunnelModel", "TFFunnelBaseModel")),
        ("gpt-sw3", "TFGPT2Model"),
        ("gpt2", "TFGPT2Model"),
        ("gptj", "TFGPTJModel"),
        ("groupvit", "TFGroupViTModel"),
        ("hubert", "TFHubertModel"),
        ("idefics", "TFIdeficsModel"),
        ("layoutlm", "TFLayoutLMModel"),
        ("layoutlmv3", "TFLayoutLMv3Model"),
        ("led", "TFLEDModel"),
        ("longformer", "TFLongformerModel"),
        ("lxmert", "TFLxmertModel"),
        ("marian", "TFMarianModel"),
        ("mbart", "TFMBartModel"),
        ("mistral", "TFMistralModel"),
        ("mobilebert", "TFMobileBertModel"),
        ("mobilevit", "TFMobileViTModel"),
        ("mpnet", "TFMPNetModel"),
        ("mt5", "TFMT5Model"),
        ("openai-gpt", "TFOpenAIGPTModel"),
        ("opt", "TFOPTModel"),
        ("pegasus", "TFPegasusModel"),
        ("regnet", "TFRegNetModel"),
        ("rembert", "TFRemBertModel"),
        ("resnet", "TFResNetModel"),
        ("roberta", "TFRobertaModel"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        ("roformer", "TFRoFormerModel"),
        ("sam", "TFSamModel"),
        ("segformer", "TFSegformerModel"),
        ("speech_to_text", "TFSpeech2TextModel"),
        ("swiftformer", "TFSwiftFormerModel"),
        ("swin", "TFSwinModel"),
        ("t5", "TFT5Model"),
        ("tapas", "TFTapasModel"),
        ("transfo-xl", "TFTransfoXLModel"),
        ("vision-text-dual-encoder", "TFVisionTextDualEncoderModel"),
        ("vit", "TFViTModel"),
        ("vit_mae", "TFViTMAEModel"),
        ("wav2vec2", "TFWav2Vec2Model"),
        ("whisper", "TFWhisperModel"),
        ("xglm", "TFXGLMModel"),
        ("xlm", "TFXLMModel"),
        ("xlm-roberta", "TFXLMRobertaModel"),
        ("xlnet", "TFXLNetModel"),
    ]
)

TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "TFAlbertForPreTraining"),
        ("bart", "TFBartForConditionalGeneration"),
        ("bert", "TFBertForPreTraining"),
        ("brand-new-bert", "TFBrandNewBERTForPreTraining"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("electra", "TFElectraForPreTraining"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("funnel", "TFFunnelForPreTraining"),
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("idefics", "TFIdeficsForVisionText2Text"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("lxmert", "TFLxmertForPreTraining"),
        ("mobilebert", "TFMobileBertForPreTraining"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("tapas", "TFTapasForMaskedLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("vit_mae", "TFViTMAEForPreTraining"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "TFAlbertForMaskedLM"),
        ("bart", "TFBartForConditionalGeneration"),
        ("bert", "TFBertForMaskedLM"),
        ("brand-new-bert", "TFBrandNewBERTForMaskedLM"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("convbert", "TFConvBertForMaskedLM"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("electra", "TFElectraForMaskedLM"),
        ("esm", "TFEsmForMaskedLM"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("funnel", "TFFunnelForMaskedLM"),
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("gptj", "TFGPTJForCausalLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("led", "TFLEDForConditionalGeneration"),
        ("longformer", "TFLongformerForMaskedLM"),
        ("marian", "TFMarianMTModel"),
        ("mobilebert", "TFMobileBertForMaskedLM"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("rembert", "TFRemBertForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        ("roformer", "TFRoFormerForMaskedLM"),
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("tapas", "TFTapasForMaskedLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("whisper", "TFWhisperForConditionalGeneration"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bert", "TFBertLMHeadModel"),
        ("brand-new-bert", "TFBrandNewBERTLMHeadModel"),
        ("camembert", "TFCamembertForCausalLM"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("gptj", "TFGPTJForCausalLM"),
        ("mistral", "TFMistralForCausalLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("opt", "TFOPTForCausalLM"),
        ("rembert", "TFRemBertForCausalLM"),
        ("roberta", "TFRobertaForCausalLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForCausalLM"),
        ("roformer", "TFRoFormerForCausalLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("xglm", "TFXGLMForCausalLM"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForCausalLM"),
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "TFDeiTForMaskedImageModeling"),
        ("swin", "TFSwinForMaskedImageModeling"),
    ]
)

TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        ("convnext", "TFConvNextForImageClassification"),
        ("convnextv2", "TFConvNextV2ForImageClassification"),
        ("cvt", "TFCvtForImageClassification"),
        ("data2vec-vision", "TFData2VecVisionForImageClassification"),
        ("deit", ("TFDeiTForImageClassification", "TFDeiTForImageClassificationWithTeacher")),
        (
            "efficientformer",
            ("TFEfficientFormerForImageClassification", "TFEfficientFormerForImageClassificationWithTeacher"),
        ),
        ("mobilevit", "TFMobileViTForImageClassification"),
        ("regnet", "TFRegNetForImageClassification"),
        ("resnet", "TFResNetForImageClassification"),
        ("segformer", "TFSegformerForImageClassification"),
        ("swiftformer", "TFSwiftFormerForImageClassification"),
        ("swin", "TFSwinForImageClassification"),
        ("vit", "TFViTForImageClassification"),
    ]
)


TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("blip", "TFBlipModel"),
        ("clip", "TFCLIPModel"),
    ]
)


TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        ("data2vec-vision", "TFData2VecVisionForSemanticSegmentation"),
        ("mobilevit", "TFMobileViTForSemanticSegmentation"),
        ("segformer", "TFSegformerForSemanticSegmentation"),
    ]
)

TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "TFBlipForConditionalGeneration"),
        ("vision-encoder-decoder", "TFVisionEncoderDecoderModel"),
    ]
)

TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "TFAlbertForMaskedLM"),
        ("bert", "TFBertForMaskedLM"),
        ("brand-new-bert", "TFBrandNewBERTForMaskedLM"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("convbert", "TFConvBertForMaskedLM"),
        ("deberta", "TFDebertaForMaskedLM"),
        ("deberta-v2", "TFDebertaV2ForMaskedLM"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("electra", "TFElectraForMaskedLM"),
        ("esm", "TFEsmForMaskedLM"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("funnel", "TFFunnelForMaskedLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("longformer", "TFLongformerForMaskedLM"),
        ("mobilebert", "TFMobileBertForMaskedLM"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("rembert", "TFRemBertForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        ("roformer", "TFRoFormerForMaskedLM"),
        ("tapas", "TFTapasForMaskedLM"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
    ]
)

TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "TFBartForConditionalGeneration"),
        ("blenderbot", "TFBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "TFBlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "TFEncoderDecoderModel"),
        ("led", "TFLEDForConditionalGeneration"),
        ("marian", "TFMarianMTModel"),
        ("mbart", "TFMBartForConditionalGeneration"),
        ("mt5", "TFMT5ForConditionalGeneration"),
        ("pegasus", "TFPegasusForConditionalGeneration"),
        ("t5", "TFT5ForConditionalGeneration"),
    ]
)

TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),
        ("whisper", "TFWhisperForConditionalGeneration"),
    ]
)

TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "TFAlbertForSequenceClassification"),
        ("bart", "TFBartForSequenceClassification"),
        ("bert", "TFBertForSequenceClassification"),
        ("brand-new-bert", "TFBrandNewBERTForSequenceClassification"),
        ("camembert", "TFCamembertForSequenceClassification"),
        ("convbert", "TFConvBertForSequenceClassification"),
        ("ctrl", "TFCTRLForSequenceClassification"),
        ("deberta", "TFDebertaForSequenceClassification"),
        ("deberta-v2", "TFDebertaV2ForSequenceClassification"),
        ("distilbert", "TFDistilBertForSequenceClassification"),
        ("electra", "TFElectraForSequenceClassification"),
        ("esm", "TFEsmForSequenceClassification"),
        ("flaubert", "TFFlaubertForSequenceClassification"),
        ("funnel", "TFFunnelForSequenceClassification"),
        ("gpt-sw3", "TFGPT2ForSequenceClassification"),
        ("gpt2", "TFGPT2ForSequenceClassification"),
        ("gptj", "TFGPTJForSequenceClassification"),
        ("layoutlm", "TFLayoutLMForSequenceClassification"),
        ("layoutlmv3", "TFLayoutLMv3ForSequenceClassification"),
        ("longformer", "TFLongformerForSequenceClassification"),
        ("mistral", "TFMistralForSequenceClassification"),
        ("mobilebert", "TFMobileBertForSequenceClassification"),
        ("mpnet", "TFMPNetForSequenceClassification"),
        ("openai-gpt", "TFOpenAIGPTForSequenceClassification"),
        ("rembert", "TFRemBertForSequenceClassification"),
        ("roberta", "TFRobertaForSequenceClassification"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForSequenceClassification"),
        ("roformer", "TFRoFormerForSequenceClassification"),
        ("tapas", "TFTapasForSequenceClassification"),
        ("transfo-xl", "TFTransfoXLForSequenceClassification"),
        ("xlm", "TFXLMForSequenceClassification"),
        ("xlm-roberta", "TFXLMRobertaForSequenceClassification"),
        ("xlnet", "TFXLNetForSequenceClassification"),
    ]
)

TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "TFAlbertForQuestionAnswering"),
        ("bert", "TFBertForQuestionAnswering"),
        ("brand-new-bert", "TFBrandNewBERTForQuestionAnswering"),
        ("camembert", "TFCamembertForQuestionAnswering"),
        ("convbert", "TFConvBertForQuestionAnswering"),
        ("deberta", "TFDebertaForQuestionAnswering"),
        ("deberta-v2", "TFDebertaV2ForQuestionAnswering"),
        ("distilbert", "TFDistilBertForQuestionAnswering"),
        ("electra", "TFElectraForQuestionAnswering"),
        ("flaubert", "TFFlaubertForQuestionAnsweringSimple"),
        ("funnel", "TFFunnelForQuestionAnswering"),
        ("gptj", "TFGPTJForQuestionAnswering"),
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
        ("longformer", "TFLongformerForQuestionAnswering"),
        ("mobilebert", "TFMobileBertForQuestionAnswering"),
        ("mpnet", "TFMPNetForQuestionAnswering"),
        ("rembert", "TFRemBertForQuestionAnswering"),
        ("roberta", "TFRobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForQuestionAnswering"),
        ("roformer", "TFRoFormerForQuestionAnswering"),
        ("xlm", "TFXLMForQuestionAnsweringSimple"),
        ("xlm-roberta", "TFXLMRobertaForQuestionAnswering"),
        ("xlnet", "TFXLNetForQuestionAnsweringSimple"),
    ]
)
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict([("wav2vec2", "TFWav2Vec2ForSequenceClassification")])

TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "TFLayoutLMForQuestionAnswering"),
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
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
        ("albert", "TFAlbertForTokenClassification"),
        ("bert", "TFBertForTokenClassification"),
        ("brand-new-bert", "TFBrandNewBERTForTokenClassification"),
        ("camembert", "TFCamembertForTokenClassification"),
        ("convbert", "TFConvBertForTokenClassification"),
        ("deberta", "TFDebertaForTokenClassification"),
        ("deberta-v2", "TFDebertaV2ForTokenClassification"),
        ("distilbert", "TFDistilBertForTokenClassification"),
        ("electra", "TFElectraForTokenClassification"),
        ("esm", "TFEsmForTokenClassification"),
        ("flaubert", "TFFlaubertForTokenClassification"),
        ("funnel", "TFFunnelForTokenClassification"),
        ("layoutlm", "TFLayoutLMForTokenClassification"),
        ("layoutlmv3", "TFLayoutLMv3ForTokenClassification"),
        ("longformer", "TFLongformerForTokenClassification"),
        ("mobilebert", "TFMobileBertForTokenClassification"),
        ("mpnet", "TFMPNetForTokenClassification"),
        ("rembert", "TFRemBertForTokenClassification"),
        ("roberta", "TFRobertaForTokenClassification"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForTokenClassification"),
        ("roformer", "TFRoFormerForTokenClassification"),
        ("xlm", "TFXLMForTokenClassification"),
        ("xlm-roberta", "TFXLMRobertaForTokenClassification"),
        ("xlnet", "TFXLNetForTokenClassification"),
    ]
)

TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "TFAlbertForMultipleChoice"),
        ("bert", "TFBertForMultipleChoice"),
        ("brand-new-bert", "TFBrandNewBERTForMultipleChoice"),
        ("camembert", "TFCamembertForMultipleChoice"),
        ("convbert", "TFConvBertForMultipleChoice"),
        ("deberta-v2", "TFDebertaV2ForMultipleChoice"),
        ("distilbert", "TFDistilBertForMultipleChoice"),
        ("electra", "TFElectraForMultipleChoice"),
        ("flaubert", "TFFlaubertForMultipleChoice"),
        ("funnel", "TFFunnelForMultipleChoice"),
        ("longformer", "TFLongformerForMultipleChoice"),
        ("mobilebert", "TFMobileBertForMultipleChoice"),
        ("mpnet", "TFMPNetForMultipleChoice"),
        ("rembert", "TFRemBertForMultipleChoice"),
        ("roberta", "TFRobertaForMultipleChoice"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMultipleChoice"),
        ("roformer", "TFRoFormerForMultipleChoice"),
        ("xlm", "TFXLMForMultipleChoice"),
        ("xlm-roberta", "TFXLMRobertaForMultipleChoice"),
        ("xlnet", "TFXLNetForMultipleChoice"),
    ]
)

TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "TFBertForNextSentencePrediction"),
        ("brand-new-bert", "TFBrandNewBERTForNextSentencePrediction"),
        ("mobilebert", "TFMobileBertForNextSentencePrediction"),
    ]
)
TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "TFSamModel"),
    ]
)
TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "TFAlbertModel"),
        ("bert", "TFBertModel"),
        ("brand-new-bert", "TFBrandNewBERTModel"),
        ("convbert", "TFConvBertModel"),
        ("deberta", "TFDebertaModel"),
        ("deberta-v2", "TFDebertaV2Model"),
        ("distilbert", "TFDistilBertModel"),
        ("electra", "TFElectraModel"),
        ("flaubert", "TFFlaubertModel"),
        ("longformer", "TFLongformerModel"),
        ("mobilebert", "TFMobileBertModel"),
        ("mt5", "TFMT5EncoderModel"),
        ("rembert", "TFRemBertModel"),
        ("roberta", "TFRobertaModel"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        ("roformer", "TFRoFormerModel"),
        ("t5", "TFT5EncoderModel"),
        ("xlm", "TFXLMModel"),
        ("xlm-roberta", "TFXLMRobertaModel"),
    ]
)

TF_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_MAPPING_NAMES)
TF_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
TF_MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES)
TF_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
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
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
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
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)

TF_MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
)

TF_MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)


class TFAutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MASK_GENERATION_MAPPING


class TFAutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_TEXT_ENCODING_MAPPING


class TFAutoModel(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_MAPPING


TFAutoModel = auto_class_update(TFAutoModel)


class TFAutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


TFAutoModelForAudioClassification = auto_class_update(
    TFAutoModelForAudioClassification, head_doc="audio classification"
)


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


class TFAutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


TFAutoModelForMaskedImageModeling = auto_class_update(
    TFAutoModelForMaskedImageModeling, head_doc="masked image modeling"
)


class TFAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


TFAutoModelForImageClassification = auto_class_update(
    TFAutoModelForImageClassification, head_doc="image classification"
)


class TFAutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


TFAutoModelForZeroShotImageClassification = auto_class_update(
    TFAutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)


class TFAutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


TFAutoModelForSemanticSegmentation = auto_class_update(
    TFAutoModelForSemanticSegmentation, head_doc="semantic segmentation"
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
    TFAutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)


class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


TFAutoModelForSequenceClassification = auto_class_update(
    TFAutoModelForSequenceClassification, head_doc="sequence classification"
)


class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING


TFAutoModelForQuestionAnswering = auto_class_update(TFAutoModelForQuestionAnswering, head_doc="question answering")


class TFAutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


TFAutoModelForDocumentQuestionAnswering = auto_class_update(
    TFAutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


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
