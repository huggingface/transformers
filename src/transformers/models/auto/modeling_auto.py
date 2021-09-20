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
""" Auto Model class. """

import warnings
from collections import OrderedDict

from ...utils import logging
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("fnet", "FNetModel"),
        ("gptj", "GPTJModel"),
        ("layoutlmv2", "LayoutLMv2Model"),
        ("beit", "BeitModel"),
        ("rembert", "RemBertModel"),
        ("visual_bert", "VisualBertModel"),
        ("canine", "CanineModel"),
        ("roformer", "RoFormerModel"),
        ("clip", "CLIPModel"),
        ("bigbird_pegasus", "BigBirdPegasusModel"),
        ("deit", "DeiTModel"),
        ("luke", "LukeModel"),
        ("detr", "DetrModel"),
        ("gpt_neo", "GPTNeoModel"),
        ("big_bird", "BigBirdModel"),
        ("speech_to_text", "Speech2TextModel"),
        ("vit", "ViTModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("hubert", "HubertModel"),
        ("m2m_100", "M2M100Model"),
        ("convbert", "ConvBertModel"),
        ("led", "LEDModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("retribert", "RetriBertModel"),
        ("mt5", "MT5Model"),
        ("t5", "T5Model"),
        ("pegasus", "PegasusModel"),
        ("marian", "MarianModel"),
        ("mbart", "MBartModel"),
        ("blenderbot", "BlenderbotModel"),
        ("distilbert", "DistilBertModel"),
        ("albert", "AlbertModel"),
        ("camembert", "CamembertModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("bart", "BartModel"),
        ("longformer", "LongformerModel"),
        ("roberta", "RobertaModel"),
        ("layoutlm", "LayoutLMModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("bert", "BertModel"),
        ("openai-gpt", "OpenAIGPTModel"),
        ("gpt2", "GPT2Model"),
        ("megatron-bert", "MegatronBertModel"),
        ("mobilebert", "MobileBertModel"),
        ("transfo-xl", "TransfoXLModel"),
        ("xlnet", "XLNetModel"),
        ("flaubert", "FlaubertModel"),
        ("fsmt", "FSMTModel"),
        ("xlm", "XLMModel"),
        ("ctrl", "CTRLModel"),
        ("electra", "ElectraModel"),
        ("reformer", "ReformerModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("lxmert", "LxmertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("dpr", "DPRQuestionEncoder"),
        ("xlm-prophetnet", "XLMProphetNetModel"),
        ("prophetnet", "ProphetNetModel"),
        ("mpnet", "MPNetModel"),
        ("tapas", "TapasModel"),
        ("ibert", "IBertModel"),
        ("splinter", "SplinterModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("fnet", "FNetForPreTraining"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("retribert", "RetriBertModel"),
        ("t5", "T5ForConditionalGeneration"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("albert", "AlbertForPreTraining"),
        ("camembert", "CamembertForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("bert", "BertForPreTraining"),
        ("big_bird", "BigBirdForPreTraining"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("xlnet", "XLNetLMHeadModel"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("xlm", "XLMWithLMHeadModel"),
        ("ctrl", "CTRLLMHeadModel"),
        ("electra", "ElectraForPreTraining"),
        ("lxmert", "LxmertForPreTraining"),
        ("funnel", "FunnelForPreTraining"),
        ("mpnet", "MPNetForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("fnet", "FNetForMaskedLM"),
        ("gptj", "GPTJForCausalLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("convbert", "ConvBertForMaskedLM"),
        ("led", "LEDForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("albert", "AlbertForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("marian", "MarianMTModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("xlnet", "XLNetLMHeadModel"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("xlm", "XLMWithLMHeadModel"),
        ("ctrl", "CTRLLMHeadModel"),
        ("electra", "ElectraForMaskedLM"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("reformer", "ReformerModelWithLMHead"),
        ("funnel", "FunnelForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("gptj", "GPTJForCausalLM"),
        ("rembert", "RemBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("big_bird", "BigBirdForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xlm", "XLMWithLMHeadModel"),
        ("ctrl", "CTRLLMHeadModel"),
        ("reformer", "ReformerModelWithLMHead"),
        ("bert-generation", "BertGenerationDecoder"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("vit", "ViTForImageClassification"),
        ("deit", ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher")),
        ("beit", "BeitForImageClassification"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("fnet", "FNetForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("mbart", "MBartForConditionalGeneration"),
        ("camembert", "CamembertForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("megatron-bert", "MegatronBertForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("xlm", "XLMWithLMHeadModel"),
        ("electra", "ElectraForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("detr", "DetrForObjectDetection"),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mbart", "MBartForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("fnet", "FNetForSequenceClassification"),
        ("gptj", "GPTJForSequenceClassification"),
        ("layoutlmv2", "LayoutLMv2ForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("roformer", "RoFormerForSequenceClassification"),
        ("bigbird_pegasus", "BigBirdPegasusForSequenceClassification"),
        ("big_bird", "BigBirdForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("albert", "AlbertForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("layoutlm", "LayoutLMForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("xlnet", "XLNetForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("flaubert", "FlaubertForSequenceClassification"),
        ("xlm", "XLMForSequenceClassification"),
        ("electra", "ElectraForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("gpt2", "GPT2ForSequenceClassification"),
        ("gpt_neo", "GPTNeoForSequenceClassification"),
        ("openai-gpt", "OpenAIGPTForSequenceClassification"),
        ("reformer", "ReformerForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("transfo-xl", "TransfoXLForSequenceClassification"),
        ("mpnet", "MPNetForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("fnet", "FNetForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("roformer", "RoFormerForQuestionAnswering"),
        ("bigbird_pegasus", "BigBirdPegasusForQuestionAnswering"),
        ("big_bird", "BigBirdForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("albert", "AlbertForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("xlnet", "XLNetForQuestionAnsweringSimple"),
        ("flaubert", "FlaubertForQuestionAnsweringSimple"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("xlm", "XLMForQuestionAnsweringSimple"),
        ("electra", "ElectraForQuestionAnswering"),
        ("reformer", "ReformerForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("lxmert", "LxmertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("splinter", "SplinterForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("fnet", "FNetForTokenClassification"),
        ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("big_bird", "BigBirdForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("layoutlm", "LayoutLMForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("flaubert", "FlaubertForTokenClassification"),
        ("xlm", "XLMForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("xlnet", "XLNetForTokenClassification"),
        ("albert", "AlbertForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("gpt2", "GPT2ForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("fnet", "FNetForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("albert", "AlbertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("flaubert", "FlaubertForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("fnet", "FNetForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("hubert", "HubertForCTC"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeing"
)


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
