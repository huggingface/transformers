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

print("Loading auto models.")

import warnings
from collections import OrderedDict

from transformers.utils.modeling_auto_mapping import MODEL_MAPPING_NAMES

from ...utils import logging

from .auto_factory import _BaseAutoModelClass, LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES

logger = logging.get_logger(__name__)


MODEL_MAPPING_NAMES = OrderedDict(
    [
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
        ("funnel", ("FunnelModel","FunnelBaseModel")),
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
    ]
)


MODEL_MAPPING = LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
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


MODEL_FOR_PRETRAINING_MAPPING = LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        # Model with LM heads mapping
        (RemBertConfig, RemBertForMaskedLM),
        (RoFormerConfig, RoFormerForMaskedLM),
        (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForMaskedLM),
        (Speech2TextConfig, Speech2TextForConditionalGeneration),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (M2M100Config, M2M100ForConditionalGeneration),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (T5Config, T5ForConditionalGeneration),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (MarianConfig, MarianMTModel),
        (FSMTConfig, FSMTForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
        (MegatronBertConfig, MegatronBertForCausalLM),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (RemBertConfig, RemBertForCausalLM),
        (RoFormerConfig, RoFormerForCausalLM),
        (BigBirdPegasusConfig, BigBirdPegasusForCausalLM),
        (GPTNeoConfig, GPTNeoForCausalLM),
        (BigBirdConfig, BigBirdForCausalLM),
        (CamembertConfig, CamembertForCausalLM),
        (XLMRobertaConfig, XLMRobertaForCausalLM),
        (RobertaConfig, RobertaForCausalLM),
        (BertConfig, BertLMHeadModel),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (
            XLMConfig,
            XLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, CTRLLMHeadModel),
        (ReformerConfig, ReformerModelWithLMHead),
        (BertGenerationConfig, BertGenerationDecoder),
        (XLMProphetNetConfig, XLMProphetNetForCausalLM),
        (ProphetNetConfig, ProphetNetForCausalLM),
        (BartConfig, BartForCausalLM),
        (MBartConfig, MBartForCausalLM),
        (PegasusConfig, PegasusForCausalLM),
        (MarianConfig, MarianForCausalLM),
        (BlenderbotConfig, BlenderbotForCausalLM),
        (BlenderbotSmallConfig, BlenderbotSmallForCausalLM),
        (MegatronBertConfig, MegatronBertForCausalLM),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image Classification mapping
        (ViTConfig, ViTForImageClassification),
        (DeiTConfig, (DeiTForImageClassification, DeiTForImageClassificationWithTeacher)),
        (BeitConfig, BeitForImageClassification),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (RemBertConfig, RemBertForMaskedLM),
        (RoFormerConfig, RoFormerForMaskedLM),
        (BigBirdConfig, BigBirdForMaskedLM),
        (Wav2Vec2Config, Wav2Vec2ForMaskedLM),
        (ConvBertConfig, ConvBertForMaskedLM),
        (LayoutLMConfig, LayoutLMForMaskedLM),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (MBartConfig, MBartForConditionalGeneration),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (LongformerConfig, LongformerForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (SqueezeBertConfig, SqueezeBertForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (MobileBertConfig, MobileBertForMaskedLM),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (ElectraConfig, ElectraForMaskedLM),
        (ReformerConfig, ReformerForMaskedLM),
        (FunnelConfig, FunnelForMaskedLM),
        (MPNetConfig, MPNetForMaskedLM),
        (TapasConfig, TapasForMaskedLM),
        (DebertaConfig, DebertaForMaskedLM),
        (DebertaV2Config, DebertaV2ForMaskedLM),
        (IBertConfig, IBertForMaskedLM),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING = OrderedDict(
    [
        # Model for Object Detection mapping
        (DetrConfig, DetrForObjectDetection),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
        (M2M100Config, M2M100ForConditionalGeneration),
        (LEDConfig, LEDForConditionalGeneration),
        (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration),
        (MT5Config, MT5ForConditionalGeneration),
        (T5Config, T5ForConditionalGeneration),
        (PegasusConfig, PegasusForConditionalGeneration),
        (MarianConfig, MarianMTModel),
        (MBartConfig, MBartForConditionalGeneration),
        (BlenderbotConfig, BlenderbotForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (FSMTConfig, FSMTForConditionalGeneration),
        (EncoderDecoderConfig, EncoderDecoderModel),
        (XLMProphetNetConfig, XLMProphetNetForConditionalGeneration),
        (ProphetNetConfig, ProphetNetForConditionalGeneration),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (RemBertConfig, RemBertForSequenceClassification),
        (CanineConfig, CanineForSequenceClassification),
        (RoFormerConfig, RoFormerForSequenceClassification),
        (BigBirdPegasusConfig, BigBirdPegasusForSequenceClassification),
        (BigBirdConfig, BigBirdForSequenceClassification),
        (ConvBertConfig, ConvBertForSequenceClassification),
        (LEDConfig, LEDForSequenceClassification),
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (MBartConfig, MBartForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (LongformerConfig, LongformerForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (SqueezeBertConfig, SqueezeBertForSequenceClassification),
        (LayoutLMConfig, LayoutLMForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (MegatronBertConfig, MegatronBertForSequenceClassification),
        (MobileBertConfig, MobileBertForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
        (ElectraConfig, ElectraForSequenceClassification),
        (FunnelConfig, FunnelForSequenceClassification),
        (DebertaConfig, DebertaForSequenceClassification),
        (DebertaV2Config, DebertaV2ForSequenceClassification),
        (GPT2Config, GPT2ForSequenceClassification),
        (GPTNeoConfig, GPTNeoForSequenceClassification),
        (OpenAIGPTConfig, OpenAIGPTForSequenceClassification),
        (ReformerConfig, ReformerForSequenceClassification),
        (CTRLConfig, CTRLForSequenceClassification),
        (TransfoXLConfig, TransfoXLForSequenceClassification),
        (MPNetConfig, MPNetForSequenceClassification),
        (TapasConfig, TapasForSequenceClassification),
        (IBertConfig, IBertForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (RemBertConfig, RemBertForQuestionAnswering),
        (CanineConfig, CanineForQuestionAnswering),
        (RoFormerConfig, RoFormerForQuestionAnswering),
        (BigBirdPegasusConfig, BigBirdPegasusForQuestionAnswering),
        (BigBirdConfig, BigBirdForQuestionAnswering),
        (ConvBertConfig, ConvBertForQuestionAnswering),
        (LEDConfig, LEDForQuestionAnswering),
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (CamembertConfig, CamembertForQuestionAnswering),
        (BartConfig, BartForQuestionAnswering),
        (MBartConfig, MBartForQuestionAnswering),
        (LongformerConfig, LongformerForQuestionAnswering),
        (XLMRobertaConfig, XLMRobertaForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (SqueezeBertConfig, SqueezeBertForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnsweringSimple),
        (FlaubertConfig, FlaubertForQuestionAnsweringSimple),
        (MegatronBertConfig, MegatronBertForQuestionAnswering),
        (MobileBertConfig, MobileBertForQuestionAnswering),
        (XLMConfig, XLMForQuestionAnsweringSimple),
        (ElectraConfig, ElectraForQuestionAnswering),
        (ReformerConfig, ReformerForQuestionAnswering),
        (FunnelConfig, FunnelForQuestionAnswering),
        (LxmertConfig, LxmertForQuestionAnswering),
        (MPNetConfig, MPNetForQuestionAnswering),
        (DebertaConfig, DebertaForQuestionAnswering),
        (DebertaV2Config, DebertaV2ForQuestionAnswering),
        (IBertConfig, IBertForQuestionAnswering),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Table Question Answering mapping
        (TapasConfig, TapasForQuestionAnswering),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (RemBertConfig, RemBertForTokenClassification),
        (CanineConfig, CanineForTokenClassification),
        (RoFormerConfig, RoFormerForTokenClassification),
        (BigBirdConfig, BigBirdForTokenClassification),
        (ConvBertConfig, ConvBertForTokenClassification),
        (LayoutLMConfig, LayoutLMForTokenClassification),
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (XLMConfig, XLMForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (LongformerConfig, LongformerForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (SqueezeBertConfig, SqueezeBertForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (MegatronBertConfig, MegatronBertForTokenClassification),
        (MobileBertConfig, MobileBertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
        (AlbertConfig, AlbertForTokenClassification),
        (ElectraConfig, ElectraForTokenClassification),
        (FlaubertConfig, FlaubertForTokenClassification),
        (FunnelConfig, FunnelForTokenClassification),
        (MPNetConfig, MPNetForTokenClassification),
        (DebertaConfig, DebertaForTokenClassification),
        (DebertaV2Config, DebertaV2ForTokenClassification),
        (IBertConfig, IBertForTokenClassification),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (RemBertConfig, RemBertForMultipleChoice),
        (CanineConfig, CanineForMultipleChoice),
        (RoFormerConfig, RoFormerForMultipleChoice),
        (BigBirdConfig, BigBirdForMultipleChoice),
        (ConvBertConfig, ConvBertForMultipleChoice),
        (CamembertConfig, CamembertForMultipleChoice),
        (ElectraConfig, ElectraForMultipleChoice),
        (XLMRobertaConfig, XLMRobertaForMultipleChoice),
        (LongformerConfig, LongformerForMultipleChoice),
        (RobertaConfig, RobertaForMultipleChoice),
        (SqueezeBertConfig, SqueezeBertForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (DistilBertConfig, DistilBertForMultipleChoice),
        (MegatronBertConfig, MegatronBertForMultipleChoice),
        (MobileBertConfig, MobileBertForMultipleChoice),
        (XLNetConfig, XLNetForMultipleChoice),
        (AlbertConfig, AlbertForMultipleChoice),
        (XLMConfig, XLMForMultipleChoice),
        (FlaubertConfig, FlaubertForMultipleChoice),
        (FunnelConfig, FunnelForMultipleChoice),
        (MPNetConfig, MPNetForMultipleChoice),
        (IBertConfig, IBertForMultipleChoice),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, BertForNextSentencePrediction),
        (MegatronBertConfig, MegatronBertForNextSentencePrediction),
        (MobileBertConfig, MobileBertForNextSentencePrediction),
    ]
)


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
