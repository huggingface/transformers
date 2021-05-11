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

# Add modeling imports here
from ..albert.modeling_tf_albert import (
    TFAlbertForMaskedLM,
    TFAlbertForMultipleChoice,
    TFAlbertForPreTraining,
    TFAlbertForQuestionAnswering,
    TFAlbertForSequenceClassification,
    TFAlbertForTokenClassification,
    TFAlbertModel,
)
from ..bart.modeling_tf_bart import TFBartForConditionalGeneration, TFBartModel
from ..bert.modeling_tf_bert import (
    TFBertForMaskedLM,
    TFBertForMultipleChoice,
    TFBertForNextSentencePrediction,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFBertForTokenClassification,
    TFBertLMHeadModel,
    TFBertModel,
)
from ..blenderbot.modeling_tf_blenderbot import TFBlenderbotForConditionalGeneration, TFBlenderbotModel
from ..blenderbot_small.modeling_tf_blenderbot_small import (
    TFBlenderbotSmallForConditionalGeneration,
    TFBlenderbotSmallModel,
)
from ..camembert.modeling_tf_camembert import (
    TFCamembertForMaskedLM,
    TFCamembertForMultipleChoice,
    TFCamembertForQuestionAnswering,
    TFCamembertForSequenceClassification,
    TFCamembertForTokenClassification,
    TFCamembertModel,
)
from ..convbert.modeling_tf_convbert import (
    TFConvBertForMaskedLM,
    TFConvBertForMultipleChoice,
    TFConvBertForQuestionAnswering,
    TFConvBertForSequenceClassification,
    TFConvBertForTokenClassification,
    TFConvBertModel,
)
from ..ctrl.modeling_tf_ctrl import TFCTRLForSequenceClassification, TFCTRLLMHeadModel, TFCTRLModel
from ..distilbert.modeling_tf_distilbert import (
    TFDistilBertForMaskedLM,
    TFDistilBertForMultipleChoice,
    TFDistilBertForQuestionAnswering,
    TFDistilBertForSequenceClassification,
    TFDistilBertForTokenClassification,
    TFDistilBertModel,
)
from ..dpr.modeling_tf_dpr import TFDPRQuestionEncoder
from ..electra.modeling_tf_electra import (
    TFElectraForMaskedLM,
    TFElectraForMultipleChoice,
    TFElectraForPreTraining,
    TFElectraForQuestionAnswering,
    TFElectraForSequenceClassification,
    TFElectraForTokenClassification,
    TFElectraModel,
)
from ..flaubert.modeling_tf_flaubert import (
    TFFlaubertForMultipleChoice,
    TFFlaubertForQuestionAnsweringSimple,
    TFFlaubertForSequenceClassification,
    TFFlaubertForTokenClassification,
    TFFlaubertModel,
    TFFlaubertWithLMHeadModel,
)
from ..funnel.modeling_tf_funnel import (
    TFFunnelBaseModel,
    TFFunnelForMaskedLM,
    TFFunnelForMultipleChoice,
    TFFunnelForPreTraining,
    TFFunnelForQuestionAnswering,
    TFFunnelForSequenceClassification,
    TFFunnelForTokenClassification,
    TFFunnelModel,
)
from ..gpt2.modeling_tf_gpt2 import TFGPT2ForSequenceClassification, TFGPT2LMHeadModel, TFGPT2Model
from ..layoutlm.modeling_tf_layoutlm import (
    TFLayoutLMForMaskedLM,
    TFLayoutLMForSequenceClassification,
    TFLayoutLMForTokenClassification,
    TFLayoutLMModel,
)
from ..led.modeling_tf_led import TFLEDForConditionalGeneration, TFLEDModel
from ..longformer.modeling_tf_longformer import (
    TFLongformerForMaskedLM,
    TFLongformerForMultipleChoice,
    TFLongformerForQuestionAnswering,
    TFLongformerForSequenceClassification,
    TFLongformerForTokenClassification,
    TFLongformerModel,
)
from ..lxmert.modeling_tf_lxmert import TFLxmertForPreTraining, TFLxmertModel
from ..marian.modeling_tf_marian import TFMarianModel, TFMarianMTModel
from ..mbart.modeling_tf_mbart import TFMBartForConditionalGeneration, TFMBartModel
from ..mobilebert.modeling_tf_mobilebert import (
    TFMobileBertForMaskedLM,
    TFMobileBertForMultipleChoice,
    TFMobileBertForNextSentencePrediction,
    TFMobileBertForPreTraining,
    TFMobileBertForQuestionAnswering,
    TFMobileBertForSequenceClassification,
    TFMobileBertForTokenClassification,
    TFMobileBertModel,
)
from ..mpnet.modeling_tf_mpnet import (
    TFMPNetForMaskedLM,
    TFMPNetForMultipleChoice,
    TFMPNetForQuestionAnswering,
    TFMPNetForSequenceClassification,
    TFMPNetForTokenClassification,
    TFMPNetModel,
)
from ..mt5.modeling_tf_mt5 import TFMT5ForConditionalGeneration, TFMT5Model
from ..openai.modeling_tf_openai import TFOpenAIGPTForSequenceClassification, TFOpenAIGPTLMHeadModel, TFOpenAIGPTModel
from ..pegasus.modeling_tf_pegasus import TFPegasusForConditionalGeneration, TFPegasusModel
from ..roberta.modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForMultipleChoice,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from ..t5.modeling_tf_t5 import TFT5ForConditionalGeneration, TFT5Model
from ..transfo_xl.modeling_tf_transfo_xl import (
    TFTransfoXLForSequenceClassification,
    TFTransfoXLLMHeadModel,
    TFTransfoXLModel,
)
from ..xlm.modeling_tf_xlm import (
    TFXLMForMultipleChoice,
    TFXLMForQuestionAnsweringSimple,
    TFXLMForSequenceClassification,
    TFXLMForTokenClassification,
    TFXLMModel,
    TFXLMWithLMHeadModel,
)
from ..xlm_roberta.modeling_tf_xlm_roberta import (
    TFXLMRobertaForMaskedLM,
    TFXLMRobertaForMultipleChoice,
    TFXLMRobertaForQuestionAnswering,
    TFXLMRobertaForSequenceClassification,
    TFXLMRobertaForTokenClassification,
    TFXLMRobertaModel,
)
from ..xlnet.modeling_tf_xlnet import (
    TFXLNetForMultipleChoice,
    TFXLNetForQuestionAnsweringSimple,
    TFXLNetForSequenceClassification,
    TFXLNetForTokenClassification,
    TFXLNetLMHeadModel,
    TFXLNetModel,
)
from .auto_factory import auto_class_factory
from .configuration_auto import (
    AlbertConfig,
    BartConfig,
    BertConfig,
    BlenderbotConfig,
    BlenderbotSmallConfig,
    CamembertConfig,
    ConvBertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    FlaubertConfig,
    FunnelConfig,
    GPT2Config,
    LayoutLMConfig,
    LEDConfig,
    LongformerConfig,
    LxmertConfig,
    MarianConfig,
    MBartConfig,
    MobileBertConfig,
    MPNetConfig,
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)


logger = logging.get_logger(__name__)


TF_MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (ConvBertConfig, TFConvBertModel),
        (LEDConfig, TFLEDModel),
        (LxmertConfig, TFLxmertModel),
        (MT5Config, TFMT5Model),
        (T5Config, TFT5Model),
        (DistilBertConfig, TFDistilBertModel),
        (AlbertConfig, TFAlbertModel),
        (BartConfig, TFBartModel),
        (CamembertConfig, TFCamembertModel),
        (XLMRobertaConfig, TFXLMRobertaModel),
        (LongformerConfig, TFLongformerModel),
        (RobertaConfig, TFRobertaModel),
        (LayoutLMConfig, TFLayoutLMModel),
        (BertConfig, TFBertModel),
        (OpenAIGPTConfig, TFOpenAIGPTModel),
        (GPT2Config, TFGPT2Model),
        (MobileBertConfig, TFMobileBertModel),
        (TransfoXLConfig, TFTransfoXLModel),
        (XLNetConfig, TFXLNetModel),
        (FlaubertConfig, TFFlaubertModel),
        (XLMConfig, TFXLMModel),
        (CTRLConfig, TFCTRLModel),
        (ElectraConfig, TFElectraModel),
        (FunnelConfig, (TFFunnelModel, TFFunnelBaseModel)),
        (DPRConfig, TFDPRQuestionEncoder),
        (MPNetConfig, TFMPNetModel),
        (BartConfig, TFBartModel),
        (MBartConfig, TFMBartModel),
        (MarianConfig, TFMarianModel),
        (PegasusConfig, TFPegasusModel),
        (BlenderbotConfig, TFBlenderbotModel),
        (BlenderbotSmallConfig, TFBlenderbotSmallModel),
    ]
)

TF_MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (LxmertConfig, TFLxmertForPreTraining),
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForPreTraining),
        (BartConfig, TFBartForConditionalGeneration),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (LayoutLMConfig, TFLayoutLMForMaskedLM),
        (BertConfig, TFBertForPreTraining),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (MobileBertConfig, TFMobileBertForPreTraining),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
        (ElectraConfig, TFElectraForPreTraining),
        (FunnelConfig, TFFunnelForPreTraining),
        (MPNetConfig, TFMPNetForMaskedLM),
    ]
)

TF_MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        # Model with LM heads mapping
        (ConvBertConfig, TFConvBertForMaskedLM),
        (LEDConfig, TFLEDForConditionalGeneration),
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForMaskedLM),
        (MarianConfig, TFMarianMTModel),
        (BartConfig, TFBartForConditionalGeneration),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (LongformerConfig, TFLongformerForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (LayoutLMConfig, TFLayoutLMForMaskedLM),
        (BertConfig, TFBertForMaskedLM),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (MobileBertConfig, TFMobileBertForMaskedLM),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
        (ElectraConfig, TFElectraForMaskedLM),
        (FunnelConfig, TFFunnelForMaskedLM),
        (MPNetConfig, TFMPNetForMaskedLM),
    ]
)

TF_MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (BertConfig, TFBertLMHeadModel),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (
            XLMConfig,
            TFXLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, TFCTRLLMHeadModel),
    ]
)

TF_MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (ConvBertConfig, TFConvBertForMaskedLM),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForMaskedLM),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (LongformerConfig, TFLongformerForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (LayoutLMConfig, TFLayoutLMForMaskedLM),
        (BertConfig, TFBertForMaskedLM),
        (MobileBertConfig, TFMobileBertForMaskedLM),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (ElectraConfig, TFElectraForMaskedLM),
        (FunnelConfig, TFFunnelForMaskedLM),
        (MPNetConfig, TFMPNetForMaskedLM),
    ]
)


TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        (LEDConfig, TFLEDForConditionalGeneration),
        (MT5Config, TFMT5ForConditionalGeneration),
        (T5Config, TFT5ForConditionalGeneration),
        (MarianConfig, TFMarianMTModel),
        (MBartConfig, TFMBartForConditionalGeneration),
        (PegasusConfig, TFPegasusForConditionalGeneration),
        (BlenderbotConfig, TFBlenderbotForConditionalGeneration),
        (BlenderbotSmallConfig, TFBlenderbotSmallForConditionalGeneration),
        (BartConfig, TFBartForConditionalGeneration),
    ]
)

TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (ConvBertConfig, TFConvBertForSequenceClassification),
        (DistilBertConfig, TFDistilBertForSequenceClassification),
        (AlbertConfig, TFAlbertForSequenceClassification),
        (CamembertConfig, TFCamembertForSequenceClassification),
        (XLMRobertaConfig, TFXLMRobertaForSequenceClassification),
        (LongformerConfig, TFLongformerForSequenceClassification),
        (RobertaConfig, TFRobertaForSequenceClassification),
        (LayoutLMConfig, TFLayoutLMForSequenceClassification),
        (BertConfig, TFBertForSequenceClassification),
        (XLNetConfig, TFXLNetForSequenceClassification),
        (MobileBertConfig, TFMobileBertForSequenceClassification),
        (FlaubertConfig, TFFlaubertForSequenceClassification),
        (XLMConfig, TFXLMForSequenceClassification),
        (ElectraConfig, TFElectraForSequenceClassification),
        (FunnelConfig, TFFunnelForSequenceClassification),
        (GPT2Config, TFGPT2ForSequenceClassification),
        (MPNetConfig, TFMPNetForSequenceClassification),
        (OpenAIGPTConfig, TFOpenAIGPTForSequenceClassification),
        (TransfoXLConfig, TFTransfoXLForSequenceClassification),
        (CTRLConfig, TFCTRLForSequenceClassification),
    ]
)

TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (ConvBertConfig, TFConvBertForQuestionAnswering),
        (DistilBertConfig, TFDistilBertForQuestionAnswering),
        (AlbertConfig, TFAlbertForQuestionAnswering),
        (CamembertConfig, TFCamembertForQuestionAnswering),
        (XLMRobertaConfig, TFXLMRobertaForQuestionAnswering),
        (LongformerConfig, TFLongformerForQuestionAnswering),
        (RobertaConfig, TFRobertaForQuestionAnswering),
        (BertConfig, TFBertForQuestionAnswering),
        (XLNetConfig, TFXLNetForQuestionAnsweringSimple),
        (MobileBertConfig, TFMobileBertForQuestionAnswering),
        (FlaubertConfig, TFFlaubertForQuestionAnsweringSimple),
        (XLMConfig, TFXLMForQuestionAnsweringSimple),
        (ElectraConfig, TFElectraForQuestionAnswering),
        (FunnelConfig, TFFunnelForQuestionAnswering),
        (MPNetConfig, TFMPNetForQuestionAnswering),
    ]
)

TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (ConvBertConfig, TFConvBertForTokenClassification),
        (DistilBertConfig, TFDistilBertForTokenClassification),
        (AlbertConfig, TFAlbertForTokenClassification),
        (CamembertConfig, TFCamembertForTokenClassification),
        (FlaubertConfig, TFFlaubertForTokenClassification),
        (XLMConfig, TFXLMForTokenClassification),
        (XLMRobertaConfig, TFXLMRobertaForTokenClassification),
        (LongformerConfig, TFLongformerForTokenClassification),
        (RobertaConfig, TFRobertaForTokenClassification),
        (LayoutLMConfig, TFLayoutLMForTokenClassification),
        (BertConfig, TFBertForTokenClassification),
        (MobileBertConfig, TFMobileBertForTokenClassification),
        (XLNetConfig, TFXLNetForTokenClassification),
        (ElectraConfig, TFElectraForTokenClassification),
        (FunnelConfig, TFFunnelForTokenClassification),
        (MPNetConfig, TFMPNetForTokenClassification),
    ]
)

TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (ConvBertConfig, TFConvBertForMultipleChoice),
        (CamembertConfig, TFCamembertForMultipleChoice),
        (XLMConfig, TFXLMForMultipleChoice),
        (XLMRobertaConfig, TFXLMRobertaForMultipleChoice),
        (LongformerConfig, TFLongformerForMultipleChoice),
        (RobertaConfig, TFRobertaForMultipleChoice),
        (BertConfig, TFBertForMultipleChoice),
        (DistilBertConfig, TFDistilBertForMultipleChoice),
        (MobileBertConfig, TFMobileBertForMultipleChoice),
        (XLNetConfig, TFXLNetForMultipleChoice),
        (FlaubertConfig, TFFlaubertForMultipleChoice),
        (AlbertConfig, TFAlbertForMultipleChoice),
        (ElectraConfig, TFElectraForMultipleChoice),
        (FunnelConfig, TFFunnelForMultipleChoice),
        (MPNetConfig, TFMPNetForMultipleChoice),
    ]
)

TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, TFBertForNextSentencePrediction),
        (MobileBertConfig, TFMobileBertForNextSentencePrediction),
    ]
)


TFAutoModel = auto_class_factory("TFAutoModel", TF_MODEL_MAPPING)

TFAutoModelForPreTraining = auto_class_factory(
    "TFAutoModelForPreTraining", TF_MODEL_FOR_PRETRAINING_MAPPING, head_doc="pretraining"
)

# Private on purpose, the public class will add the deprecation warnings.
_TFAutoModelWithLMHead = auto_class_factory(
    "TFAutoModelWithLMHead", TF_MODEL_WITH_LM_HEAD_MAPPING, head_doc="language modeling"
)

TFAutoModelForCausalLM = auto_class_factory(
    "TFAutoModelForCausalLM", TF_MODEL_FOR_CAUSAL_LM_MAPPING, head_doc="causal language modeling"
)

TFAutoModelForMaskedLM = auto_class_factory(
    "TFAutoModelForMaskedLM", TF_MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked language modeling"
)

TFAutoModelForSeq2SeqLM = auto_class_factory(
    "TFAutoModelForSeq2SeqLM",
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="t5-base",
)

TFAutoModelForSequenceClassification = auto_class_factory(
    "TFAutoModelForSequenceClassification",
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    head_doc="sequence classification",
)

TFAutoModelForQuestionAnswering = auto_class_factory(
    "TFAutoModelForQuestionAnswering", TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

TFAutoModelForTokenClassification = auto_class_factory(
    "TFAutoModelForTokenClassification", TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

TFAutoModelForMultipleChoice = auto_class_factory(
    "TFAutoModelForMultipleChoice", TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)

TFAutoModelForNextSentencePrediction = auto_class_factory(
    "TFAutoModelForNextSentencePrediction",
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    head_doc="next sentence prediction",
)


class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and "
            "`TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and "
            "`TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
