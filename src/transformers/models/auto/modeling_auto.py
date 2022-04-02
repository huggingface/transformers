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


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("dpt", "DPTModel"),
        ("decision_transformer", "DecisionTransformerModel"),
        ("glpn", "GLPNModel"),
        ("maskformer", "MaskFormerModel"),
        ("decision_transformer", "DecisionTransformerModel"),
        ("decision_transformer_gpt2", "DecisionTransformerGPT2Model"),
        ("poolformer", "PoolFormerModel"),
        ("convnext", "ConvNextModel"),
        ("van", "VanModel"),
        ("resnet", "ResNetModel"),
        ("yoso", "YosoModel"),
        ("swin", "SwinModel"),
        ("vilt", "ViltModel"),
        ("vit_mae", "ViTMAEModel"),
        ("nystromformer", "NystromformerModel"),
        ("xglm", "XGLMModel"),
        ("imagegpt", "ImageGPTModel"),
        ("qdqbert", "QDQBertModel"),
        ("fnet", "FNetModel"),
        ("segformer", "SegformerModel"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderModel"),
        ("perceiver", "PerceiverModel"),
        ("gptj", "GPTJModel"),
        ("layoutlmv2", "LayoutLMv2Model"),
        ("plbart", "PLBartModel"),
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
        ("unispeech-sat", "UniSpeechSatModel"),
        ("wavlm", "WavLMModel"),
        ("unispeech", "UniSpeechModel"),
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
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("bart", "BartModel"),
        ("longformer", "LongformerModel"),
        ("roberta", "RobertaModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
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
        ("sew", "SEWModel"),
        ("sew-d", "SEWDModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("vit_mae", "ViTMAEForPreTraining"),
        ("fnet", "FNetForPreTraining"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("retribert", "RetriBertModel"),
        ("t5", "T5ForConditionalGeneration"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("albert", "AlbertForPreTraining"),
        ("camembert", "CamembertForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
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
        ("unispeech-sat", "UniSpeechSatForPreTraining"),
        ("unispeech", "UniSpeechForPreTraining"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("yoso", "YosoForMaskedLM"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("qdqbert", "QDQBertForMaskedLM"),
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
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("marian", "MarianMTModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
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
        ("xglm", "XGLMForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("rembert", "RemBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("big_bird", "BigBirdForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xlm", "XLMWithLMHeadModel"),
        ("electra", "ElectraForCausalLM"),
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
        ("data2vec-text", "Data2VecTextForCausalLM"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("vit", "ViTForMaskedImageModeling"),
        ("deit", "DeiTForMaskedImageModeling"),
        ("swin", "SwinForMaskedImageModeling"),
    ]
)


MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    # Model for Causal Image Modeling mapping
    [
        ("imagegpt", "ImageGPTForCausalImageModeling"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("vit", "ViTForImageClassification"),
        ("deit", ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher")),
        ("beit", "BeitForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("swin", "SwinForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("van", "VanForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("poolformer", "PoolFormerForImageClassification"),
    ]
)

MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Do not add new models here, this class will be deprecated in the future.
        # Model for Image Segmentation mapping
        ("detr", "DetrForSegmentation"),
    ]
)

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        ("beit", "BeitForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
    ]
)

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Instance Segmentation mapping
        ("maskformer", "MaskFormerForInstanceSegmentation"),
    ]
)

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("yoso", "YosoForMaskedLM"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("perceiver", "PerceiverForMaskedLM"),
        ("qdqbert", "QDQBertForMaskedLM"),
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
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
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
        ("plbart", "PLBartForConditionalGeneration"),
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
        ("yoso", "YosoForSequenceClassification"),
        ("nystromformer", "NystromformerForSequenceClassification"),
        ("plbart", "PLBartForSequenceClassification"),
        ("perceiver", "PerceiverForSequenceClassification"),
        ("qdqbert", "QDQBertForSequenceClassification"),
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
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("data2vec-text", "Data2VecTextForSequenceClassification"),
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
        ("yoso", "YosoForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("qdqbert", "QDQBertForQuestionAnswering"),
        ("fnet", "FNetForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
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
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
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
        ("data2vec-text", "Data2VecTextForQuestionAnswering"),
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
        ("yoso", "YosoForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("qdqbert", "QDQBertForTokenClassification"),
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
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
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
        ("data2vec-text", "Data2VecTextForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("yoso", "YosoForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("qdqbert", "QDQBertForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("data2vec-text", "Data2VecTextForMultipleChoice"),
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
        ("qdqbert", "QDQBertForNextSentencePrediction"),
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
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("hubert", "HubertForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("wavlm", "WavLMForCTC"),
        ("data2vec-audio", "Data2VecAudioForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wavlm", "WavLMForAudioFrameClassification"),
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wavlm", "WavLMForXVector"),
        ("data2vec-audio", "Data2VecAudioForXVector"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
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
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)


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


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")


class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


AutoModelForSemanticSegmentation = auto_class_update(
    AutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


AutoModelForInstanceSegmentation = auto_class_update(
    AutoModelForInstanceSegmentation, head_doc="instance segmentation"
)


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)


class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")


class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")


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
