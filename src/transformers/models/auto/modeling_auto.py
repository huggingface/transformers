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
        ("albert", "AlbertModel"),
        ("bart", "BartModel"),
        ("beit", "BeitModel"),
        ("bert", "BertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("big_bird", "BigBirdModel"),
        ("bigbird_pegasus", "BigBirdPegasusModel"),
        ("blenderbot", "BlenderbotModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("bloom", "BloomModel"),
        ("camembert", "CamembertModel"),
        ("canine", "CanineModel"),
        ("clip", "CLIPModel"),
        ("codegen", "CodeGenModel"),
        ("convbert", "ConvBertModel"),
        ("convnext", "ConvNextModel"),
        ("ctrl", "CTRLModel"),
        ("cvt", "CvtModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("decision_transformer", "DecisionTransformerModel"),
        ("decision_transformer_gpt2", "DecisionTransformerGPT2Model"),
        ("deit", "DeiTModel"),
        ("detr", "DetrModel"),
        ("distilbert", "DistilBertModel"),
        ("dpr", "DPRQuestionEncoder"),
        ("dpt", "DPTModel"),
        ("electra", "ElectraModel"),
        ("flaubert", "FlaubertModel"),
        ("flava", "FlavaModel"),
        ("fnet", "FNetModel"),
        ("fsmt", "FSMTModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("glpn", "GLPNModel"),
        ("gpt2", "GPT2Model"),
        ("gpt_neo", "GPTNeoModel"),
        ("gpt_neox", "GPTNeoXModel"),
        ("gptj", "GPTJModel"),
        ("groupvit", "GroupViTModel"),
        ("hubert", "HubertModel"),
        ("ibert", "IBertModel"),
        ("imagegpt", "ImageGPTModel"),
        ("layoutlm", "LayoutLMModel"),
        ("layoutlmv2", "LayoutLMv2Model"),
        ("layoutlmv3", "LayoutLMv3Model"),
        ("led", "LEDModel"),
        ("levit", "LevitModel"),
        ("longformer", "LongformerModel"),
        ("longt5", "LongT5Model"),
        ("luke", "LukeModel"),
        ("lxmert", "LxmertModel"),
        ("m2m_100", "M2M100Model"),
        ("marian", "MarianModel"),
        ("maskformer", "MaskFormerModel"),
        ("mbart", "MBartModel"),
        ("mctct", "MCTCTModel"),
        ("megatron-bert", "MegatronBertModel"),
        ("mobilebert", "MobileBertModel"),
        ("mobilevit", "MobileViTModel"),
        ("mpnet", "MPNetModel"),
        ("mt5", "MT5Model"),
        ("mvp", "MvpModel"),
        ("nezha", "NezhaModel"),
        ("nllb", "M2M100Model"),
        ("nystromformer", "NystromformerModel"),
        ("openai-gpt", "OpenAIGPTModel"),
        ("opt", "OPTModel"),
        ("owlvit", "OwlViTModel"),
        ("pegasus", "PegasusModel"),
        ("perceiver", "PerceiverModel"),
        ("plbart", "PLBartModel"),
        ("poolformer", "PoolFormerModel"),
        ("prophetnet", "ProphetNetModel"),
        ("qdqbert", "QDQBertModel"),
        ("reformer", "ReformerModel"),
        ("regnet", "RegNetModel"),
        ("rembert", "RemBertModel"),
        ("resnet", "ResNetModel"),
        ("retribert", "RetriBertModel"),
        ("roberta", "RobertaModel"),
        ("roformer", "RoFormerModel"),
        ("segformer", "SegformerModel"),
        ("sew", "SEWModel"),
        ("sew-d", "SEWDModel"),
        ("speech_to_text", "Speech2TextModel"),
        ("splinter", "SplinterModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("swin", "SwinModel"),
        ("t5", "T5Model"),
        ("tapas", "TapasModel"),
        ("trajectory_transformer", "TrajectoryTransformerModel"),
        ("transfo-xl", "TransfoXLModel"),
        ("unispeech", "UniSpeechModel"),
        ("unispeech-sat", "UniSpeechSatModel"),
        ("van", "VanModel"),
        ("vilt", "ViltModel"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderModel"),
        ("visual_bert", "VisualBertModel"),
        ("vit", "ViTModel"),
        ("vit_mae", "ViTMAEModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerModel"),
        ("wavlm", "WavLMModel"),
        ("xglm", "XGLMModel"),
        ("xlm", "XLMModel"),
        ("xlm-prophetnet", "XLMProphetNetModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("xlnet", "XLNetModel"),
        ("yolos", "YolosModel"),
        ("yoso", "YosoModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForPreTraining"),
        ("big_bird", "BigBirdForPreTraining"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForPreTraining"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("flava", "FlavaForPreTraining"),
        ("fnet", "FNetForPreTraining"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForPreTraining"),
        ("gpt2", "GPT2LMHeadModel"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("lxmert", "LxmertForPreTraining"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForPreTraining"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("retribert", "RetriBertModel"),
        ("roberta", "RobertaForMaskedLM"),
        ("splinter", "SplinterForPreTraining"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("unispeech", "UniSpeechForPreTraining"),
        ("unispeech-sat", "UniSpeechSatForPreTraining"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("vit_mae", "ViTMAEForPreTraining"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForPreTraining"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("fnet", "FNetForMaskedLM"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForMaskedLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("led", "LEDForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForMaskedLM"),
        ("nllb", "M2M100ForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("qdqbert", "QDQBertForMaskedLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),
        ("swin", "SwinForMaskedImageModeling"),
        ("vit", "ViTForMaskedImageModeling"),
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
        ("beit", "BeitForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("cvt", "CvtForImageClassification"),
        ("data2vec-vision", "Data2VecVisionForImageClassification"),
        ("deit", ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher")),
        ("imagegpt", "ImageGPTForImageClassification"),
        ("levit", ("LevitForImageClassification", "LevitForImageClassificationWithTeacher")),
        ("mobilevit", "MobileViTForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("regnet", "RegNetForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("swin", "SwinForImageClassification"),
        ("van", "VanForImageClassification"),
        ("vit", "ViTForImageClassification"),
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
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
        ("mobilevit", "MobileViTForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
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
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("megatron-bert", "MegatronBertForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForMaskedLM"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("perceiver", "PerceiverForMaskedLM"),
        ("qdqbert", "QDQBertForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("detr", "DetrForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb", "M2M100ForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
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
        ("albert", "AlbertForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("big_bird", "BigBirdForSequenceClassification"),
        ("bigbird_pegasus", "BigBirdPegasusForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("data2vec-text", "Data2VecTextForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("electra", "ElectraForSequenceClassification"),
        ("flaubert", "FlaubertForSequenceClassification"),
        ("fnet", "FNetForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("gpt2", "GPT2ForSequenceClassification"),
        ("gpt_neo", "GPTNeoForSequenceClassification"),
        ("gptj", "GPTJForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
        ("layoutlm", "LayoutLMForSequenceClassification"),
        ("layoutlmv2", "LayoutLMv2ForSequenceClassification"),
        ("layoutlmv3", "LayoutLMv3ForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("mpnet", "MPNetForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("nezha", "NezhaForSequenceClassification"),
        ("nystromformer", "NystromformerForSequenceClassification"),
        ("openai-gpt", "OpenAIGPTForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("perceiver", "PerceiverForSequenceClassification"),
        ("plbart", "PLBartForSequenceClassification"),
        ("qdqbert", "QDQBertForSequenceClassification"),
        ("reformer", "ReformerForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("roformer", "RoFormerForSequenceClassification"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("transfo-xl", "TransfoXLForSequenceClassification"),
        ("xlm", "XLMForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("xlnet", "XLNetForSequenceClassification"),
        ("yoso", "YosoForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("big_bird", "BigBirdForQuestionAnswering"),
        ("bigbird_pegasus", "BigBirdPegasusForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("data2vec-text", "Data2VecTextForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("electra", "ElectraForQuestionAnswering"),
        ("flaubert", "FlaubertForQuestionAnsweringSimple"),
        ("fnet", "FNetForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("lxmert", "LxmertForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("nezha", "NezhaForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("qdqbert", "QDQBertForQuestionAnswering"),
        ("reformer", "ReformerForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("roformer", "RoFormerForQuestionAnswering"),
        ("splinter", "SplinterForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("xlm", "XLMForQuestionAnsweringSimple"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
        ("xlnet", "XLNetForQuestionAnsweringSimple"),
        ("yoso", "YosoForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("big_bird", "BigBirdForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("data2vec-text", "Data2VecTextForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("flaubert", "FlaubertForTokenClassification"),
        ("fnet", "FNetForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("gpt2", "GPT2ForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
        ("layoutlm", "LayoutLMForTokenClassification"),
        ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("nezha", "NezhaForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("qdqbert", "QDQBertForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("xlm", "XLMForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        ("xlnet", "XLNetForTokenClassification"),
        ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "AlbertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("data2vec-text", "Data2VecTextForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("flaubert", "FlaubertForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("nezha", "NezhaForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("qdqbert", "QDQBertForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("yoso", "YosoForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("fnet", "FNetForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
        ("nezha", "NezhaForNextSentencePrediction"),
        ("qdqbert", "QDQBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("data2vec-audio", "Data2VecAudioForCTC"),
        ("hubert", "HubertForCTC"),
        ("mctct", "MCTCTForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
        ("wavlm", "WavLMForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),
        ("wavlm", "WavLMForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForXVector"),
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
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
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
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


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa",
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
