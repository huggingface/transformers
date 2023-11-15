# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.35.2"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "audio_utils": [],
    "benchmark": [],
    "commands": [],
    "configuration_utils": ["PretrainedConfig"],
    "convert_graph_to_onnx": [],
    "convert_slow_tokenizers_checkpoints_to_fast": [],
    "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "data.data_collator": [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "deepspeed": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": ["GenerationConfig", "TextIteratorStreamer", "TextStreamer"],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "modelcard": ["ModelCard"],
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    "models": [],
    # Models
    "models.albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig"],
    "models.align": [
        "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "models.altclip": [
        "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "models.audio_spectrogram_transformer": [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ASTConfig",
    ],
    "models.auto": [
        "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    "models.autoformer": [
        "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AutoformerConfig",
    ],
    "models.bark": [
        "BarkCoarseConfig",
        "BarkConfig",
        "BarkFineConfig",
        "BarkProcessor",
        "BarkSemanticConfig",
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],
    "models.barthez": [],
    "models.bartpho": [],
    "models.beit": ["BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BeitConfig"],
    "models.bert": [
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"],
    "models.bertweet": ["BertweetTokenizer"],
    "models.big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig"],
    "models.bigbird_pegasus": [
        "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BigBirdPegasusConfig",
    ],
    "models.biogpt": ["BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BioGptConfig", "BioGptTokenizer"],
    "models.bit": ["BIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BitConfig"],
    "models.blenderbot": ["BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotConfig", "BlenderbotTokenizer"],
    "models.blenderbot_small": [
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.blip": [
        "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlipConfig",
        "BlipProcessor",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "models.blip_2": [
        "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Blip2Config",
        "Blip2Processor",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "models.bloom": ["BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP", "BloomConfig"],
    "models.bridgetower": [
        "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BridgeTowerConfig",
        "BridgeTowerProcessor",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "models.bros": ["BROS_PRETRAINED_CONFIG_ARCHIVE_MAP", "BrosConfig", "BrosProcessor"],
    "models.byt5": ["ByT5Tokenizer"],
    "models.camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"],
    "models.canine": ["CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP", "CanineConfig", "CanineTokenizer"],
    "models.chinese_clip": [
        "CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ChineseCLIPConfig",
        "ChineseCLIPProcessor",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "models.clap": [
        "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ClapAudioConfig",
        "ClapConfig",
        "ClapProcessor",
        "ClapTextConfig",
    ],
    "models.clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.code_llama": [],
    "models.codegen": ["CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP", "CodeGenConfig", "CodeGenTokenizer"],
    "models.conditional_detr": ["CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConditionalDetrConfig"],
    "models.convbert": ["CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvBertConfig", "ConvBertTokenizer"],
    "models.convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig"],
    "models.convnextv2": ["CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextV2Config"],
    "models.cpm": [],
    "models.cpmant": ["CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CpmAntConfig", "CpmAntTokenizer"],
    "models.ctrl": ["CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CTRLConfig", "CTRLTokenizer"],
    "models.cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"],
    "models.data2vec": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig", "DebertaTokenizer"],
    "models.deberta_v2": ["DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaV2Config"],
    "models.decision_transformer": ["DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "DecisionTransformerConfig"],
    "models.deformable_detr": ["DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeformableDetrConfig"],
    "models.deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.mctct": [
        "MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.open_llama": ["OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenLlamaConfig"],
    "models.deprecated.retribert": [
        "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RetriBertConfig",
        "RetriBertTokenizer",
    ],
    "models.deprecated.tapex": ["TapexTokenizer"],
    "models.deprecated.trajectory_transformer": [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TrajectoryTransformerConfig",
    ],
    "models.deprecated.van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"],
    "models.deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],
    "models.detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig"],
    "models.dialogpt": [],
    "models.dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"],
    "models.dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config"],
    "models.distilbert": ["DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DistilBertConfig", "DistilBertTokenizer"],
    "models.dit": [],
    "models.donut": ["DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "DonutProcessor", "DonutSwinConfig"],
    "models.dpr": [
        "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.dpt": ["DPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPTConfig"],
    "models.efficientformer": ["EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "EfficientFormerConfig"],
    "models.efficientnet": ["EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "EfficientNetConfig"],
    "models.electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraTokenizer"],
    "models.encodec": [
        "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.ernie": [
        "ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ErnieConfig",
    ],
    "models.ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
    "models.esm": ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig", "EsmTokenizer"],
    "models.falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],
    "models.flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    "models.fnet": ["FNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"],
    "models.focalnet": ["FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FocalNetConfig"],
    "models.fsmt": ["FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FSMTConfig", "FSMTTokenizer"],
    "models.funnel": ["FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP", "FunnelConfig", "FunnelTokenizer"],
    "models.fuyu": ["FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP", "FuyuConfig"],
    "models.git": ["GIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "GitConfig", "GitProcessor", "GitVisionConfig"],
    "models.glpn": ["GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP", "GLPNConfig"],
    "models.gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2Tokenizer"],
    "models.gpt_bigcode": ["GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTBigCodeConfig"],
    "models.gpt_neo": ["GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoConfig"],
    "models.gpt_neox": ["GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXConfig"],
    "models.gpt_neox_japanese": ["GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXJapaneseConfig"],
    "models.gpt_sw3": [],
    "models.gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig"],
    "models.gptsan_japanese": [
        "GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    "models.graphormer": ["GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "GraphormerConfig"],
    "models.groupvit": [
        "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    "models.herbert": ["HerbertTokenizer"],
    "models.hubert": ["HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "HubertConfig"],
    "models.ibert": ["IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "IBertConfig"],
    "models.idefics": [
        "IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "IdeficsConfig",
    ],
    "models.imagegpt": ["IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ImageGPTConfig"],
    "models.informer": ["INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "InformerConfig"],
    "models.instructblip": [
        "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "models.jukebox": [
        "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxTokenizer",
        "JukeboxVQVAEConfig",
    ],
    "models.kosmos2": [
        "KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Kosmos2Config",
        "Kosmos2Processor",
    ],
    "models.layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMTokenizer"],
    "models.layoutlmv2": [
        "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LayoutLMv2Config",
        "LayoutLMv2FeatureExtractor",
        "LayoutLMv2ImageProcessor",
        "LayoutLMv2Processor",
        "LayoutLMv2Tokenizer",
    ],
    "models.layoutlmv3": [
        "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LayoutLMv3Config",
        "LayoutLMv3FeatureExtractor",
        "LayoutLMv3ImageProcessor",
        "LayoutLMv3Processor",
        "LayoutLMv3Tokenizer",
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],
    "models.led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig", "LEDTokenizer"],
    "models.levit": ["LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LevitConfig"],
    "models.lilt": ["LILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LiltConfig"],
    "models.llama": ["LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlamaConfig"],
    "models.longformer": ["LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongformerConfig", "LongformerTokenizer"],
    "models.longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongT5Config"],
    "models.luke": ["LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP", "LukeConfig", "LukeTokenizer"],
    "models.lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig", "LxmertTokenizer"],
    "models.m2m_100": ["M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP", "M2M100Config"],
    "models.marian": ["MarianConfig"],
    "models.markuplm": [
        "MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MarkupLMConfig",
        "MarkupLMFeatureExtractor",
        "MarkupLMProcessor",
        "MarkupLMTokenizer",
    ],
    "models.mask2former": [
        "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Mask2FormerConfig",
    ],
    "models.maskformer": ["MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "MaskFormerConfig", "MaskFormerSwinConfig"],
    "models.mbart": ["MBartConfig"],
    "models.mbart50": [],
    "models.mega": ["MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MegaConfig"],
    "models.megatron_bert": ["MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MegatronBertConfig"],
    "models.megatron_gpt2": [],
    "models.mgp_str": ["MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP", "MgpstrConfig", "MgpstrProcessor", "MgpstrTokenizer"],
    "models.mistral": ["MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MistralConfig"],
    "models.mluke": [],
    "models.mobilebert": ["MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileBertConfig", "MobileBertTokenizer"],
    "models.mobilenet_v1": ["MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileNetV1Config"],
    "models.mobilenet_v2": ["MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileNetV2Config"],
    "models.mobilevit": ["MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileViTConfig"],
    "models.mobilevitv2": ["MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileViTV2Config"],
    "models.mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig", "MPNetTokenizer"],
    "models.mpt": ["MPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MptConfig"],
    "models.mra": ["MRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MraConfig"],
    "models.mt5": ["MT5Config"],
    "models.musicgen": [
        "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],
    "models.nat": ["NAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "NatConfig"],
    "models.nezha": ["NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP", "NezhaConfig"],
    "models.nllb": [],
    "models.nllb_moe": ["NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP", "NllbMoeConfig"],
    "models.nougat": ["NougatProcessor"],
    "models.nystromformer": [
        "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "NystromformerConfig",
    ],
    "models.oneformer": ["ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "OneFormerConfig", "OneFormerProcessor"],
    "models.openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig", "OpenAIGPTTokenizer"],
    "models.opt": ["OPTConfig"],
    "models.owlv2": [
        "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Owlv2Config",
        "Owlv2Processor",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "models.owlvit": [
        "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "OwlViTConfig",
        "OwlViTProcessor",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "models.pegasus": ["PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusConfig", "PegasusTokenizer"],
    "models.pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
    "models.perceiver": ["PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP", "PerceiverConfig", "PerceiverTokenizer"],
    "models.persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"],
    "models.poolformer": ["POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "PoolFormerConfig"],
    "models.pop2piano": [
        "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pop2PianoConfig",
    ],
    "models.prophetnet": ["PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ProphetNetConfig", "ProphetNetTokenizer"],
    "models.pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig"],
    "models.qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.realm": ["REALM_PRETRAINED_CONFIG_ARCHIVE_MAP", "RealmConfig", "RealmTokenizer"],
    "models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],
    "models.regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"],
    "models.rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig"],
    "models.resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig"],
    "models.roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaTokenizer"],
    "models.roberta_prelayernorm": ["ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaPreLayerNormConfig"],
    "models.roc_bert": ["ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoCBertConfig", "RoCBertTokenizer"],
    "models.roformer": ["ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoFormerConfig", "RoFormerTokenizer"],
    "models.rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig"],
    "models.sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "models.seamless_m4t": [
        "SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SeamlessM4TConfig",
        "SeamlessM4TFeatureExtractor",
        "SeamlessM4TProcessor",
    ],
    "models.segformer": ["SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SegformerConfig"],
    "models.sew": ["SEW_PRETRAINED_CONFIG_ARCHIVE_MAP", "SEWConfig"],
    "models.sew_d": ["SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP", "SEWDConfig"],
    "models.speech_encoder_decoder": ["SpeechEncoderDecoderConfig"],
    "models.speech_to_text": [
        "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Speech2TextConfig",
        "Speech2TextProcessor",
    ],
    "models.speech_to_text_2": [
        "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Speech2Text2Config",
        "Speech2Text2Processor",
        "Speech2Text2Tokenizer",
    ],
    "models.speecht5": [
        "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP",
        "SpeechT5Config",
        "SpeechT5FeatureExtractor",
        "SpeechT5HifiGanConfig",
        "SpeechT5Processor",
    ],
    "models.splinter": ["SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SplinterConfig", "SplinterTokenizer"],
    "models.squeezebert": ["SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "SqueezeBertConfig", "SqueezeBertTokenizer"],
    "models.swiftformer": ["SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwiftFormerConfig"],
    "models.swin": ["SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwinConfig"],
    "models.swin2sr": ["SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swin2SRConfig"],
    "models.swinv2": ["SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swinv2Config"],
    "models.switch_transformers": ["SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwitchTransformersConfig"],
    "models.t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config"],
    "models.table_transformer": ["TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TableTransformerConfig"],
    "models.tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig", "TapasTokenizer"],
    "models.time_series_transformer": [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TimeSeriesTransformerConfig",
    ],
    "models.timesformer": ["TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TimesformerConfig"],
    "models.timm_backbone": ["TimmBackboneConfig"],
    "models.transfo_xl": [
        "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.trocr": [
        "TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    "models.tvlt": [
        "TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    "models.umt5": ["UMT5Config"],
    "models.unispeech": [
        "UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UniSpeechConfig",
    ],
    "models.unispeech_sat": [
        "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UniSpeechSatConfig",
    ],
    "models.upernet": ["UperNetConfig"],
    "models.videomae": ["VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VideoMAEConfig"],
    "models.vilt": [
        "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    "models.vision_text_dual_encoder": ["VisionTextDualEncoderConfig", "VisionTextDualEncoderProcessor"],
    "models.visual_bert": ["VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "VisualBertConfig"],
    "models.vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],
    "models.vit_hybrid": ["VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTHybridConfig"],
    "models.vit_mae": ["VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMAEConfig"],
    "models.vit_msn": ["VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMSNConfig"],
    "models.vitdet": ["VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitDetConfig"],
    "models.vitmatte": ["VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitMatteConfig"],
    "models.vits": [
        "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VitsConfig",
        "VitsTokenizer",
    ],
    "models.vivit": [
        "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VivitConfig",
    ],
    "models.wav2vec2": [
        "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_conformer": [
        "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2ConformerConfig",
    ],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],
    "models.wavlm": [
        "WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "WavLMConfig",
    ],
    "models.whisper": [
        "WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "WhisperConfig",
        "WhisperFeatureExtractor",
        "WhisperProcessor",
        "WhisperTokenizer",
    ],
    "models.x_clip": [
        "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XCLIPConfig",
        "XCLIPProcessor",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "models.xglm": ["XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XGLMConfig"],
    "models.xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMTokenizer"],
    "models.xlm_prophetnet": ["XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMProphetNetConfig"],
    "models.xlm_roberta": ["XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMRobertaConfig"],
    "models.xlm_roberta_xl": ["XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMRobertaXLConfig"],
    "models.xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],
    "models.xmod": ["XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP", "XmodConfig"],
    "models.yolos": ["YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP", "YolosConfig"],
    "models.yoso": ["YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP", "YosoConfig"],
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "Conversation",
        "ConversationalPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageSegmentationPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "MaskGenerationPipeline",
        "NerPipeline",
        "ObjectDetectionPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TextToAudioPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "VideoClassificationPipeline",
        "VisualQuestionAnsweringPipeline",
        "ZeroShotAudioClassificationPipeline",
        "ZeroShotClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "pipeline",
    ],
    "processing_utils": ["ProcessorMixin"],
    "testing_utils": [],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "tools": [
        "Agent",
        "AzureOpenAiAgent",
        "HfAgent",
        "LocalAgent",
        "OpenAiAgent",
        "PipelineTool",
        "RemoteTool",
        "Tool",
        "launch_gradio_demo",
        "load_tool",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": ["EvalPrediction", "IntervalStrategy", "SchedulerType", "enable_full_determinism", "set_seed"],
    "training_args": ["TrainingArguments"],
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
    "training_args_tf": ["TFTrainingArguments"],
    "utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "is_apex_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_decord_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_safetensors_available",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_speech_available",
        "is_tensorflow_text_available",
        "is_tf_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torch_tpu_available",
        "is_torchvision_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": ["AwqConfig", "BitsAndBytesConfig", "GPTQConfig"],
}

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bartpho"].append("BartphoTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.big_bird"].append("BigBirdTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizer")
    _import_structure["models.cpm"].append("CpmTokenizer")
    _import_structure["models.deberta_v2"].append("DebertaV2Tokenizer")
    _import_structure["models.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gpt_sw3"].append("GPTSw3Tokenizer")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizer")
    _import_structure["models.llama"].append("LlamaTokenizer")
    _import_structure["models.m2m_100"].append("M2M100Tokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart50"].append("MBart50Tokenizer")
    _import_structure["models.mluke"].append("MLukeTokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.nllb"].append("NllbTokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.plbart"].append("PLBartTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.rembert"].append("RemBertTokenizer")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    _import_structure["models.bart"].append("BartTokenizerFast")
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    _import_structure["models.bert"].append("BertTokenizerFast")
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        ["DPRContextEncoderTokenizerFast", "DPRQuestionEncoderTokenizerFast", "DPRReaderTokenizerFast"]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    _import_structure["models.led"].append("LEDTokenizerFast")
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    _import_structure["models.realm"].append("RealmTokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = ["SLOW_TO_FAST_CONVERTERS", "convert_slow_tokenizer"]

# Speech-specific objects
try:
    if not is_speech_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_speech_objects

    _import_structure["utils.dummy_speech_objects"] = [
        name for name in dir(dummy_speech_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.audio_spectrogram_transformer"].append("ASTFeatureExtractor")
    _import_structure["models.speech_to_text"].append("Speech2TextFeatureExtractor")

# Tensorflow-text-specific objects
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tensorflow_text_objects

    _import_structure["utils.dummy_tensorflow_text_objects"] = [
        name for name in dir(dummy_tensorflow_text_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.bert"].append("TFBertTokenizer")

# keras-nlp-specific objects
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_keras_nlp_objects

    _import_structure["utils.dummy_keras_nlp_objects"] = [
        name for name in dir(dummy_keras_nlp_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.gpt2"].append("TFGPT2Tokenizer")

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils"] = ["ImageProcessingMixin"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chinese_clip"].extend(["ChineseCLIPFeatureExtractor", "ChineseCLIPImageProcessor"])
    _import_structure["models.clip"].extend(["CLIPFeatureExtractor", "CLIPImageProcessor"])
    _import_structure["models.conditional_detr"].extend(
        ["ConditionalDetrFeatureExtractor", "ConditionalDetrImageProcessor"]
    )
    _import_structure["models.convnext"].extend(["ConvNextFeatureExtractor", "ConvNextImageProcessor"])
    _import_structure["models.deformable_detr"].extend(
        ["DeformableDetrFeatureExtractor", "DeformableDetrImageProcessor"]
    )
    _import_structure["models.deit"].extend(["DeiTFeatureExtractor", "DeiTImageProcessor"])
    _import_structure["models.deta"].append("DetaImageProcessor")
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    _import_structure["models.efficientformer"].append("EfficientFormerImageProcessor")
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    _import_structure["models.nougat"].append("NougatImageProcessor")
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    _import_structure["models.tvlt"].append("TvltImageProcessor")
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    _import_structure["models.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    _import_structure["models.vivit"].append("VivitImageProcessor")
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])


# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["activations"] = []
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "ForceTokensLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "LogitsWarper",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
            "top_k_top_p_filtering",
        ]
    )
    _import_structure["generation_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch models structure

    _import_structure["models.albert"].extend(
        [
            "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
            "load_tf_weights_in_albert",
        ]
    )
    _import_structure["models.align"].extend(
        [
            "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    _import_structure["models.altclip"].extend(
        [
            "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ASTForAudioClassification",
            "ASTModel",
            "ASTPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
            "MODEL_FOR_BACKBONE_MAPPING",
            "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_CTC_MAPPING",
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
            "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
            "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_VISION_2_SEQ_MAPPING",
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoBackbone",
            "AutoModel",
            "AutoModelForAudioClassification",
            "AutoModelForAudioFrameClassification",
            "AutoModelForAudioXVector",
            "AutoModelForCausalLM",
            "AutoModelForCTC",
            "AutoModelForDepthEstimation",
            "AutoModelForDocumentQuestionAnswering",
            "AutoModelForImageClassification",
            "AutoModelForImageSegmentation",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForMaskedImageModeling",
            "AutoModelForMaskedLM",
            "AutoModelForMaskGeneration",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForObjectDetection",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSemanticSegmentation",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTextEncoding",
            "AutoModelForTextToSpectrogram",
            "AutoModelForTextToWaveform",
            "AutoModelForTokenClassification",
            "AutoModelForUniversalSegmentation",
            "AutoModelForVideoClassification",
            "AutoModelForVision2Seq",
            "AutoModelForVisualQuestionAnswering",
            "AutoModelForZeroShotImageClassification",
            "AutoModelForZeroShotObjectDetection",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.autoformer"].extend(
        [
            "AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    _import_structure["models.bark"].extend(
        [
            "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BarkCausalModel",
            "BarkCoarseModel",
            "BarkFineModel",
            "BarkModel",
            "BarkPreTrainedModel",
            "BarkSemanticModel",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "BartPreTrainedModel",
            "PretrainedBartModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLayer",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
            "load_tf_weights_in_bert",
        ]
    )
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "BertGenerationPreTrainedModel",
            "load_tf_weights_in_bert_generation",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdLayer",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
            "load_tf_weights_in_big_bird",
        ]
    )
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BigBirdPegasusForCausalLM",
            "BigBirdPegasusForConditionalGeneration",
            "BigBirdPegasusForQuestionAnswering",
            "BigBirdPegasusForSequenceClassification",
            "BigBirdPegasusModel",
            "BigBirdPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.biogpt"].extend(
        [
            "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    _import_structure["models.bit"].extend(
        [
            "BIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlipForConditionalGeneration",
            "BlipForImageTextRetrieval",
            "BlipForQuestionAnswering",
            "BlipModel",
            "BlipPreTrainedModel",
            "BlipTextModel",
            "BlipVisionModel",
        ]
    )
    _import_structure["models.blip_2"].extend(
        [
            "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Blip2ForConditionalGeneration",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2VisionModel",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BloomForCausalLM",
            "BloomForQuestionAnswering",
            "BloomForSequenceClassification",
            "BloomForTokenClassification",
            "BloomModel",
            "BloomPreTrainedModel",
        ]
    )
    _import_structure["models.bridgetower"].extend(
        [
            "BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    _import_structure["models.bros"].extend(
        [
            "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BrosForTokenClassification",
            "BrosModel",
            "BrosPreTrainedModel",
            "BrosProcessor",
            "BrosSpadeEEForTokenClassification",
            "BrosSpadeELForTokenClassification",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
            "CamembertPreTrainedModel",
        ]
    )
    _import_structure["models.canine"].extend(
        [
            "CANINE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineLayer",
            "CanineModel",
            "CaninePreTrainedModel",
            "load_tf_weights_in_canine",
        ]
    )
    _import_structure["models.chinese_clip"].extend(
        [
            "CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    _import_structure["models.clap"].extend(
        [
            "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ClapAudioModel",
            "ClapAudioModelWithProjection",
            "ClapFeatureExtractor",
            "ClapModel",
            "ClapPreTrainedModel",
            "ClapTextModel",
            "ClapTextModelWithProjection",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    _import_structure["models.clipseg"].extend(
        [
            "CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    _import_structure["models.codegen"].extend(
        [
            "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    _import_structure["models.conditional_detr"].extend(
        [
            "CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertLayer",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.cpmant"].extend(
        [
            "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Data2VecAudioForAudioFrameClassification",
            "Data2VecAudioForCTC",
            "Data2VecAudioForSequenceClassification",
            "Data2VecAudioForXVector",
            "Data2VecAudioModel",
            "Data2VecAudioPreTrainedModel",
            "Data2VecTextForCausalLM",
            "Data2VecTextForMaskedLM",
            "Data2VecTextForMultipleChoice",
            "Data2VecTextForQuestionAnswering",
            "Data2VecTextForSequenceClassification",
            "Data2VecTextForTokenClassification",
            "Data2VecTextModel",
            "Data2VecTextPreTrainedModel",
            "Data2VecVisionForImageClassification",
            "Data2VecVisionForSemanticSegmentation",
            "Data2VecVisionModel",
            "Data2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaForMaskedLM",
            "DebertaForQuestionAnswering",
            "DebertaForSequenceClassification",
            "DebertaForTokenClassification",
            "DebertaModel",
            "DebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaV2ForMaskedLM",
            "DebertaV2ForMultipleChoice",
            "DebertaV2ForQuestionAnswering",
            "DebertaV2ForSequenceClassification",
            "DebertaV2ForTokenClassification",
            "DebertaV2Model",
            "DebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.decision_transformer"].extend(
        [
            "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.deprecated.open_llama"].extend(
        ["OpenLlamaForCausalLM", "OpenLlamaForSequenceClassification", "OpenLlamaModel", "OpenLlamaPreTrainedModel"]
    )
    _import_structure["models.deprecated.retribert"].extend(
        ["RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST", "RetriBertModel", "RetriBertPreTrainedModel"]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.deta"].extend(
        [
            "DETA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    _import_structure["models.detr"].extend(
        [
            "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    _import_structure["models.dinat"].extend(
        [
            "DINAT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.donut"].extend(
        [
            "DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPreTrainedModel",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    _import_structure["models.dpt"].extend(
        [
            "DPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    _import_structure["models.efficientformer"].extend(
        [
            "EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.efficientnet"].extend(
        [
            "EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ElectraForCausalLM",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
            "load_tf_weights_in_electra",
        ]
    )
    _import_structure["models.encodec"].extend(
        [
            "ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.ernie"].extend(
        [
            "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ErnieForCausalLM",
            "ErnieForMaskedLM",
            "ErnieForMultipleChoice",
            "ErnieForNextSentencePrediction",
            "ErnieForPreTraining",
            "ErnieForQuestionAnswering",
            "ErnieForSequenceClassification",
            "ErnieForTokenClassification",
            "ErnieModel",
            "ErniePreTrainedModel",
        ]
    )
    _import_structure["models.ernie_m"].extend(
        [
            "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    _import_structure["models.esm"].extend(
        [
            "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EsmFoldPreTrainedModel",
            "EsmForMaskedLM",
            "EsmForProteinFolding",
            "EsmForSequenceClassification",
            "EsmForTokenClassification",
            "EsmModel",
            "EsmPreTrainedModel",
        ]
    )
    _import_structure["models.falcon"].extend(
        [
            "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertPreTrainedModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.flava"].extend(
        [
            "FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlavaForPreTraining",
            "FlavaImageCodebook",
            "FlavaImageModel",
            "FlavaModel",
            "FlavaMultimodalModel",
            "FlavaPreTrainedModel",
            "FlavaTextModel",
        ]
    )
    _import_structure["models.fnet"].extend(
        [
            "FNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FNetForMaskedLM",
            "FNetForMultipleChoice",
            "FNetForNextSentencePrediction",
            "FNetForPreTraining",
            "FNetForQuestionAnswering",
            "FNetForSequenceClassification",
            "FNetForTokenClassification",
            "FNetLayer",
            "FNetModel",
            "FNetPreTrainedModel",
        ]
    )
    _import_structure["models.focalnet"].extend(
        [
            "FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FocalNetBackbone",
            "FocalNetForImageClassification",
            "FocalNetForMaskedImageModeling",
            "FocalNetModel",
            "FocalNetPreTrainedModel",
        ]
    )
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    _import_structure["models.funnel"].extend(
        [
            "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FunnelBaseModel",
            "FunnelForMaskedLM",
            "FunnelForMultipleChoice",
            "FunnelForPreTraining",
            "FunnelForQuestionAnswering",
            "FunnelForSequenceClassification",
            "FunnelForTokenClassification",
            "FunnelModel",
            "FunnelPreTrainedModel",
            "load_tf_weights_in_funnel",
        ]
    )
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    _import_structure["models.git"].extend(
        [
            "GIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GitForCausalLM",
            "GitModel",
            "GitPreTrainedModel",
            "GitVisionModel",
        ]
    )
    _import_structure["models.glpn"].extend(
        [
            "GLPN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GLPNForDepthEstimation",
            "GLPNModel",
            "GLPNPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPT2DoubleHeadsModel",
            "GPT2ForQuestionAnswering",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2PreTrainedModel",
            "load_tf_weights_in_gpt2",
        ]
    )
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTBigCodeForCausalLM",
            "GPTBigCodeForSequenceClassification",
            "GPTBigCodeForTokenClassification",
            "GPTBigCodeModel",
            "GPTBigCodePreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neo"].extend(
        [
            "GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTNeoForCausalLM",
            "GPTNeoForQuestionAnswering",
            "GPTNeoForSequenceClassification",
            "GPTNeoForTokenClassification",
            "GPTNeoModel",
            "GPTNeoPreTrainedModel",
            "load_tf_weights_in_gpt_neo",
        ]
    )
    _import_structure["models.gpt_neox"].extend(
        [
            "GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTNeoXForCausalLM",
            "GPTNeoXForQuestionAnswering",
            "GPTNeoXForSequenceClassification",
            "GPTNeoXForTokenClassification",
            "GPTNeoXLayer",
            "GPTNeoXModel",
            "GPTNeoXPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTNeoXJapaneseForCausalLM",
            "GPTNeoXJapaneseLayer",
            "GPTNeoXJapaneseModel",
            "GPTNeoXJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTJForCausalLM",
            "GPTJForQuestionAnswering",
            "GPTJForSequenceClassification",
            "GPTJModel",
            "GPTJPreTrainedModel",
        ]
    )
    _import_structure["models.gptsan_japanese"].extend(
        [
            "GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GPTSanJapaneseForConditionalGeneration",
            "GPTSanJapaneseModel",
            "GPTSanJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.graphormer"].extend(
        [
            "GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GraphormerForGraphClassification",
            "GraphormerModel",
            "GraphormerPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "GroupViTModel",
            "GroupViTPreTrainedModel",
            "GroupViTTextModel",
            "GroupViTVisionModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
            "IBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "IBertForMaskedLM",
            "IBertForMultipleChoice",
            "IBertForQuestionAnswering",
            "IBertForSequenceClassification",
            "IBertForTokenClassification",
            "IBertModel",
            "IBertPreTrainedModel",
        ]
    )
    _import_structure["models.idefics"].extend(
        [
            "IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
            "load_tf_weights_in_imagegpt",
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    _import_structure["models.jukebox"].extend(
        [
            "JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST",
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMForMaskedLM",
            "LayoutLMForQuestionAnswering",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
            "LayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv2"].extend(
        [
            "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LED_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    _import_structure["models.levit"].extend(
        [
            "LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    _import_structure["models.lilt"].extend(
        [
            "LILT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    _import_structure["models.llama"].extend(
        ["LlamaForCausalLM", "LlamaForSequenceClassification", "LlamaModel", "LlamaPreTrainedModel"]
    )
    _import_structure["models.longformer"].extend(
        [
            "LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
            "LongformerSelfAttention",
        ]
    )
    _import_structure["models.longt5"].extend(
        [
            "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    _import_structure["models.luke"].extend(
        [
            "LUKE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LukeForEntityClassification",
            "LukeForEntityPairClassification",
            "LukeForEntitySpanClassification",
            "LukeForMaskedLM",
            "LukeForMultipleChoice",
            "LukeForQuestionAnswering",
            "LukeForSequenceClassification",
            "LukeForTokenClassification",
            "LukeModel",
            "LukePreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
            "LxmertXLayer",
        ]
    )
    _import_structure["models.m2m_100"].extend(
        [
            "M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST",
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(["MarianForCausalLM", "MarianModel", "MarianMTModel"])
    _import_structure["models.markuplm"].extend(
        [
            "MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    _import_structure["models.mask2former"].extend(
        [
            "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    _import_structure["models.maskformer"].extend(
        [
            "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MaskFormerForInstanceSegmentation",
            "MaskFormerModel",
            "MaskFormerPreTrainedModel",
            "MaskFormerSwinBackbone",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
            "MBartPreTrainedModel",
        ]
    )
    _import_structure["models.mega"].extend(
        [
            "MEGA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MegaForCausalLM",
            "MegaForMaskedLM",
            "MegaForMultipleChoice",
            "MegaForQuestionAnswering",
            "MegaForSequenceClassification",
            "MegaForTokenClassification",
            "MegaModel",
            "MegaPreTrainedModel",
        ]
    )
    _import_structure["models.megatron_bert"].extend(
        [
            "MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MegatronBertForCausalLM",
            "MegatronBertForMaskedLM",
            "MegatronBertForMultipleChoice",
            "MegatronBertForNextSentencePrediction",
            "MegatronBertForPreTraining",
            "MegatronBertForQuestionAnswering",
            "MegatronBertForSequenceClassification",
            "MegatronBertForTokenClassification",
            "MegatronBertModel",
            "MegatronBertPreTrainedModel",
        ]
    )
    _import_structure["models.mgp_str"].extend(
        [
            "MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        ["MistralForCausalLM", "MistralForSequenceClassification", "MistralModel", "MistralPreTrainedModel"]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertLayer",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
            "load_tf_weights_in_mobilenet_v1",
        ]
    )
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
            "load_tf_weights_in_mobilenet_v2",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevitv2"].extend(
        [
            "MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileViTV2ForImageClassification",
            "MobileViTV2ForSemanticSegmentation",
            "MobileViTV2Model",
            "MobileViTV2PreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetLayer",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mpt"].extend(
        [
            "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MptForCausalLM",
            "MptForQuestionAnswering",
            "MptForSequenceClassification",
            "MptForTokenClassification",
            "MptModel",
            "MptPreTrainedModel",
        ]
    )
    _import_structure["models.mra"].extend(
        [
            "MRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MraForMaskedLM",
            "MraForMultipleChoice",
            "MraForQuestionAnswering",
            "MraForSequenceClassification",
            "MraForTokenClassification",
            "MraModel",
            "MraPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",
            "MT5ForConditionalGeneration",
            "MT5ForQuestionAnswering",
            "MT5ForSequenceClassification",
            "MT5Model",
            "MT5PreTrainedModel",
        ]
    )
    _import_structure["models.musicgen"].extend(
        [
            "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MusicgenForCausalLM",
            "MusicgenForConditionalGeneration",
            "MusicgenModel",
            "MusicgenPreTrainedModel",
            "MusicgenProcessor",
        ]
    )
    _import_structure["models.mvp"].extend(
        [
            "MVP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MvpForCausalLM",
            "MvpForConditionalGeneration",
            "MvpForQuestionAnswering",
            "MvpForSequenceClassification",
            "MvpModel",
            "MvpPreTrainedModel",
        ]
    )
    _import_structure["models.nat"].extend(
        [
            "NAT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "NatBackbone",
            "NatForImageClassification",
            "NatModel",
            "NatPreTrainedModel",
        ]
    )
    _import_structure["models.nezha"].extend(
        [
            "NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "NezhaForMaskedLM",
            "NezhaForMultipleChoice",
            "NezhaForNextSentencePrediction",
            "NezhaForPreTraining",
            "NezhaForQuestionAnswering",
            "NezhaForSequenceClassification",
            "NezhaForTokenClassification",
            "NezhaModel",
            "NezhaPreTrainedModel",
        ]
    )
    _import_structure["models.nllb_moe"].extend(
        [
            "NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "NllbMoeForConditionalGeneration",
            "NllbMoeModel",
            "NllbMoePreTrainedModel",
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "NystromformerForMaskedLM",
            "NystromformerForMultipleChoice",
            "NystromformerForQuestionAnswering",
            "NystromformerForSequenceClassification",
            "NystromformerForTokenClassification",
            "NystromformerLayer",
            "NystromformerModel",
            "NystromformerPreTrainedModel",
        ]
    )
    _import_structure["models.oneformer"].extend(
        [
            "ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "OneFormerForUniversalSegmentation",
            "OneFormerModel",
            "OneFormerPreTrainedModel",
        ]
    )
    _import_structure["models.openai"].extend(
        [
            "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "OpenAIGPTDoubleHeadsModel",
            "OpenAIGPTForSequenceClassification",
            "OpenAIGPTLMHeadModel",
            "OpenAIGPTModel",
            "OpenAIGPTPreTrainedModel",
            "load_tf_weights_in_openai_gpt",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "OPTForCausalLM",
            "OPTForQuestionAnswering",
            "OPTForSequenceClassification",
            "OPTModel",
            "OPTPreTrainedModel",
        ]
    )
    _import_structure["models.owlv2"].extend(
        [
            "OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Owlv2ForObjectDetection",
            "Owlv2Model",
            "Owlv2PreTrainedModel",
            "Owlv2TextModel",
            "Owlv2VisionModel",
        ]
    )
    _import_structure["models.owlvit"].extend(
        [
            "OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "OwlViTForObjectDetection",
            "OwlViTModel",
            "OwlViTPreTrainedModel",
            "OwlViTTextModel",
            "OwlViTVisionModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        ["PegasusForCausalLM", "PegasusForConditionalGeneration", "PegasusModel", "PegasusPreTrainedModel"]
    )
    _import_structure["models.pegasus_x"].extend(
        [
            "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    _import_structure["models.perceiver"].extend(
        [
            "PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverLayer",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    _import_structure["models.persimmon"].extend(
        ["PersimmonForCausalLM", "PersimmonForSequenceClassification", "PersimmonModel", "PersimmonPreTrainedModel"]
    )
    _import_structure["models.pix2struct"].extend(
        [
            "PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    _import_structure["models.plbart"].extend(
        [
            "PLBART_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    _import_structure["models.poolformer"].extend(
        [
            "POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    _import_structure["models.pop2piano"].extend(
        [
            "POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    _import_structure["models.prophetnet"].extend(
        [
            "PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.pvt"].extend(
        [
            "PVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    _import_structure["models.qdqbert"].extend(
        [
            "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLayer",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
            "load_tf_weights_in_qdqbert",
        ]
    )
    _import_structure["models.rag"].extend(
        ["RagModel", "RagPreTrainedModel", "RagSequenceForGeneration", "RagTokenForGeneration"]
    )
    _import_structure["models.realm"].extend(
        [
            "REALM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RealmEmbedder",
            "RealmForOpenQA",
            "RealmKnowledgeAugEncoder",
            "RealmPreTrainedModel",
            "RealmReader",
            "RealmRetriever",
            "RealmScorer",
            "load_tf_weights_in_realm",
        ]
    )
    _import_structure["models.reformer"].extend(
        [
            "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ReformerAttention",
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerLayer",
            "ReformerModel",
            "ReformerModelWithLMHead",
            "ReformerPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RegNetForImageClassification",
            "RegNetModel",
            "RegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RemBertForCausalLM",
            "RemBertForMaskedLM",
            "RemBertForMultipleChoice",
            "RemBertForQuestionAnswering",
            "RemBertForSequenceClassification",
            "RemBertForTokenClassification",
            "RemBertLayer",
            "RemBertModel",
            "RemBertPreTrainedModel",
            "load_tf_weights_in_rembert",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ResNetBackbone",
            "ResNetForImageClassification",
            "ResNetModel",
            "ResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
            "RobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RobertaPreLayerNormForCausalLM",
            "RobertaPreLayerNormForMaskedLM",
            "RobertaPreLayerNormForMultipleChoice",
            "RobertaPreLayerNormForQuestionAnswering",
            "RobertaPreLayerNormForSequenceClassification",
            "RobertaPreLayerNormForTokenClassification",
            "RobertaPreLayerNormModel",
            "RobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roc_bert"].extend(
        [
            "ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertLayer",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
            "load_tf_weights_in_roc_bert",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerLayer",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
            "load_tf_weights_in_roformer",
        ]
    )
    _import_structure["models.rwkv"].extend(
        [
            "RWKV_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SamModel",
            "SamPreTrainedModel",
        ]
    )
    _import_structure["models.seamless_m4t"].extend(
        [
            "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SeamlessM4TCodeHifiGan",
            "SeamlessM4TForSpeechToSpeech",
            "SeamlessM4TForSpeechToText",
            "SeamlessM4TForTextToSpeech",
            "SeamlessM4TForTextToText",
            "SeamlessM4THifiGan",
            "SeamlessM4TModel",
            "SeamlessM4TPreTrainedModel",
            "SeamlessM4TTextToUnitForConditionalGeneration",
            "SeamlessM4TTextToUnitModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SegformerDecodeHead",
            "SegformerForImageClassification",
            "SegformerForSemanticSegmentation",
            "SegformerLayer",
            "SegformerModel",
            "SegformerPreTrainedModel",
        ]
    )
    _import_structure["models.sew"].extend(
        [
            "SEW_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SEWForCTC",
            "SEWForSequenceClassification",
            "SEWModel",
            "SEWPreTrainedModel",
        ]
    )
    _import_structure["models.sew_d"].extend(
        [
            "SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SEWDForCTC",
            "SEWDForSequenceClassification",
            "SEWDModel",
            "SEWDPreTrainedModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    _import_structure["models.speech_to_text"].extend(
        [
            "SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.speech_to_text_2"].extend(["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"])
    _import_structure["models.speecht5"].extend(
        [
            "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SpeechT5ForSpeechToSpeech",
            "SpeechT5ForSpeechToText",
            "SpeechT5ForTextToSpeech",
            "SpeechT5HifiGan",
            "SpeechT5Model",
            "SpeechT5PreTrainedModel",
        ]
    )
    _import_structure["models.splinter"].extend(
        [
            "SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterLayer",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertModule",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SwinBackbone",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinModel",
            "SwinPreTrainedModel",
        ]
    )
    _import_structure["models.swin2sr"].extend(
        [
            "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Swin2SRForImageSuperResolution",
            "Swin2SRModel",
            "Swin2SRPreTrainedModel",
        ]
    )
    _import_structure["models.swinv2"].extend(
        [
            "SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Swinv2ForImageClassification",
            "Swinv2ForMaskedImageModeling",
            "Swinv2Model",
            "Swinv2PreTrainedModel",
        ]
    )
    _import_structure["models.switch_transformers"].extend(
        [
            "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SwitchTransformersEncoderModel",
            "SwitchTransformersForConditionalGeneration",
            "SwitchTransformersModel",
            "SwitchTransformersPreTrainedModel",
            "SwitchTransformersSparseMLP",
            "SwitchTransformersTop1Router",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5ForQuestionAnswering",
            "T5ForSequenceClassification",
            "T5Model",
            "T5PreTrainedModel",
            "load_tf_weights_in_t5",
        ]
    )
    _import_structure["models.table_transformer"].extend(
        [
            "TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TableTransformerForObjectDetection",
            "TableTransformerModel",
            "TableTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
            "TapasPreTrainedModel",
            "load_tf_weights_in_tapas",
        ]
    )
    _import_structure["models.time_series_transformer"].extend(
        [
            "TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TimeSeriesTransformerForPrediction",
            "TimeSeriesTransformerModel",
            "TimeSeriesTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.timesformer"].extend(
        [
            "TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TimesformerForVideoClassification",
            "TimesformerModel",
            "TimesformerPreTrainedModel",
        ]
    )
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])
    _import_structure["models.transfo_xl"].extend(
        [
            "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.trocr"].extend(
        ["TROCR_PRETRAINED_MODEL_ARCHIVE_LIST", "TrOCRForCausalLM", "TrOCRPreTrainedModel"]
    )
    _import_structure["models.tvlt"].extend(
        [
            "TVLT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    _import_structure["models.unispeech"].extend(
        [
            "UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    _import_structure["models.unispeech_sat"].extend(
        [
            "UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    _import_structure["models.videomae"].extend(
        [
            "VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vilt"].extend(
        [
            "VILT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltLayer",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    _import_structure["models.visual_bert"].extend(
        [
            "VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertLayer",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    _import_structure["models.vit"].extend(
        [
            "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_hybrid"].extend(
        [
            "VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTMAEForPreTraining",
            "ViTMAELayer",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vit_msn"].extend(
        [
            "VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    _import_structure["models.vitdet"].extend(
        [
            "VITDET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    _import_structure["models.vitmatte"].extend(
        [
            "VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    _import_structure["models.vits"].extend(
        [
            "VITS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    _import_structure["models.vivit"].extend(
        [
            "VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2ForAudioFrameClassification",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2ForPreTraining",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2ForXVector",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2ConformerForAudioFrameClassification",
            "Wav2Vec2ConformerForCTC",
            "Wav2Vec2ConformerForPreTraining",
            "Wav2Vec2ConformerForSequenceClassification",
            "Wav2Vec2ConformerForXVector",
            "Wav2Vec2ConformerModel",
            "Wav2Vec2ConformerPreTrainedModel",
        ]
    )
    _import_structure["models.wavlm"].extend(
        [
            "WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "WavLMForAudioFrameClassification",
            "WavLMForCTC",
            "WavLMForSequenceClassification",
            "WavLMForXVector",
            "WavLMModel",
            "WavLMPreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_prophetnet"].extend(
        [
            "XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
            "XLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMRobertaXLForCausalLM",
            "XLMRobertaXLForMaskedLM",
            "XLMRobertaXLForMultipleChoice",
            "XLMRobertaXLForQuestionAnswering",
            "XLMRobertaXLForSequenceClassification",
            "XLMRobertaXLForTokenClassification",
            "XLMRobertaXLModel",
            "XLMRobertaXLPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
            "load_tf_weights_in_xlnet",
        ]
    )
    _import_structure["models.xmod"].extend(
        [
            "XMOD_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XmodForCausalLM",
            "XmodForMaskedLM",
            "XmodForMultipleChoice",
            "XmodForQuestionAnswering",
            "XmodForSequenceClassification",
            "XmodForTokenClassification",
            "XmodModel",
            "XmodPreTrainedModel",
        ]
    )
    _import_structure["models.yolos"].extend(
        [
            "YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YOSO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoLayer",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["pytorch_utils"] = ["Conv1D", "apply_chunking_to_forward", "prune_layer"]
    _import_structure["sagemaker"] = []
    _import_structure["time_series_utils"] = []
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]

# TensorFlow-backed objects
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    _import_structure["activations_tf"] = []
    _import_structure["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
    _import_structure["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
    _import_structure["generation"].extend(
        [
            "TFForcedBOSTokenLogitsProcessor",
            "TFForcedEOSTokenLogitsProcessor",
            "TFForceTokensLogitsProcessor",
            "TFGenerationMixin",
            "TFLogitsProcessor",
            "TFLogitsProcessorList",
            "TFLogitsWarper",
            "TFMinLengthLogitsProcessor",
            "TFNoBadWordsLogitsProcessor",
            "TFNoRepeatNGramLogitsProcessor",
            "TFRepetitionPenaltyLogitsProcessor",
            "TFSuppressTokensAtBeginLogitsProcessor",
            "TFSuppressTokensLogitsProcessor",
            "TFTemperatureLogitsWarper",
            "TFTopKLogitsWarper",
            "TFTopPLogitsWarper",
            "tf_top_k_top_p_filtering",
        ]
    )
    _import_structure["generation_tf_utils"] = []
    _import_structure["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
    _import_structure["modeling_tf_outputs"] = []
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    # TensorFlow models structure
    _import_structure["models.albert"].extend(
        [
            "TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAlbertForMaskedLM",
            "TFAlbertForMultipleChoice",
            "TFAlbertForPreTraining",
            "TFAlbertForQuestionAnswering",
            "TFAlbertForSequenceClassification",
            "TFAlbertForTokenClassification",
            "TFAlbertMainLayer",
            "TFAlbertModel",
            "TFAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "TF_MODEL_FOR_MASKED_LM_MAPPING",
            "TF_MODEL_FOR_MASK_GENERATION_MAPPING",
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "TF_MODEL_FOR_PRETRAINING_MAPPING",
            "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",
            "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_MAPPING",
            "TF_MODEL_WITH_LM_HEAD_MAPPING",
            "TFAutoModel",
            "TFAutoModelForAudioClassification",
            "TFAutoModelForCausalLM",
            "TFAutoModelForDocumentQuestionAnswering",
            "TFAutoModelForImageClassification",
            "TFAutoModelForMaskedImageModeling",
            "TFAutoModelForMaskedLM",
            "TFAutoModelForMaskGeneration",
            "TFAutoModelForMultipleChoice",
            "TFAutoModelForNextSentencePrediction",
            "TFAutoModelForPreTraining",
            "TFAutoModelForQuestionAnswering",
            "TFAutoModelForSemanticSegmentation",
            "TFAutoModelForSeq2SeqLM",
            "TFAutoModelForSequenceClassification",
            "TFAutoModelForSpeechSeq2Seq",
            "TFAutoModelForTableQuestionAnswering",
            "TFAutoModelForTextEncoding",
            "TFAutoModelForTokenClassification",
            "TFAutoModelForVision2Seq",
            "TFAutoModelForZeroShotImageClassification",
            "TFAutoModelWithLMHead",
        ]
    )
    _import_structure["models.bart"].extend(
        ["TFBartForConditionalGeneration", "TFBartForSequenceClassification", "TFBartModel", "TFBartPretrainedModel"]
    )
    _import_structure["models.bert"].extend(
        [
            "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFBertEmbeddings",
            "TFBertForMaskedLM",
            "TFBertForMultipleChoice",
            "TFBertForNextSentencePrediction",
            "TFBertForPreTraining",
            "TFBertForQuestionAnswering",
            "TFBertForSequenceClassification",
            "TFBertForTokenClassification",
            "TFBertLMHeadModel",
            "TFBertMainLayer",
            "TFBertModel",
            "TFBertPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        ["TFBlenderbotForConditionalGeneration", "TFBlenderbotModel", "TFBlenderbotPreTrainedModel"]
    )
    _import_structure["models.blenderbot_small"].extend(
        ["TFBlenderbotSmallForConditionalGeneration", "TFBlenderbotSmallModel", "TFBlenderbotSmallPreTrainedModel"]
    )
    _import_structure["models.blip"].extend(
        [
            "TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFBlipForConditionalGeneration",
            "TFBlipForImageTextRetrieval",
            "TFBlipForQuestionAnswering",
            "TFBlipModel",
            "TFBlipPreTrainedModel",
            "TFBlipTextModel",
            "TFBlipVisionModel",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCamembertForCausalLM",
            "TFCamembertForMaskedLM",
            "TFCamembertForMultipleChoice",
            "TFCamembertForQuestionAnswering",
            "TFCamembertForSequenceClassification",
            "TFCamembertForTokenClassification",
            "TFCamembertModel",
            "TFCamembertPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCLIPModel",
            "TFCLIPPreTrainedModel",
            "TFCLIPTextModel",
            "TFCLIPVisionModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
            "TFConvBertLayer",
            "TFConvBertModel",
            "TFConvBertPreTrainedModel",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "TFConvNextForImageClassification",
            "TFConvNextModel",
            "TFConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "TFConvNextV2ForImageClassification",
            "TFConvNextV2Model",
            "TFConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCvtForImageClassification",
            "TFCvtModel",
            "TFCvtPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "TFData2VecVisionForImageClassification",
            "TFData2VecVisionForSemanticSegmentation",
            "TFData2VecVisionModel",
            "TFData2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDebertaForMaskedLM",
            "TFDebertaForQuestionAnswering",
            "TFDebertaForSequenceClassification",
            "TFDebertaForTokenClassification",
            "TFDebertaModel",
            "TFDebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDebertaV2ForMaskedLM",
            "TFDebertaV2ForMultipleChoice",
            "TFDebertaV2ForQuestionAnswering",
            "TFDebertaV2ForSequenceClassification",
            "TFDebertaV2ForTokenClassification",
            "TFDebertaV2Model",
            "TFDebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDeiTForImageClassification",
            "TFDeiTForImageClassificationWithTeacher",
            "TFDeiTForMaskedImageModeling",
            "TFDeiTModel",
            "TFDeiTPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDistilBertForMaskedLM",
            "TFDistilBertForMultipleChoice",
            "TFDistilBertForQuestionAnswering",
            "TFDistilBertForSequenceClassification",
            "TFDistilBertForTokenClassification",
            "TFDistilBertMainLayer",
            "TFDistilBertModel",
            "TFDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    _import_structure["models.efficientformer"].extend(
        [
            "TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFEfficientFormerForImageClassification",
            "TFEfficientFormerForImageClassificationWithTeacher",
            "TFEfficientFormerModel",
            "TFEfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFElectraForMaskedLM",
            "TFElectraForMultipleChoice",
            "TFElectraForPreTraining",
            "TFElectraForQuestionAnswering",
            "TFElectraForSequenceClassification",
            "TFElectraForTokenClassification",
            "TFElectraModel",
            "TFElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("TFEncoderDecoderModel")
    _import_structure["models.esm"].extend(
        [
            "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFEsmForMaskedLM",
            "TFEsmForSequenceClassification",
            "TFEsmForTokenClassification",
            "TFEsmModel",
            "TFEsmPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFlaubertForMultipleChoice",
            "TFFlaubertForQuestionAnsweringSimple",
            "TFFlaubertForSequenceClassification",
            "TFFlaubertForTokenClassification",
            "TFFlaubertModel",
            "TFFlaubertPreTrainedModel",
            "TFFlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.funnel"].extend(
        [
            "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFunnelBaseModel",
            "TFFunnelForMaskedLM",
            "TFFunnelForMultipleChoice",
            "TFFunnelForPreTraining",
            "TFFunnelForQuestionAnswering",
            "TFFunnelForSequenceClassification",
            "TFFunnelForTokenClassification",
            "TFFunnelModel",
            "TFFunnelPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFGPT2DoubleHeadsModel",
            "TFGPT2ForSequenceClassification",
            "TFGPT2LMHeadModel",
            "TFGPT2MainLayer",
            "TFGPT2Model",
            "TFGPT2PreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "TFGPTJForCausalLM",
            "TFGPTJForQuestionAnswering",
            "TFGPTJForSequenceClassification",
            "TFGPTJModel",
            "TFGPTJPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFGroupViTModel",
            "TFGroupViTPreTrainedModel",
            "TFGroupViTTextModel",
            "TFGroupViTVisionModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFHubertForCTC",
            "TFHubertModel",
            "TFHubertPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLayoutLMForMaskedLM",
            "TFLayoutLMForQuestionAnswering",
            "TFLayoutLMForSequenceClassification",
            "TFLayoutLMForTokenClassification",
            "TFLayoutLMMainLayer",
            "TFLayoutLMModel",
            "TFLayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLayoutLMv3ForQuestionAnswering",
            "TFLayoutLMv3ForSequenceClassification",
            "TFLayoutLMv3ForTokenClassification",
            "TFLayoutLMv3Model",
            "TFLayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"])
    _import_structure["models.longformer"].extend(
        [
            "TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerPreTrainedModel",
            "TFLongformerSelfAttention",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLxmertForPreTraining",
            "TFLxmertMainLayer",
            "TFLxmertModel",
            "TFLxmertPreTrainedModel",
            "TFLxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.marian"].extend(["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"])
    _import_structure["models.mbart"].extend(
        ["TFMBartForConditionalGeneration", "TFMBartModel", "TFMBartPreTrainedModel"]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMobileBertForMaskedLM",
            "TFMobileBertForMultipleChoice",
            "TFMobileBertForNextSentencePrediction",
            "TFMobileBertForPreTraining",
            "TFMobileBertForQuestionAnswering",
            "TFMobileBertForSequenceClassification",
            "TFMobileBertForTokenClassification",
            "TFMobileBertMainLayer",
            "TFMobileBertModel",
            "TFMobileBertPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMobileViTForImageClassification",
            "TFMobileViTForSemanticSegmentation",
            "TFMobileViTModel",
            "TFMobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMPNetForMaskedLM",
            "TFMPNetForMultipleChoice",
            "TFMPNetForQuestionAnswering",
            "TFMPNetForSequenceClassification",
            "TFMPNetForTokenClassification",
            "TFMPNetMainLayer",
            "TFMPNetModel",
            "TFMPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"])
    _import_structure["models.openai"].extend(
        [
            "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFOpenAIGPTDoubleHeadsModel",
            "TFOpenAIGPTForSequenceClassification",
            "TFOpenAIGPTLMHeadModel",
            "TFOpenAIGPTMainLayer",
            "TFOpenAIGPTModel",
            "TFOpenAIGPTPreTrainedModel",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "TFOPTForCausalLM",
            "TFOPTModel",
            "TFOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        ["TFPegasusForConditionalGeneration", "TFPegasusModel", "TFPegasusPreTrainedModel"]
    )
    _import_structure["models.rag"].extend(
        [
            "TFRagModel",
            "TFRagPreTrainedModel",
            "TFRagSequenceForGeneration",
            "TFRagTokenForGeneration",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRegNetForImageClassification",
            "TFRegNetModel",
            "TFRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRemBertForCausalLM",
            "TFRemBertForMaskedLM",
            "TFRemBertForMultipleChoice",
            "TFRemBertForQuestionAnswering",
            "TFRemBertForSequenceClassification",
            "TFRemBertForTokenClassification",
            "TFRemBertLayer",
            "TFRemBertModel",
            "TFRemBertPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFResNetForImageClassification",
            "TFResNetModel",
            "TFResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRobertaForCausalLM",
            "TFRobertaForMaskedLM",
            "TFRobertaForMultipleChoice",
            "TFRobertaForQuestionAnswering",
            "TFRobertaForSequenceClassification",
            "TFRobertaForTokenClassification",
            "TFRobertaMainLayer",
            "TFRobertaModel",
            "TFRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRobertaPreLayerNormForCausalLM",
            "TFRobertaPreLayerNormForMaskedLM",
            "TFRobertaPreLayerNormForMultipleChoice",
            "TFRobertaPreLayerNormForQuestionAnswering",
            "TFRobertaPreLayerNormForSequenceClassification",
            "TFRobertaPreLayerNormForTokenClassification",
            "TFRobertaPreLayerNormMainLayer",
            "TFRobertaPreLayerNormModel",
            "TFRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRoFormerForCausalLM",
            "TFRoFormerForMaskedLM",
            "TFRoFormerForMultipleChoice",
            "TFRoFormerForQuestionAnswering",
            "TFRoFormerForSequenceClassification",
            "TFRoFormerForTokenClassification",
            "TFRoFormerLayer",
            "TFRoFormerModel",
            "TFRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSamModel",
            "TFSamPreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSegformerDecodeHead",
            "TFSegformerForImageClassification",
            "TFSegformerForSemanticSegmentation",
            "TFSegformerModel",
            "TFSegformerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_to_text"].extend(
        [
            "TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFTapasForMaskedLM",
            "TFTapasForQuestionAnswering",
            "TFTapasForSequenceClassification",
            "TFTapasModel",
            "TFTapasPreTrainedModel",
        ]
    )
    _import_structure["models.transfo_xl"].extend(
        [
            "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["TFVisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["TFVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(
        [
            "TFViTForImageClassification",
            "TFViTModel",
            "TFViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "TFViTMAEForPreTraining",
            "TFViTMAEModel",
            "TFViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMForMultipleChoice",
            "TFXLMForQuestionAnsweringSimple",
            "TFXLMForSequenceClassification",
            "TFXLMForTokenClassification",
            "TFXLMMainLayer",
            "TFXLMModel",
            "TFXLMPreTrainedModel",
            "TFXLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMRobertaForCausalLM",
            "TFXLMRobertaForMaskedLM",
            "TFXLMRobertaForMultipleChoice",
            "TFXLMRobertaForQuestionAnswering",
            "TFXLMRobertaForSequenceClassification",
            "TFXLMRobertaForTokenClassification",
            "TFXLMRobertaModel",
            "TFXLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLNetForMultipleChoice",
            "TFXLNetForQuestionAnsweringSimple",
            "TFXLNetForSequenceClassification",
            "TFXLNetForTokenClassification",
            "TFXLNetLMHeadModel",
            "TFXLNetMainLayer",
            "TFXLNetModel",
            "TFXLNetPreTrainedModel",
        ]
    )
    _import_structure["optimization_tf"] = ["AdamWeightDecay", "GradientAccumulator", "WarmUp", "create_optimizer"]
    _import_structure["tf_utils"] = []
    _import_structure["trainer_tf"] = ["TFTrainer"]


try:
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects

    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")


# FLAX-backed objects
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    _import_structure["generation"].extend(
        [
            "FlaxForcedBOSTokenLogitsProcessor",
            "FlaxForcedEOSTokenLogitsProcessor",
            "FlaxForceTokensLogitsProcessor",
            "FlaxGenerationMixin",
            "FlaxLogitsProcessor",
            "FlaxLogitsProcessorList",
            "FlaxLogitsWarper",
            "FlaxMinLengthLogitsProcessor",
            "FlaxTemperatureLogitsWarper",
            "FlaxSuppressTokensAtBeginLogitsProcessor",
            "FlaxSuppressTokensLogitsProcessor",
            "FlaxTopKLogitsWarper",
            "FlaxTopPLogitsWarper",
            "FlaxWhisperTimeStampLogitsProcessor",
        ]
    )
    _import_structure["generation_flax_utils"] = []
    _import_structure["modeling_flax_outputs"] = []
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]
    _import_structure["models.albert"].extend(
        [
            "FlaxAlbertForMaskedLM",
            "FlaxAlbertForMultipleChoice",
            "FlaxAlbertForPreTraining",
            "FlaxAlbertForQuestionAnswering",
            "FlaxAlbertForSequenceClassification",
            "FlaxAlbertForTokenClassification",
            "FlaxAlbertModel",
            "FlaxAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
            "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
            "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "FLAX_MODEL_MAPPING",
            "FlaxAutoModel",
            "FlaxAutoModelForCausalLM",
            "FlaxAutoModelForImageClassification",
            "FlaxAutoModelForMaskedLM",
            "FlaxAutoModelForMultipleChoice",
            "FlaxAutoModelForNextSentencePrediction",
            "FlaxAutoModelForPreTraining",
            "FlaxAutoModelForQuestionAnswering",
            "FlaxAutoModelForSeq2SeqLM",
            "FlaxAutoModelForSequenceClassification",
            "FlaxAutoModelForSpeechSeq2Seq",
            "FlaxAutoModelForTokenClassification",
            "FlaxAutoModelForVision2Seq",
        ]
    )

    # Flax models structure

    _import_structure["models.bart"].extend(
        [
            "FlaxBartDecoderPreTrainedModel",
            "FlaxBartForCausalLM",
            "FlaxBartForConditionalGeneration",
            "FlaxBartForQuestionAnswering",
            "FlaxBartForSequenceClassification",
            "FlaxBartModel",
            "FlaxBartPreTrainedModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "FlaxBeitForImageClassification",
            "FlaxBeitForMaskedImageModeling",
            "FlaxBeitModel",
            "FlaxBeitPreTrainedModel",
        ]
    )

    _import_structure["models.bert"].extend(
        [
            "FlaxBertForCausalLM",
            "FlaxBertForMaskedLM",
            "FlaxBertForMultipleChoice",
            "FlaxBertForNextSentencePrediction",
            "FlaxBertForPreTraining",
            "FlaxBertForQuestionAnswering",
            "FlaxBertForSequenceClassification",
            "FlaxBertForTokenClassification",
            "FlaxBertModel",
            "FlaxBertPreTrainedModel",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "FlaxBigBirdForCausalLM",
            "FlaxBigBirdForMaskedLM",
            "FlaxBigBirdForMultipleChoice",
            "FlaxBigBirdForPreTraining",
            "FlaxBigBirdForQuestionAnswering",
            "FlaxBigBirdForSequenceClassification",
            "FlaxBigBirdForTokenClassification",
            "FlaxBigBirdModel",
            "FlaxBigBirdPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        ["FlaxBlenderbotForConditionalGeneration", "FlaxBlenderbotModel", "FlaxBlenderbotPreTrainedModel"]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "FlaxBlenderbotSmallForConditionalGeneration",
            "FlaxBlenderbotSmallModel",
            "FlaxBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "FlaxBloomForCausalLM",
            "FlaxBloomModel",
            "FlaxBloomPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "FlaxCLIPModel",
            "FlaxCLIPPreTrainedModel",
            "FlaxCLIPTextModel",
            "FlaxCLIPTextPreTrainedModel",
            "FlaxCLIPTextModelWithProjection",
            "FlaxCLIPVisionModel",
            "FlaxCLIPVisionPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "FlaxDistilBertForMaskedLM",
            "FlaxDistilBertForMultipleChoice",
            "FlaxDistilBertForQuestionAnswering",
            "FlaxDistilBertForSequenceClassification",
            "FlaxDistilBertForTokenClassification",
            "FlaxDistilBertModel",
            "FlaxDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "FlaxElectraForCausalLM",
            "FlaxElectraForMaskedLM",
            "FlaxElectraForMultipleChoice",
            "FlaxElectraForPreTraining",
            "FlaxElectraForQuestionAnswering",
            "FlaxElectraForSequenceClassification",
            "FlaxElectraForTokenClassification",
            "FlaxElectraModel",
            "FlaxElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("FlaxEncoderDecoderModel")
    _import_structure["models.gpt2"].extend(["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"])
    _import_structure["models.gpt_neo"].extend(
        ["FlaxGPTNeoForCausalLM", "FlaxGPTNeoModel", "FlaxGPTNeoPreTrainedModel"]
    )
    _import_structure["models.gptj"].extend(["FlaxGPTJForCausalLM", "FlaxGPTJModel", "FlaxGPTJPreTrainedModel"])
    _import_structure["models.longt5"].extend(
        ["FlaxLongT5ForConditionalGeneration", "FlaxLongT5Model", "FlaxLongT5PreTrainedModel"]
    )
    _import_structure["models.marian"].extend(
        [
            "FlaxMarianModel",
            "FlaxMarianMTModel",
            "FlaxMarianPreTrainedModel",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "FlaxMBartForConditionalGeneration",
            "FlaxMBartForQuestionAnswering",
            "FlaxMBartForSequenceClassification",
            "FlaxMBartModel",
            "FlaxMBartPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["FlaxMT5EncoderModel", "FlaxMT5ForConditionalGeneration", "FlaxMT5Model"])
    _import_structure["models.opt"].extend(
        [
            "FlaxOPTForCausalLM",
            "FlaxOPTModel",
            "FlaxOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "FlaxPegasusForConditionalGeneration",
            "FlaxPegasusModel",
            "FlaxPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        ["FlaxRegNetForImageClassification", "FlaxRegNetModel", "FlaxRegNetPreTrainedModel"]
    )
    _import_structure["models.resnet"].extend(
        ["FlaxResNetForImageClassification", "FlaxResNetModel", "FlaxResNetPreTrainedModel"]
    )
    _import_structure["models.roberta"].extend(
        [
            "FlaxRobertaForCausalLM",
            "FlaxRobertaForMaskedLM",
            "FlaxRobertaForMultipleChoice",
            "FlaxRobertaForQuestionAnswering",
            "FlaxRobertaForSequenceClassification",
            "FlaxRobertaForTokenClassification",
            "FlaxRobertaModel",
            "FlaxRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "FlaxRobertaPreLayerNormForCausalLM",
            "FlaxRobertaPreLayerNormForMaskedLM",
            "FlaxRobertaPreLayerNormForMultipleChoice",
            "FlaxRobertaPreLayerNormForQuestionAnswering",
            "FlaxRobertaPreLayerNormForSequenceClassification",
            "FlaxRobertaPreLayerNormForTokenClassification",
            "FlaxRobertaPreLayerNormModel",
            "FlaxRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "FlaxRoFormerForMaskedLM",
            "FlaxRoFormerForMultipleChoice",
            "FlaxRoFormerForQuestionAnswering",
            "FlaxRoFormerForSequenceClassification",
            "FlaxRoFormerForTokenClassification",
            "FlaxRoFormerModel",
            "FlaxRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].append("FlaxSpeechEncoderDecoderModel")
    _import_structure["models.t5"].extend(
        ["FlaxT5EncoderModel", "FlaxT5ForConditionalGeneration", "FlaxT5Model", "FlaxT5PreTrainedModel"]
    )
    _import_structure["models.vision_encoder_decoder"].append("FlaxVisionEncoderDecoderModel")
    _import_structure["models.vision_text_dual_encoder"].extend(["FlaxVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    _import_structure["models.wav2vec2"].extend(
        ["FlaxWav2Vec2ForCTC", "FlaxWav2Vec2ForPreTraining", "FlaxWav2Vec2Model", "FlaxWav2Vec2PreTrainedModel"]
    )
    _import_structure["models.whisper"].extend(
        [
            "FlaxWhisperForConditionalGeneration",
            "FlaxWhisperModel",
            "FlaxWhisperPreTrainedModel",
            "FlaxWhisperForAudioClassification",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "FlaxXGLMForCausalLM",
            "FlaxXGLMModel",
            "FlaxXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlaxXLMRobertaForMaskedLM",
            "FlaxXLMRobertaForMultipleChoice",
            "FlaxXLMRobertaForQuestionAnswering",
            "FlaxXLMRobertaForSequenceClassification",
            "FlaxXLMRobertaForTokenClassification",
            "FlaxXLMRobertaModel",
            "FlaxXLMRobertaForCausalLM",
            "FlaxXLMRobertaPreTrainedModel",
        ]
    )


# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    from .configuration_utils import PretrainedConfig

    # Data
    from .data import (
        DataProcessor,
        InputExample,
        InputFeatures,
        SingleSentenceClassificationProcessor,
        SquadExample,
        SquadFeatures,
        SquadV1Processor,
        SquadV2Processor,
        glue_compute_metrics,
        glue_convert_examples_to_features,
        glue_output_modes,
        glue_processors,
        glue_tasks_num_labels,
        squad_convert_examples_to_features,
        xnli_compute_metrics,
        xnli_output_modes,
        xnli_processors,
        xnli_tasks_num_labels,
    )
    from .data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    from .generation import GenerationConfig, TextIteratorStreamer, TextStreamer
    from .hf_argparser import HfArgumentParser

    # Integrations
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

    # Model Cards
    from .modelcard import ModelCard

    # TF 2.0 <=> PyTorch conversion utilities
    from .modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name,
        load_pytorch_checkpoint_in_tf2_model,
        load_pytorch_model_in_tf2_model,
        load_pytorch_weights_in_tf2_model,
        load_tf2_checkpoint_in_pytorch_model,
        load_tf2_model_in_pytorch_model,
        load_tf2_weights_in_pytorch_model,
    )
    from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
    from .models.align import (
        ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AlignConfig,
        AlignProcessor,
        AlignTextConfig,
        AlignVisionConfig,
    )
    from .models.altclip import (
        ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AltCLIPConfig,
        AltCLIPProcessor,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    from .models.audio_spectrogram_transformer import (
        AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ASTConfig,
    )
    from .models.auto import (
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CONFIG_MAPPING,
        FEATURE_EXTRACTOR_MAPPING,
        IMAGE_PROCESSOR_MAPPING,
        MODEL_NAMES_MAPPING,
        PROCESSOR_MAPPING,
        TOKENIZER_MAPPING,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )
    from .models.autoformer import (
        AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AutoformerConfig,
    )
    from .models.bark import (
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        BarkProcessor,
        BarkSemanticConfig,
    )
    from .models.bart import BartConfig, BartTokenizer
    from .models.beit import BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BeitConfig
    from .models.bert import (
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BasicTokenizer,
        BertConfig,
        BertTokenizer,
        WordpieceTokenizer,
    )
    from .models.bert_generation import BertGenerationConfig
    from .models.bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
    from .models.bertweet import BertweetTokenizer
    from .models.big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig
    from .models.bigbird_pegasus import BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdPegasusConfig
    from .models.biogpt import BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BioGptConfig, BioGptTokenizer
    from .models.bit import BIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BitConfig
    from .models.blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig, BlenderbotTokenizer
    from .models.blenderbot_small import (
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotSmallConfig,
        BlenderbotSmallTokenizer,
    )
    from .models.blip import (
        BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlipConfig,
        BlipProcessor,
        BlipTextConfig,
        BlipVisionConfig,
    )
    from .models.blip_2 import (
        BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Blip2Config,
        Blip2Processor,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    from .models.bloom import BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP, BloomConfig
    from .models.bridgetower import (
        BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BridgeTowerConfig,
        BridgeTowerProcessor,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from .models.bros import BROS_PRETRAINED_CONFIG_ARCHIVE_MAP, BrosConfig, BrosProcessor
    from .models.byt5 import ByT5Tokenizer
    from .models.camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
    from .models.canine import CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP, CanineConfig, CanineTokenizer
    from .models.chinese_clip import (
        CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ChineseCLIPConfig,
        ChineseCLIPProcessor,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    from .models.clap import (
        CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
        ClapAudioConfig,
        ClapConfig,
        ClapProcessor,
        ClapTextConfig,
    )
    from .models.clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPConfig,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTokenizer,
        CLIPVisionConfig,
    )
    from .models.clipseg import (
        CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPSegConfig,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegVisionConfig,
    )
    from .models.codegen import CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP, CodeGenConfig, CodeGenTokenizer
    from .models.conditional_detr import CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, ConditionalDetrConfig
    from .models.convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig, ConvBertTokenizer
    from .models.convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig
    from .models.convnextv2 import CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextV2Config
    from .models.cpmant import CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP, CpmAntConfig, CpmAntTokenizer
    from .models.ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig, CTRLTokenizer
    from .models.cvt import CVT_PRETRAINED_CONFIG_ARCHIVE_MAP, CvtConfig
    from .models.data2vec import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecAudioConfig,
        Data2VecTextConfig,
        Data2VecVisionConfig,
    )
    from .models.deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig, DebertaTokenizer
    from .models.deberta_v2 import DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaV2Config
    from .models.decision_transformer import (
        DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DecisionTransformerConfig,
    )
    from .models.deformable_detr import DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DeformableDetrConfig
    from .models.deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig
    from .models.deprecated.mctct import (
        MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTProcessor,
    )
    from .models.deprecated.mmbt import MMBTConfig
    from .models.deprecated.open_llama import OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenLlamaConfig
    from .models.deprecated.retribert import (
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RetriBertConfig,
        RetriBertTokenizer,
    )
    from .models.deprecated.tapex import TapexTokenizer
    from .models.deprecated.trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrajectoryTransformerConfig,
    )
    from .models.deprecated.van import VAN_PRETRAINED_CONFIG_ARCHIVE_MAP, VanConfig
    from .models.deta import DETA_PRETRAINED_CONFIG_ARCHIVE_MAP, DetaConfig
    from .models.detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig
    from .models.dinat import DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP, DinatConfig
    from .models.dinov2 import DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Dinov2Config
    from .models.distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig, DistilBertTokenizer
    from .models.donut import DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, DonutProcessor, DonutSwinConfig
    from .models.dpr import (
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPRConfig,
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )
    from .models.dpt import DPT_PRETRAINED_CONFIG_ARCHIVE_MAP, DPTConfig
    from .models.efficientformer import EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, EfficientFormerConfig
    from .models.efficientnet import EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP, EfficientNetConfig
    from .models.electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraTokenizer
    from .models.encodec import (
        ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EncodecConfig,
        EncodecFeatureExtractor,
    )
    from .models.encoder_decoder import EncoderDecoderConfig
    from .models.ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig
    from .models.ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig
    from .models.esm import ESM_PRETRAINED_CONFIG_ARCHIVE_MAP, EsmConfig, EsmTokenizer
    from .models.falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig
    from .models.flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertTokenizer
    from .models.flava import (
        FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )
    from .models.fnet import FNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig
    from .models.focalnet import FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FocalNetConfig
    from .models.fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig, FSMTTokenizer
    from .models.funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig, FunnelTokenizer
    from .models.fuyu import FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP, FuyuConfig
    from .models.git import GIT_PRETRAINED_CONFIG_ARCHIVE_MAP, GitConfig, GitProcessor, GitVisionConfig
    from .models.glpn import GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP, GLPNConfig
    from .models.gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2Tokenizer
    from .models.gpt_bigcode import GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTBigCodeConfig
    from .models.gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig
    from .models.gpt_neox import GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXConfig
    from .models.gpt_neox_japanese import GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXJapaneseConfig
    from .models.gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig
    from .models.gptsan_japanese import (
        GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTSanJapaneseConfig,
        GPTSanJapaneseTokenizer,
    )
    from .models.graphormer import GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, GraphormerConfig
    from .models.groupvit import (
        GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GroupViTConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )
    from .models.herbert import HerbertTokenizer
    from .models.hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig
    from .models.ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig
    from .models.idefics import (
        IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        IdeficsConfig,
    )
    from .models.imagegpt import IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig
    from .models.informer import INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, InformerConfig
    from .models.instructblip import (
        INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        InstructBlipConfig,
        InstructBlipProcessor,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    from .models.jukebox import (
        JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP,
        JukeboxConfig,
        JukeboxPriorConfig,
        JukeboxTokenizer,
        JukeboxVQVAEConfig,
    )
    from .models.kosmos2 import (
        KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Kosmos2Config,
        Kosmos2Processor,
    )
    from .models.layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMTokenizer
    from .models.layoutlmv2 import (
        LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv2Config,
        LayoutLMv2FeatureExtractor,
        LayoutLMv2ImageProcessor,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
    )
    from .models.layoutlmv3 import (
        LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv3Config,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ImageProcessor,
        LayoutLMv3Processor,
        LayoutLMv3Tokenizer,
    )
    from .models.layoutxlm import LayoutXLMProcessor
    from .models.led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig, LEDTokenizer
    from .models.levit import LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, LevitConfig
    from .models.lilt import LILT_PRETRAINED_CONFIG_ARCHIVE_MAP, LiltConfig
    from .models.llama import LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlamaConfig
    from .models.longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig, LongformerTokenizer
    from .models.longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config
    from .models.luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig, LukeTokenizer
    from .models.lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig, LxmertTokenizer
    from .models.m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config
    from .models.marian import MarianConfig
    from .models.markuplm import (
        MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMProcessor,
        MarkupLMTokenizer,
    )
    from .models.mask2former import MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, Mask2FormerConfig
    from .models.maskformer import MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, MaskFormerConfig, MaskFormerSwinConfig
    from .models.mbart import MBartConfig
    from .models.mega import MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP, MegaConfig
    from .models.megatron_bert import MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MegatronBertConfig
    from .models.mgp_str import MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP, MgpstrConfig, MgpstrProcessor, MgpstrTokenizer
    from .models.mistral import MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MistralConfig
    from .models.mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig, MobileBertTokenizer
    from .models.mobilenet_v1 import MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileNetV1Config
    from .models.mobilenet_v2 import MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileNetV2Config
    from .models.mobilevit import MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileViTConfig
    from .models.mobilevitv2 import MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileViTV2Config
    from .models.mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig, MPNetTokenizer
    from .models.mpt import MPT_PRETRAINED_CONFIG_ARCHIVE_MAP, MptConfig
    from .models.mra import MRA_PRETRAINED_CONFIG_ARCHIVE_MAP, MraConfig
    from .models.mt5 import MT5Config
    from .models.musicgen import (
        MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MusicgenConfig,
        MusicgenDecoderConfig,
    )
    from .models.mvp import MvpConfig, MvpTokenizer
    from .models.nat import NAT_PRETRAINED_CONFIG_ARCHIVE_MAP, NatConfig
    from .models.nezha import NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP, NezhaConfig
    from .models.nllb_moe import NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP, NllbMoeConfig
    from .models.nougat import NougatProcessor
    from .models.nystromformer import NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, NystromformerConfig
    from .models.oneformer import ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, OneFormerConfig, OneFormerProcessor
    from .models.openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig, OpenAIGPTTokenizer
    from .models.opt import OPTConfig
    from .models.owlv2 import (
        OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Owlv2Config,
        Owlv2Processor,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    from .models.owlvit import (
        OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OwlViTConfig,
        OwlViTProcessor,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    from .models.pegasus import PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusConfig, PegasusTokenizer
    from .models.pegasus_x import PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusXConfig
    from .models.perceiver import PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP, PerceiverConfig, PerceiverTokenizer
    from .models.persimmon import PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP, PersimmonConfig
    from .models.phobert import PhobertTokenizer
    from .models.pix2struct import (
        PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pix2StructConfig,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    from .models.plbart import PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP, PLBartConfig
    from .models.poolformer import POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, PoolFormerConfig
    from .models.pop2piano import (
        POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pop2PianoConfig,
    )
    from .models.prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig, ProphetNetTokenizer
    from .models.pvt import PVT_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtConfig
    from .models.qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig
    from .models.rag import RagConfig, RagRetriever, RagTokenizer
    from .models.realm import REALM_PRETRAINED_CONFIG_ARCHIVE_MAP, RealmConfig, RealmTokenizer
    from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
    from .models.regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig
    from .models.rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig
    from .models.resnet import RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetConfig
    from .models.roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaTokenizer
    from .models.roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
    )
    from .models.roc_bert import ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RoCBertConfig, RoCBertTokenizer
    from .models.roformer import ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RoFormerConfig, RoFormerTokenizer
    from .models.rwkv import RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP, RwkvConfig
    from .models.sam import (
        SAM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SamConfig,
        SamMaskDecoderConfig,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    from .models.seamless_m4t import (
        SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
    )
    from .models.segformer import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, SegformerConfig
    from .models.sew import SEW_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWConfig
    from .models.sew_d import SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWDConfig
    from .models.speech_encoder_decoder import SpeechEncoderDecoderConfig
    from .models.speech_to_text import (
        SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2TextConfig,
        Speech2TextProcessor,
    )
    from .models.speech_to_text_2 import (
        SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2Text2Config,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    from .models.speecht5 import (
        SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
    )
    from .models.splinter import SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP, SplinterConfig, SplinterTokenizer
    from .models.squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig, SqueezeBertTokenizer
    from .models.swiftformer import SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, SwiftFormerConfig
    from .models.swin import SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, SwinConfig
    from .models.swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig
    from .models.swinv2 import SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Swinv2Config
    from .models.switch_transformers import SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP, SwitchTransformersConfig
    from .models.t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
    from .models.table_transformer import TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, TableTransformerConfig
    from .models.tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig, TapasTokenizer
    from .models.time_series_transformer import (
        TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimeSeriesTransformerConfig,
    )
    from .models.timesformer import TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, TimesformerConfig
    from .models.timm_backbone import TimmBackboneConfig
    from .models.transfo_xl import (
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLTokenizer,
    )
    from .models.trocr import TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP, TrOCRConfig, TrOCRProcessor
    from .models.tvlt import TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP, TvltConfig, TvltFeatureExtractor, TvltProcessor
    from .models.umt5 import UMT5Config
    from .models.unispeech import UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechConfig
    from .models.unispeech_sat import UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechSatConfig
    from .models.upernet import UperNetConfig
    from .models.videomae import VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP, VideoMAEConfig
    from .models.vilt import (
        VILT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ViltConfig,
        ViltFeatureExtractor,
        ViltImageProcessor,
        ViltProcessor,
    )
    from .models.vision_encoder_decoder import VisionEncoderDecoderConfig
    from .models.vision_text_dual_encoder import VisionTextDualEncoderConfig, VisionTextDualEncoderProcessor
    from .models.visual_bert import VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, VisualBertConfig
    from .models.vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig
    from .models.vit_hybrid import VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTHybridConfig
    from .models.vit_mae import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMAEConfig
    from .models.vit_msn import VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMSNConfig
    from .models.vitdet import VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP, VitDetConfig
    from .models.vitmatte import VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP, VitMatteConfig
    from .models.vits import (
        VITS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VitsConfig,
        VitsTokenizer,
    )
    from .models.vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, VivitConfig
    from .models.wav2vec2 import (
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    from .models.wav2vec2_conformer import WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2ConformerConfig
    from .models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    from .models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    from .models.wavlm import WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP, WavLMConfig
    from .models.whisper import (
        WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from .models.x_clip import (
        XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XCLIPConfig,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from .models.xglm import XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XGLMConfig
    from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer
    from .models.xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig
    from .models.xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
    from .models.xlm_roberta_xl import XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaXLConfig
    from .models.xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
    from .models.xmod import XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP, XmodConfig
    from .models.yolos import YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP, YolosConfig
    from .models.yoso import YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig

    # Pipelines
    from .pipelines import (
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        Conversation,
        ConversationalPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageSegmentationPipeline,
        ImageToImagePipeline,
        ImageToTextPipeline,
        JsonPipelineDataFormat,
        MaskGenerationPipeline,
        NerPipeline,
        ObjectDetectionPipeline,
        PipedPipelineDataFormat,
        Pipeline,
        PipelineDataFormat,
        QuestionAnsweringPipeline,
        SummarizationPipeline,
        TableQuestionAnsweringPipeline,
        Text2TextGenerationPipeline,
        TextClassificationPipeline,
        TextGenerationPipeline,
        TextToAudioPipeline,
        TokenClassificationPipeline,
        TranslationPipeline,
        VideoClassificationPipeline,
        VisualQuestionAnsweringPipeline,
        ZeroShotAudioClassificationPipeline,
        ZeroShotClassificationPipeline,
        ZeroShotImageClassificationPipeline,
        ZeroShotObjectDetectionPipeline,
        pipeline,
    )
    from .processing_utils import ProcessorMixin

    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Tools
    from .tools import (
        Agent,
        AzureOpenAiAgent,
        HfAgent,
        LocalAgent,
        OpenAiAgent,
        PipelineTool,
        RemoteTool,
        Tool,
        launch_gradio_demo,
        load_tool,
    )

    # Trainer
    from .trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from .trainer_utils import EvalPrediction, IntervalStrategy, SchedulerType, enable_full_determinism, set_seed
    from .training_args import TrainingArguments
    from .training_args_seq2seq import Seq2SeqTrainingArguments
    from .training_args_tf import TFTrainingArguments

    # Files and general utilities
    from .utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_decord_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from .utils.quantization_config import AwqConfig, BitsAndBytesConfig, GPTQConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_sentencepiece_objects import *
    else:
        from .models.albert import AlbertTokenizer
        from .models.barthez import BarthezTokenizer
        from .models.bartpho import BartphoTokenizer
        from .models.bert_generation import BertGenerationTokenizer
        from .models.big_bird import BigBirdTokenizer
        from .models.camembert import CamembertTokenizer
        from .models.code_llama import CodeLlamaTokenizer
        from .models.cpm import CpmTokenizer
        from .models.deberta_v2 import DebertaV2Tokenizer
        from .models.ernie_m import ErnieMTokenizer
        from .models.fnet import FNetTokenizer
        from .models.gpt_sw3 import GPTSw3Tokenizer
        from .models.layoutxlm import LayoutXLMTokenizer
        from .models.llama import LlamaTokenizer
        from .models.m2m_100 import M2M100Tokenizer
        from .models.marian import MarianTokenizer
        from .models.mbart import MBart50Tokenizer, MBartTokenizer
        from .models.mluke import MLukeTokenizer
        from .models.mt5 import MT5Tokenizer
        from .models.nllb import NllbTokenizer
        from .models.pegasus import PegasusTokenizer
        from .models.plbart import PLBartTokenizer
        from .models.reformer import ReformerTokenizer
        from .models.rembert import RemBertTokenizer
        from .models.seamless_m4t import SeamlessM4TTokenizer
        from .models.speech_to_text import Speech2TextTokenizer
        from .models.speecht5 import SpeechT5Tokenizer
        from .models.t5 import T5Tokenizer
        from .models.xglm import XGLMTokenizer
        from .models.xlm_prophetnet import XLMProphetNetTokenizer
        from .models.xlm_roberta import XLMRobertaTokenizer
        from .models.xlnet import XLNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tokenizers_objects import *
    else:
        # Fast tokenizers imports
        from .models.albert import AlbertTokenizerFast
        from .models.bart import BartTokenizerFast
        from .models.barthez import BarthezTokenizerFast
        from .models.bert import BertTokenizerFast
        from .models.big_bird import BigBirdTokenizerFast
        from .models.blenderbot import BlenderbotTokenizerFast
        from .models.blenderbot_small import BlenderbotSmallTokenizerFast
        from .models.bloom import BloomTokenizerFast
        from .models.camembert import CamembertTokenizerFast
        from .models.clip import CLIPTokenizerFast
        from .models.code_llama import CodeLlamaTokenizerFast
        from .models.codegen import CodeGenTokenizerFast
        from .models.convbert import ConvBertTokenizerFast
        from .models.cpm import CpmTokenizerFast
        from .models.deberta import DebertaTokenizerFast
        from .models.deberta_v2 import DebertaV2TokenizerFast
        from .models.deprecated.retribert import RetriBertTokenizerFast
        from .models.distilbert import DistilBertTokenizerFast
        from .models.dpr import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast, DPRReaderTokenizerFast
        from .models.electra import ElectraTokenizerFast
        from .models.fnet import FNetTokenizerFast
        from .models.funnel import FunnelTokenizerFast
        from .models.gpt2 import GPT2TokenizerFast
        from .models.gpt_neox import GPTNeoXTokenizerFast
        from .models.gpt_neox_japanese import GPTNeoXJapaneseTokenizer
        from .models.herbert import HerbertTokenizerFast
        from .models.layoutlm import LayoutLMTokenizerFast
        from .models.layoutlmv2 import LayoutLMv2TokenizerFast
        from .models.layoutlmv3 import LayoutLMv3TokenizerFast
        from .models.layoutxlm import LayoutXLMTokenizerFast
        from .models.led import LEDTokenizerFast
        from .models.llama import LlamaTokenizerFast
        from .models.longformer import LongformerTokenizerFast
        from .models.lxmert import LxmertTokenizerFast
        from .models.markuplm import MarkupLMTokenizerFast
        from .models.mbart import MBartTokenizerFast
        from .models.mbart50 import MBart50TokenizerFast
        from .models.mobilebert import MobileBertTokenizerFast
        from .models.mpnet import MPNetTokenizerFast
        from .models.mt5 import MT5TokenizerFast
        from .models.mvp import MvpTokenizerFast
        from .models.nllb import NllbTokenizerFast
        from .models.nougat import NougatTokenizerFast
        from .models.openai import OpenAIGPTTokenizerFast
        from .models.pegasus import PegasusTokenizerFast
        from .models.realm import RealmTokenizerFast
        from .models.reformer import ReformerTokenizerFast
        from .models.rembert import RemBertTokenizerFast
        from .models.roberta import RobertaTokenizerFast
        from .models.roformer import RoFormerTokenizerFast
        from .models.seamless_m4t import SeamlessM4TTokenizerFast
        from .models.splinter import SplinterTokenizerFast
        from .models.squeezebert import SqueezeBertTokenizerFast
        from .models.t5 import T5TokenizerFast
        from .models.whisper import WhisperTokenizerFast
        from .models.xglm import XGLMTokenizerFast
        from .models.xlm_roberta import XLMRobertaTokenizerFast
        from .models.xlnet import XLNetTokenizerFast
        from .tokenization_utils_fast import PreTrainedTokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer

    try:
        if not is_speech_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_speech_objects import *
    else:
        from .models.audio_spectrogram_transformer import ASTFeatureExtractor
        from .models.speech_to_text import Speech2TextFeatureExtractor

    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tensorflow_text_objects import *
    else:
        from .models.bert import TFBertTokenizer

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_keras_nlp_objects import *
    else:
        from .models.gpt2 import TFGPT2Tokenizer

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_vision_objects import *
    else:
        from .image_processing_utils import ImageProcessingMixin
        from .image_utils import ImageFeatureExtractionMixin
        from .models.beit import BeitFeatureExtractor, BeitImageProcessor
        from .models.bit import BitImageProcessor
        from .models.blip import BlipImageProcessor
        from .models.bridgetower import BridgeTowerImageProcessor
        from .models.chinese_clip import ChineseCLIPFeatureExtractor, ChineseCLIPImageProcessor
        from .models.clip import CLIPFeatureExtractor, CLIPImageProcessor
        from .models.conditional_detr import ConditionalDetrFeatureExtractor, ConditionalDetrImageProcessor
        from .models.convnext import ConvNextFeatureExtractor, ConvNextImageProcessor
        from .models.deformable_detr import DeformableDetrFeatureExtractor, DeformableDetrImageProcessor
        from .models.deit import DeiTFeatureExtractor, DeiTImageProcessor
        from .models.deta import DetaImageProcessor
        from .models.detr import DetrFeatureExtractor, DetrImageProcessor
        from .models.donut import DonutFeatureExtractor, DonutImageProcessor
        from .models.dpt import DPTFeatureExtractor, DPTImageProcessor
        from .models.efficientformer import EfficientFormerImageProcessor
        from .models.efficientnet import EfficientNetImageProcessor
        from .models.flava import FlavaFeatureExtractor, FlavaImageProcessor, FlavaProcessor
        from .models.fuyu import FuyuImageProcessor, FuyuProcessor
        from .models.glpn import GLPNFeatureExtractor, GLPNImageProcessor
        from .models.idefics import IdeficsImageProcessor
        from .models.imagegpt import ImageGPTFeatureExtractor, ImageGPTImageProcessor
        from .models.layoutlmv2 import LayoutLMv2FeatureExtractor, LayoutLMv2ImageProcessor
        from .models.layoutlmv3 import LayoutLMv3FeatureExtractor, LayoutLMv3ImageProcessor
        from .models.levit import LevitFeatureExtractor, LevitImageProcessor
        from .models.mask2former import Mask2FormerImageProcessor
        from .models.maskformer import MaskFormerFeatureExtractor, MaskFormerImageProcessor
        from .models.mobilenet_v1 import MobileNetV1FeatureExtractor, MobileNetV1ImageProcessor
        from .models.mobilenet_v2 import MobileNetV2FeatureExtractor, MobileNetV2ImageProcessor
        from .models.mobilevit import MobileViTFeatureExtractor, MobileViTImageProcessor
        from .models.nougat import NougatImageProcessor
        from .models.oneformer import OneFormerImageProcessor
        from .models.owlv2 import Owlv2ImageProcessor
        from .models.owlvit import OwlViTFeatureExtractor, OwlViTImageProcessor
        from .models.perceiver import PerceiverFeatureExtractor, PerceiverImageProcessor
        from .models.pix2struct import Pix2StructImageProcessor
        from .models.poolformer import PoolFormerFeatureExtractor, PoolFormerImageProcessor
        from .models.pvt import PvtImageProcessor
        from .models.sam import SamImageProcessor
        from .models.segformer import SegformerFeatureExtractor, SegformerImageProcessor
        from .models.swin2sr import Swin2SRImageProcessor
        from .models.tvlt import TvltImageProcessor
        from .models.videomae import VideoMAEFeatureExtractor, VideoMAEImageProcessor
        from .models.vilt import ViltFeatureExtractor, ViltImageProcessor, ViltProcessor
        from .models.vit import ViTFeatureExtractor, ViTImageProcessor
        from .models.vit_hybrid import ViTHybridImageProcessor
        from .models.vitmatte import VitMatteImageProcessor
        from .models.vivit import VivitImageProcessor
        from .models.yolos import YolosFeatureExtractor, YolosImageProcessor

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:
        # Benchmarks
        from .benchmark.benchmark import PyTorchBenchmark
        from .benchmark.benchmark_args import PyTorchBenchmarkArguments
        from .data.datasets import (
            GlueDataset,
            GlueDataTrainingArguments,
            LineByLineTextDataset,
            LineByLineWithRefDataset,
            LineByLineWithSOPTextDataset,
            SquadDataset,
            SquadDataTrainingArguments,
            TextDataset,
            TextDatasetForNextSentencePrediction,
        )
        from .generation import (
            AlternatingCodebooksLogitsProcessor,
            BeamScorer,
            BeamSearchScorer,
            ClassifierFreeGuidanceLogitsProcessor,
            ConstrainedBeamSearchScorer,
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            ForceTokensLogitsProcessor,
            GenerationMixin,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MaxLengthCriteria,
            MaxTimeCriteria,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PhrasalConstraint,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            StoppingCriteria,
            StoppingCriteriaList,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
            top_k_top_p_filtering,
        )
        from .modeling_utils import PreTrainedModel
        from .models.albert import (
            ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            AlbertForMaskedLM,
            AlbertForMultipleChoice,
            AlbertForPreTraining,
            AlbertForQuestionAnswering,
            AlbertForSequenceClassification,
            AlbertForTokenClassification,
            AlbertModel,
            AlbertPreTrainedModel,
            load_tf_weights_in_albert,
        )
        from .models.align import (
            ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST,
            AlignModel,
            AlignPreTrainedModel,
            AlignTextModel,
            AlignVisionModel,
        )
        from .models.altclip import (
            ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            AltCLIPModel,
            AltCLIPPreTrainedModel,
            AltCLIPTextModel,
            AltCLIPVisionModel,
        )
        from .models.audio_spectrogram_transformer import (
            AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ASTForAudioClassification,
            ASTModel,
            ASTPreTrainedModel,
        )
        from .models.auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_XVECTOR_MAPPING,
            MODEL_FOR_BACKBONE_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
            MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
            MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
            MODEL_FOR_MASK_GENERATION_MAPPING,
            MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TEXT_ENCODING_MAPPING,
            MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
            MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoBackbone,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForAudioFrameClassification,
            AutoModelForAudioXVector,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForDepthEstimation,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForMaskedImageModeling,
            AutoModelForMaskedLM,
            AutoModelForMaskGeneration,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTextEncoding,
            AutoModelForTextToSpectrogram,
            AutoModelForTextToWaveform,
            AutoModelForTokenClassification,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForVision2Seq,
            AutoModelForVisualQuestionAnswering,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
            AutoModelWithLMHead,
        )
        from .models.autoformer import (
            AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            AutoformerForPrediction,
            AutoformerModel,
            AutoformerPreTrainedModel,
        )
        from .models.bark import (
            BARK_PRETRAINED_MODEL_ARCHIVE_LIST,
            BarkCausalModel,
            BarkCoarseModel,
            BarkFineModel,
            BarkModel,
            BarkPreTrainedModel,
            BarkSemanticModel,
        )
        from .models.bart import (
            BART_PRETRAINED_MODEL_ARCHIVE_LIST,
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPreTrainedModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )
        from .models.beit import (
            BEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BeitForImageClassification,
            BeitForMaskedImageModeling,
            BeitForSemanticSegmentation,
            BeitModel,
            BeitPreTrainedModel,
        )
        from .models.bert import (
            BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLayer,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )
        from .models.bert_generation import (
            BertGenerationDecoder,
            BertGenerationEncoder,
            BertGenerationPreTrainedModel,
            load_tf_weights_in_bert_generation,
        )
        from .models.big_bird import (
            BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigBirdForCausalLM,
            BigBirdForMaskedLM,
            BigBirdForMultipleChoice,
            BigBirdForPreTraining,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
            BigBirdLayer,
            BigBirdModel,
            BigBirdPreTrainedModel,
            load_tf_weights_in_big_bird,
        )
        from .models.bigbird_pegasus import (
            BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigBirdPegasusForCausalLM,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForQuestionAnswering,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusModel,
            BigBirdPegasusPreTrainedModel,
        )
        from .models.biogpt import (
            BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BioGptForCausalLM,
            BioGptForSequenceClassification,
            BioGptForTokenClassification,
            BioGptModel,
            BioGptPreTrainedModel,
        )
        from .models.bit import (
            BIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BitBackbone,
            BitForImageClassification,
            BitModel,
            BitPreTrainedModel,
        )
        from .models.blenderbot import (
            BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotSmallForCausalLM,
            BlenderbotSmallForConditionalGeneration,
            BlenderbotSmallModel,
            BlenderbotSmallPreTrainedModel,
        )
        from .models.blip import (
            BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlipForConditionalGeneration,
            BlipForImageTextRetrieval,
            BlipForQuestionAnswering,
            BlipModel,
            BlipPreTrainedModel,
            BlipTextModel,
            BlipVisionModel,
        )
        from .models.blip_2 import (
            BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Blip2ForConditionalGeneration,
            Blip2Model,
            Blip2PreTrainedModel,
            Blip2QFormerModel,
            Blip2VisionModel,
        )
        from .models.bloom import (
            BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST,
            BloomForCausalLM,
            BloomForQuestionAnswering,
            BloomForSequenceClassification,
            BloomForTokenClassification,
            BloomModel,
            BloomPreTrainedModel,
        )
        from .models.bridgetower import (
            BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST,
            BridgeTowerForContrastiveLearning,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerModel,
            BridgeTowerPreTrainedModel,
        )
        from .models.bros import (
            BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BrosForTokenClassification,
            BrosModel,
            BrosPreTrainedModel,
            BrosProcessor,
            BrosSpadeEEForTokenClassification,
            BrosSpadeELForTokenClassification,
        )
        from .models.camembert import (
            CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CamembertForCausalLM,
            CamembertForMaskedLM,
            CamembertForMultipleChoice,
            CamembertForQuestionAnswering,
            CamembertForSequenceClassification,
            CamembertForTokenClassification,
            CamembertModel,
            CamembertPreTrainedModel,
        )
        from .models.canine import (
            CANINE_PRETRAINED_MODEL_ARCHIVE_LIST,
            CanineForMultipleChoice,
            CanineForQuestionAnswering,
            CanineForSequenceClassification,
            CanineForTokenClassification,
            CanineLayer,
            CanineModel,
            CaninePreTrainedModel,
            load_tf_weights_in_canine,
        )
        from .models.chinese_clip import (
            CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ChineseCLIPModel,
            ChineseCLIPPreTrainedModel,
            ChineseCLIPTextModel,
            ChineseCLIPVisionModel,
        )
        from .models.clap import (
            CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ClapAudioModel,
            ClapAudioModelWithProjection,
            ClapFeatureExtractor,
            ClapModel,
            ClapPreTrainedModel,
            ClapTextModel,
            ClapTextModelWithProjection,
        )
        from .models.clip import (
            CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )
        from .models.clipseg import (
            CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLIPSegForImageSegmentation,
            CLIPSegModel,
            CLIPSegPreTrainedModel,
            CLIPSegTextModel,
            CLIPSegVisionModel,
        )
        from .models.codegen import (
            CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
            CodeGenForCausalLM,
            CodeGenModel,
            CodeGenPreTrainedModel,
        )
        from .models.conditional_detr import (
            CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConditionalDetrForObjectDetection,
            ConditionalDetrForSegmentation,
            ConditionalDetrModel,
            ConditionalDetrPreTrainedModel,
        )
        from .models.convbert import (
            CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvBertForMaskedLM,
            ConvBertForMultipleChoice,
            ConvBertForQuestionAnswering,
            ConvBertForSequenceClassification,
            ConvBertForTokenClassification,
            ConvBertLayer,
            ConvBertModel,
            ConvBertPreTrainedModel,
            load_tf_weights_in_convbert,
        )
        from .models.convnext import (
            CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextBackbone,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )
        from .models.convnextv2 import (
            CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )
        from .models.cpmant import (
            CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CpmAntForCausalLM,
            CpmAntModel,
            CpmAntPreTrainedModel,
        )
        from .models.ctrl import (
            CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            CTRLForSequenceClassification,
            CTRLLMHeadModel,
            CTRLModel,
            CTRLPreTrainedModel,
        )
        from .models.cvt import (
            CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CvtForImageClassification,
            CvtModel,
            CvtPreTrainedModel,
        )
        from .models.data2vec import (
            DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST,
            DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST,
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
            Data2VecVisionForImageClassification,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )
        from .models.deberta import (
            DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DebertaForMaskedLM,
            DebertaForQuestionAnswering,
            DebertaForSequenceClassification,
            DebertaForTokenClassification,
            DebertaModel,
            DebertaPreTrainedModel,
        )
        from .models.deberta_v2 import (
            DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            DebertaV2ForMaskedLM,
            DebertaV2ForMultipleChoice,
            DebertaV2ForQuestionAnswering,
            DebertaV2ForSequenceClassification,
            DebertaV2ForTokenClassification,
            DebertaV2Model,
            DebertaV2PreTrainedModel,
        )
        from .models.decision_transformer import (
            DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DecisionTransformerGPT2Model,
            DecisionTransformerGPT2PreTrainedModel,
            DecisionTransformerModel,
            DecisionTransformerPreTrainedModel,
        )
        from .models.deformable_detr import (
            DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )
        from .models.deit import (
            DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )
        from .models.deprecated.mctct import (
            MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MCTCTForCTC,
            MCTCTModel,
            MCTCTPreTrainedModel,
        )
        from .models.deprecated.mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
        from .models.deprecated.open_llama import (
            OpenLlamaForCausalLM,
            OpenLlamaForSequenceClassification,
            OpenLlamaModel,
            OpenLlamaPreTrainedModel,
        )
        from .models.deprecated.retribert import (
            RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RetriBertModel,
            RetriBertPreTrainedModel,
        )
        from .models.deprecated.trajectory_transformer import (
            TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TrajectoryTransformerModel,
            TrajectoryTransformerPreTrainedModel,
        )
        from .models.deprecated.van import (
            VAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            VanForImageClassification,
            VanModel,
            VanPreTrainedModel,
        )
        from .models.deta import (
            DETA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetaForObjectDetection,
            DetaModel,
            DetaPreTrainedModel,
        )
        from .models.detr import (
            DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )
        from .models.dinat import (
            DINAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DinatBackbone,
            DinatForImageClassification,
            DinatModel,
            DinatPreTrainedModel,
        )
        from .models.dinov2 import (
            DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Dinov2Backbone,
            Dinov2ForImageClassification,
            Dinov2Model,
            Dinov2PreTrainedModel,
        )
        from .models.distilbert import (
            DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
            DistilBertModel,
            DistilBertPreTrainedModel,
        )
        from .models.donut import DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST, DonutSwinModel, DonutSwinPreTrainedModel
        from .models.dpr import (
            DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPRContextEncoder,
            DPRPretrainedContextEncoder,
            DPRPreTrainedModel,
            DPRPretrainedQuestionEncoder,
            DPRPretrainedReader,
            DPRQuestionEncoder,
            DPRReader,
        )
        from .models.dpt import (
            DPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPTForDepthEstimation,
            DPTForSemanticSegmentation,
            DPTModel,
            DPTPreTrainedModel,
        )
        from .models.efficientformer import (
            EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            EfficientFormerForImageClassification,
            EfficientFormerForImageClassificationWithTeacher,
            EfficientFormerModel,
            EfficientFormerPreTrainedModel,
        )
        from .models.efficientnet import (
            EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            EfficientNetForImageClassification,
            EfficientNetModel,
            EfficientNetPreTrainedModel,
        )
        from .models.electra import (
            ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )
        from .models.encodec import (
            ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST,
            EncodecModel,
            EncodecPreTrainedModel,
        )
        from .models.encoder_decoder import EncoderDecoderModel
        from .models.ernie import (
            ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieForCausalLM,
            ErnieForMaskedLM,
            ErnieForMultipleChoice,
            ErnieForNextSentencePrediction,
            ErnieForPreTraining,
            ErnieForQuestionAnswering,
            ErnieForSequenceClassification,
            ErnieForTokenClassification,
            ErnieModel,
            ErniePreTrainedModel,
        )
        from .models.ernie_m import (
            ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieMForInformationExtraction,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMModel,
            ErnieMPreTrainedModel,
        )
        from .models.esm import (
            ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
            EsmFoldPreTrainedModel,
            EsmForMaskedLM,
            EsmForProteinFolding,
            EsmForSequenceClassification,
            EsmForTokenClassification,
            EsmModel,
            EsmPreTrainedModel,
        )
        from .models.falcon import (
            FALCON_PRETRAINED_MODEL_ARCHIVE_LIST,
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )
        from .models.flaubert import (
            FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaubertForMultipleChoice,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertModel,
            FlaubertPreTrainedModel,
            FlaubertWithLMHeadModel,
        )
        from .models.flava import (
            FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlavaForPreTraining,
            FlavaImageCodebook,
            FlavaImageModel,
            FlavaModel,
            FlavaMultimodalModel,
            FlavaPreTrainedModel,
            FlavaTextModel,
        )
        from .models.fnet import (
            FNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            FNetForMaskedLM,
            FNetForMultipleChoice,
            FNetForNextSentencePrediction,
            FNetForPreTraining,
            FNetForQuestionAnswering,
            FNetForSequenceClassification,
            FNetForTokenClassification,
            FNetLayer,
            FNetModel,
            FNetPreTrainedModel,
        )
        from .models.focalnet import (
            FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            FocalNetBackbone,
            FocalNetForImageClassification,
            FocalNetForMaskedImageModeling,
            FocalNetModel,
            FocalNetPreTrainedModel,
        )
        from .models.fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
        from .models.funnel import (
            FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
            FunnelBaseModel,
            FunnelForMaskedLM,
            FunnelForMultipleChoice,
            FunnelForPreTraining,
            FunnelForQuestionAnswering,
            FunnelForSequenceClassification,
            FunnelForTokenClassification,
            FunnelModel,
            FunnelPreTrainedModel,
            load_tf_weights_in_funnel,
        )
        from .models.fuyu import (
            FuyuForCausalLM,
            FuyuPreTrainedModel,
        )
        from .models.git import (
            GIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GitForCausalLM,
            GitModel,
            GitPreTrainedModel,
            GitVisionModel,
        )
        from .models.glpn import (
            GLPN_PRETRAINED_MODEL_ARCHIVE_LIST,
            GLPNForDepthEstimation,
            GLPNModel,
            GLPNPreTrainedModel,
        )
        from .models.gpt2 import (
            GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            GPT2Model,
            GPT2PreTrainedModel,
            load_tf_weights_in_gpt2,
        )
        from .models.gpt_bigcode import (
            GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTBigCodeForCausalLM,
            GPTBigCodeForSequenceClassification,
            GPTBigCodeForTokenClassification,
            GPTBigCodeModel,
            GPTBigCodePreTrainedModel,
        )
        from .models.gpt_neo import (
            GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
            GPTNeoModel,
            GPTNeoPreTrainedModel,
            load_tf_weights_in_gpt_neo,
        )
        from .models.gpt_neox import (
            GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
            GPTNeoXLayer,
            GPTNeoXModel,
            GPTNeoXPreTrainedModel,
        )
        from .models.gpt_neox_japanese import (
            GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXJapaneseForCausalLM,
            GPTNeoXJapaneseLayer,
            GPTNeoXJapaneseModel,
            GPTNeoXJapanesePreTrainedModel,
        )
        from .models.gptj import (
            GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )
        from .models.gptsan_japanese import (
            GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTSanJapaneseForConditionalGeneration,
            GPTSanJapaneseModel,
            GPTSanJapanesePreTrainedModel,
        )
        from .models.graphormer import (
            GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            GraphormerForGraphClassification,
            GraphormerModel,
            GraphormerPreTrainedModel,
        )
        from .models.groupvit import (
            GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            GroupViTModel,
            GroupViTPreTrainedModel,
            GroupViTTextModel,
            GroupViTVisionModel,
        )
        from .models.hubert import (
            HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            HubertForCTC,
            HubertForSequenceClassification,
            HubertModel,
            HubertPreTrainedModel,
        )
        from .models.ibert import (
            IBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            IBertForMaskedLM,
            IBertForMultipleChoice,
            IBertForQuestionAnswering,
            IBertForSequenceClassification,
            IBertForTokenClassification,
            IBertModel,
            IBertPreTrainedModel,
        )
        from .models.idefics import (
            IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST,
            IdeficsForVisionText2Text,
            IdeficsModel,
            IdeficsPreTrainedModel,
            IdeficsProcessor,
        )
        from .models.imagegpt import (
            IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ImageGPTForCausalImageModeling,
            ImageGPTForImageClassification,
            ImageGPTModel,
            ImageGPTPreTrainedModel,
            load_tf_weights_in_imagegpt,
        )
        from .models.informer import (
            INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            InformerForPrediction,
            InformerModel,
            InformerPreTrainedModel,
        )
        from .models.instructblip import (
            INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            InstructBlipForConditionalGeneration,
            InstructBlipPreTrainedModel,
            InstructBlipQFormerModel,
            InstructBlipVisionModel,
        )
        from .models.jukebox import (
            JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST,
            JukeboxModel,
            JukeboxPreTrainedModel,
            JukeboxPrior,
            JukeboxVQVAE,
        )
        from .models.kosmos2 import (
            KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Kosmos2ForConditionalGeneration,
            Kosmos2Model,
            Kosmos2PreTrainedModel,
        )
        from .models.layoutlm import (
            LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMForMaskedLM,
            LayoutLMForQuestionAnswering,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
        from .models.layoutlmv2 import (
            LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Model,
            LayoutLMv2PreTrainedModel,
        )
        from .models.layoutlmv3 import (
            LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv3ForQuestionAnswering,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Model,
            LayoutLMv3PreTrainedModel,
        )
        from .models.led import (
            LED_PRETRAINED_MODEL_ARCHIVE_LIST,
            LEDForConditionalGeneration,
            LEDForQuestionAnswering,
            LEDForSequenceClassification,
            LEDModel,
            LEDPreTrainedModel,
        )
        from .models.levit import (
            LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LevitForImageClassification,
            LevitForImageClassificationWithTeacher,
            LevitModel,
            LevitPreTrainedModel,
        )
        from .models.lilt import (
            LILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LiltForQuestionAnswering,
            LiltForSequenceClassification,
            LiltForTokenClassification,
            LiltModel,
            LiltPreTrainedModel,
        )
        from .models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel
        from .models.longformer import (
            LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongformerForMaskedLM,
            LongformerForMultipleChoice,
            LongformerForQuestionAnswering,
            LongformerForSequenceClassification,
            LongformerForTokenClassification,
            LongformerModel,
            LongformerPreTrainedModel,
            LongformerSelfAttention,
        )
        from .models.longt5 import (
            LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
        )
        from .models.luke import (
            LUKE_PRETRAINED_MODEL_ARCHIVE_LIST,
            LukeForEntityClassification,
            LukeForEntityPairClassification,
            LukeForEntitySpanClassification,
            LukeForMaskedLM,
            LukeForMultipleChoice,
            LukeForQuestionAnswering,
            LukeForSequenceClassification,
            LukeForTokenClassification,
            LukeModel,
            LukePreTrainedModel,
        )
        from .models.lxmert import (
            LxmertEncoder,
            LxmertForPreTraining,
            LxmertForQuestionAnswering,
            LxmertModel,
            LxmertPreTrainedModel,
            LxmertVisualFeatureEncoder,
            LxmertXLayer,
        )
        from .models.m2m_100 import (
            M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST,
            M2M100ForConditionalGeneration,
            M2M100Model,
            M2M100PreTrainedModel,
        )
        from .models.marian import MarianForCausalLM, MarianModel, MarianMTModel
        from .models.markuplm import (
            MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarkupLMForQuestionAnswering,
            MarkupLMForSequenceClassification,
            MarkupLMForTokenClassification,
            MarkupLMModel,
            MarkupLMPreTrainedModel,
        )
        from .models.mask2former import (
            MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
        from .models.maskformer import (
            MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
            MaskFormerSwinBackbone,
        )
        from .models.mbart import (
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )
        from .models.mega import (
            MEGA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MegaForCausalLM,
            MegaForMaskedLM,
            MegaForMultipleChoice,
            MegaForQuestionAnswering,
            MegaForSequenceClassification,
            MegaForTokenClassification,
            MegaModel,
            MegaPreTrainedModel,
        )
        from .models.megatron_bert import (
            MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MegatronBertForCausalLM,
            MegatronBertForMaskedLM,
            MegatronBertForMultipleChoice,
            MegatronBertForNextSentencePrediction,
            MegatronBertForPreTraining,
            MegatronBertForQuestionAnswering,
            MegatronBertForSequenceClassification,
            MegatronBertForTokenClassification,
            MegatronBertModel,
            MegatronBertPreTrainedModel,
        )
        from .models.mgp_str import (
            MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST,
            MgpstrForSceneTextRecognition,
            MgpstrModel,
            MgpstrPreTrainedModel,
        )
        from .models.mistral import (
            MistralForCausalLM,
            MistralForSequenceClassification,
            MistralModel,
            MistralPreTrainedModel,
        )
        from .models.mobilebert import (
            MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForPreTraining,
            MobileBertForQuestionAnswering,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
            MobileBertLayer,
            MobileBertModel,
            MobileBertPreTrainedModel,
            load_tf_weights_in_mobilebert,
        )
        from .models.mobilenet_v1 import (
            MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV1ForImageClassification,
            MobileNetV1Model,
            MobileNetV1PreTrainedModel,
            load_tf_weights_in_mobilenet_v1,
        )
        from .models.mobilenet_v2 import (
            MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )
        from .models.mobilevit import (
            MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileViTForImageClassification,
            MobileViTForSemanticSegmentation,
            MobileViTModel,
            MobileViTPreTrainedModel,
        )
        from .models.mobilevitv2 import (
            MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileViTV2ForImageClassification,
            MobileViTV2ForSemanticSegmentation,
            MobileViTV2Model,
            MobileViTV2PreTrainedModel,
        )
        from .models.mpnet import (
            MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            MPNetForMaskedLM,
            MPNetForMultipleChoice,
            MPNetForQuestionAnswering,
            MPNetForSequenceClassification,
            MPNetForTokenClassification,
            MPNetLayer,
            MPNetModel,
            MPNetPreTrainedModel,
        )
        from .models.mpt import (
            MPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MptForCausalLM,
            MptForQuestionAnswering,
            MptForSequenceClassification,
            MptForTokenClassification,
            MptModel,
            MptPreTrainedModel,
        )
        from .models.mra import (
            MRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            MraForMaskedLM,
            MraForMultipleChoice,
            MraForQuestionAnswering,
            MraForSequenceClassification,
            MraForTokenClassification,
            MraModel,
            MraPreTrainedModel,
        )
        from .models.mt5 import (
            MT5EncoderModel,
            MT5ForConditionalGeneration,
            MT5ForQuestionAnswering,
            MT5ForSequenceClassification,
            MT5Model,
            MT5PreTrainedModel,
        )
        from .models.musicgen import (
            MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
            MusicgenForCausalLM,
            MusicgenForConditionalGeneration,
            MusicgenModel,
            MusicgenPreTrainedModel,
            MusicgenProcessor,
        )
        from .models.mvp import (
            MVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            MvpForCausalLM,
            MvpForConditionalGeneration,
            MvpForQuestionAnswering,
            MvpForSequenceClassification,
            MvpModel,
            MvpPreTrainedModel,
        )
        from .models.nat import (
            NAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            NatBackbone,
            NatForImageClassification,
            NatModel,
            NatPreTrainedModel,
        )
        from .models.nezha import (
            NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST,
            NezhaForMaskedLM,
            NezhaForMultipleChoice,
            NezhaForNextSentencePrediction,
            NezhaForPreTraining,
            NezhaForQuestionAnswering,
            NezhaForSequenceClassification,
            NezhaForTokenClassification,
            NezhaModel,
            NezhaPreTrainedModel,
        )
        from .models.nllb_moe import (
            NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST,
            NllbMoeForConditionalGeneration,
            NllbMoeModel,
            NllbMoePreTrainedModel,
            NllbMoeSparseMLP,
            NllbMoeTop2Router,
        )
        from .models.nystromformer import (
            NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            NystromformerForMaskedLM,
            NystromformerForMultipleChoice,
            NystromformerForQuestionAnswering,
            NystromformerForSequenceClassification,
            NystromformerForTokenClassification,
            NystromformerLayer,
            NystromformerModel,
            NystromformerPreTrainedModel,
        )
        from .models.oneformer import (
            ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            OneFormerForUniversalSegmentation,
            OneFormerModel,
            OneFormerPreTrainedModel,
        )
        from .models.openai import (
            OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OpenAIGPTDoubleHeadsModel,
            OpenAIGPTForSequenceClassification,
            OpenAIGPTLMHeadModel,
            OpenAIGPTModel,
            OpenAIGPTPreTrainedModel,
            load_tf_weights_in_openai_gpt,
        )
        from .models.opt import (
            OPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )
        from .models.owlv2 import (
            OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Owlv2ForObjectDetection,
            Owlv2Model,
            Owlv2PreTrainedModel,
            Owlv2TextModel,
            Owlv2VisionModel,
        )
        from .models.owlvit import (
            OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OwlViTForObjectDetection,
            OwlViTModel,
            OwlViTPreTrainedModel,
            OwlViTTextModel,
            OwlViTVisionModel,
        )
        from .models.pegasus import (
            PegasusForCausalLM,
            PegasusForConditionalGeneration,
            PegasusModel,
            PegasusPreTrainedModel,
        )
        from .models.pegasus_x import (
            PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusXForConditionalGeneration,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )
        from .models.perceiver import (
            PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST,
            PerceiverForImageClassificationConvProcessing,
            PerceiverForImageClassificationFourier,
            PerceiverForImageClassificationLearned,
            PerceiverForMaskedLM,
            PerceiverForMultimodalAutoencoding,
            PerceiverForOpticalFlow,
            PerceiverForSequenceClassification,
            PerceiverLayer,
            PerceiverModel,
            PerceiverPreTrainedModel,
        )
        from .models.persimmon import (
            PersimmonForCausalLM,
            PersimmonForSequenceClassification,
            PersimmonModel,
            PersimmonPreTrainedModel,
        )
        from .models.pix2struct import (
            PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Pix2StructForConditionalGeneration,
            Pix2StructPreTrainedModel,
            Pix2StructTextModel,
            Pix2StructVisionModel,
        )
        from .models.plbart import (
            PLBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            PLBartForCausalLM,
            PLBartForConditionalGeneration,
            PLBartForSequenceClassification,
            PLBartModel,
            PLBartPreTrainedModel,
        )
        from .models.poolformer import (
            POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            PoolFormerForImageClassification,
            PoolFormerModel,
            PoolFormerPreTrainedModel,
        )
        from .models.pop2piano import (
            POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Pop2PianoForConditionalGeneration,
            Pop2PianoPreTrainedModel,
        )
        from .models.prophetnet import (
            PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            ProphetNetDecoder,
            ProphetNetEncoder,
            ProphetNetForCausalLM,
            ProphetNetForConditionalGeneration,
            ProphetNetModel,
            ProphetNetPreTrainedModel,
        )
        from .models.pvt import (
            PVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            PvtForImageClassification,
            PvtModel,
            PvtPreTrainedModel,
        )
        from .models.qdqbert import (
            QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLayer,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )
        from .models.rag import RagModel, RagPreTrainedModel, RagSequenceForGeneration, RagTokenForGeneration
        from .models.realm import (
            REALM_PRETRAINED_MODEL_ARCHIVE_LIST,
            RealmEmbedder,
            RealmForOpenQA,
            RealmKnowledgeAugEncoder,
            RealmPreTrainedModel,
            RealmReader,
            RealmRetriever,
            RealmScorer,
            load_tf_weights_in_realm,
        )
        from .models.reformer import (
            REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ReformerAttention,
            ReformerForMaskedLM,
            ReformerForQuestionAnswering,
            ReformerForSequenceClassification,
            ReformerLayer,
            ReformerModel,
            ReformerModelWithLMHead,
            ReformerPreTrainedModel,
        )
        from .models.regnet import (
            REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )
        from .models.rembert import (
            REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertLayer,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )
        from .models.resnet import (
            RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            ResNetBackbone,
            ResNetForImageClassification,
            ResNetModel,
            ResNetPreTrainedModel,
        )
        from .models.roberta import (
            ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaModel,
            RobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaPreLayerNormForCausalLM,
            RobertaPreLayerNormForMaskedLM,
            RobertaPreLayerNormForMultipleChoice,
            RobertaPreLayerNormForQuestionAnswering,
            RobertaPreLayerNormForSequenceClassification,
            RobertaPreLayerNormForTokenClassification,
            RobertaPreLayerNormModel,
            RobertaPreLayerNormPreTrainedModel,
        )
        from .models.roc_bert import (
            ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoCBertForCausalLM,
            RoCBertForMaskedLM,
            RoCBertForMultipleChoice,
            RoCBertForPreTraining,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertLayer,
            RoCBertModel,
            RoCBertPreTrainedModel,
            load_tf_weights_in_roc_bert,
        )
        from .models.roformer import (
            ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoFormerForCausalLM,
            RoFormerForMaskedLM,
            RoFormerForMultipleChoice,
            RoFormerForQuestionAnswering,
            RoFormerForSequenceClassification,
            RoFormerForTokenClassification,
            RoFormerLayer,
            RoFormerModel,
            RoFormerPreTrainedModel,
            load_tf_weights_in_roformer,
        )
        from .models.rwkv import (
            RWKV_PRETRAINED_MODEL_ARCHIVE_LIST,
            RwkvForCausalLM,
            RwkvModel,
            RwkvPreTrainedModel,
        )
        from .models.sam import (
            SAM_PRETRAINED_MODEL_ARCHIVE_LIST,
            SamModel,
            SamPreTrainedModel,
        )

        # PyTorch model imports
        from .models.seamless_m4t import (
            SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST,
            SeamlessM4TCodeHifiGan,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
            SeamlessM4THifiGan,
            SeamlessM4TModel,
            SeamlessM4TPreTrainedModel,
            SeamlessM4TTextToUnitForConditionalGeneration,
            SeamlessM4TTextToUnitModel,
        )
        from .models.segformer import (
            SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerLayer,
            SegformerModel,
            SegformerPreTrainedModel,
        )
        from .models.sew import (
            SEW_PRETRAINED_MODEL_ARCHIVE_LIST,
            SEWForCTC,
            SEWForSequenceClassification,
            SEWModel,
            SEWPreTrainedModel,
        )
        from .models.sew_d import (
            SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST,
            SEWDForCTC,
            SEWDForSequenceClassification,
            SEWDModel,
            SEWDPreTrainedModel,
        )
        from .models.speech_encoder_decoder import SpeechEncoderDecoderModel
        from .models.speech_to_text import (
            SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )
        from .models.speech_to_text_2 import Speech2Text2ForCausalLM, Speech2Text2PreTrainedModel
        from .models.speecht5 import (
            SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            SpeechT5ForSpeechToSpeech,
            SpeechT5ForSpeechToText,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Model,
            SpeechT5PreTrainedModel,
        )
        from .models.splinter import (
            SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SplinterForPreTraining,
            SplinterForQuestionAnswering,
            SplinterLayer,
            SplinterModel,
            SplinterPreTrainedModel,
        )
        from .models.squeezebert import (
            SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForQuestionAnswering,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
            SqueezeBertModel,
            SqueezeBertModule,
            SqueezeBertPreTrainedModel,
        )
        from .models.swiftformer import (
            SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwiftFormerForImageClassification,
            SwiftFormerModel,
            SwiftFormerPreTrainedModel,
        )
        from .models.swin import (
            SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )
        from .models.swin2sr import (
            SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )
        from .models.swinv2 import (
            SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swinv2ForImageClassification,
            Swinv2ForMaskedImageModeling,
            Swinv2Model,
            Swinv2PreTrainedModel,
        )
        from .models.switch_transformers import (
            SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )
        from .models.t5 import (
            T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5ForQuestionAnswering,
            T5ForSequenceClassification,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )
        from .models.table_transformer import (
            TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TableTransformerForObjectDetection,
            TableTransformerModel,
            TableTransformerPreTrainedModel,
        )
        from .models.tapas import (
            TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
            TapasModel,
            TapasPreTrainedModel,
            load_tf_weights_in_tapas,
        )
        from .models.time_series_transformer import (
            TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimeSeriesTransformerForPrediction,
            TimeSeriesTransformerModel,
            TimeSeriesTransformerPreTrainedModel,
        )
        from .models.timesformer import (
            TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TimesformerForVideoClassification,
            TimesformerModel,
            TimesformerPreTrainedModel,
        )
        from .models.timm_backbone import TimmBackbone
        from .models.transfo_xl import (
            TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )
        from .models.trocr import TROCR_PRETRAINED_MODEL_ARCHIVE_LIST, TrOCRForCausalLM, TrOCRPreTrainedModel
        from .models.tvlt import (
            TVLT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TvltForAudioVisualClassification,
            TvltForPreTraining,
            TvltModel,
            TvltPreTrainedModel,
        )
        from .models.umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5ForQuestionAnswering,
            UMT5ForSequenceClassification,
            UMT5Model,
            UMT5PreTrainedModel,
        )
        from .models.unispeech import (
            UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniSpeechForCTC,
            UniSpeechForPreTraining,
            UniSpeechForSequenceClassification,
            UniSpeechModel,
            UniSpeechPreTrainedModel,
        )
        from .models.unispeech_sat import (
            UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniSpeechSatForAudioFrameClassification,
            UniSpeechSatForCTC,
            UniSpeechSatForPreTraining,
            UniSpeechSatForSequenceClassification,
            UniSpeechSatForXVector,
            UniSpeechSatModel,
            UniSpeechSatPreTrainedModel,
        )
        from .models.upernet import UperNetForSemanticSegmentation, UperNetPreTrainedModel
        from .models.videomae import (
            VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VideoMAEForPreTraining,
            VideoMAEForVideoClassification,
            VideoMAEModel,
            VideoMAEPreTrainedModel,
        )
        from .models.vilt import (
            VILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltLayer,
            ViltModel,
            ViltPreTrainedModel,
        )
        from .models.vision_encoder_decoder import VisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import VisionTextDualEncoderModel
        from .models.visual_bert import (
            VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            VisualBertForMultipleChoice,
            VisualBertForPreTraining,
            VisualBertForQuestionAnswering,
            VisualBertForRegionToPhraseAlignment,
            VisualBertForVisualReasoning,
            VisualBertLayer,
            VisualBertModel,
            VisualBertPreTrainedModel,
        )
        from .models.vit import (
            VIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )
        from .models.vit_hybrid import (
            VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTHybridForImageClassification,
            ViTHybridModel,
            ViTHybridPreTrainedModel,
        )
        from .models.vit_mae import (
            VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMAEForPreTraining,
            ViTMAELayer,
            ViTMAEModel,
            ViTMAEPreTrainedModel,
        )
        from .models.vit_msn import (
            VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMSNForImageClassification,
            ViTMSNModel,
            ViTMSNPreTrainedModel,
        )
        from .models.vitdet import (
            VITDET_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitDetBackbone,
            VitDetModel,
            VitDetPreTrainedModel,
        )
        from .models.vitmatte import (
            VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitMatteForImageMatting,
            VitMattePreTrainedModel,
        )
        from .models.vits import (
            VITS_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitsModel,
            VitsPreTrainedModel,
        )
        from .models.vivit import (
            VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            VivitForVideoClassification,
            VivitModel,
            VivitPreTrainedModel,
        )
        from .models.wav2vec2 import (
            WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForCTC,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForXVector,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
        from .models.wav2vec2_conformer import (
            WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2ConformerForAudioFrameClassification,
            Wav2Vec2ConformerForCTC,
            Wav2Vec2ConformerForPreTraining,
            Wav2Vec2ConformerForSequenceClassification,
            Wav2Vec2ConformerForXVector,
            Wav2Vec2ConformerModel,
            Wav2Vec2ConformerPreTrainedModel,
        )
        from .models.wavlm import (
            WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            WavLMForAudioFrameClassification,
            WavLMForCTC,
            WavLMForSequenceClassification,
            WavLMForXVector,
            WavLMModel,
            WavLMPreTrainedModel,
        )
        from .models.whisper import (
            WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            WhisperForAudioClassification,
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperModel,
            WhisperPreTrainedModel,
        )
        from .models.x_clip import (
            XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            XCLIPModel,
            XCLIPPreTrainedModel,
            XCLIPTextModel,
            XCLIPVisionModel,
        )
        from .models.xglm import XGLM_PRETRAINED_MODEL_ARCHIVE_LIST, XGLMForCausalLM, XGLMModel, XGLMPreTrainedModel
        from .models.xlm import (
            XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMForMultipleChoice,
            XLMForQuestionAnswering,
            XLMForQuestionAnsweringSimple,
            XLMForSequenceClassification,
            XLMForTokenClassification,
            XLMModel,
            XLMPreTrainedModel,
            XLMWithLMHeadModel,
        )
        from .models.xlm_prophetnet import (
            XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMProphetNetDecoder,
            XLMProphetNetEncoder,
            XLMProphetNetForCausalLM,
            XLMProphetNetForConditionalGeneration,
            XLMProphetNetModel,
            XLMProphetNetPreTrainedModel,
        )
        from .models.xlm_roberta import (
            XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )
        from .models.xlm_roberta_xl import (
            XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaXLForCausalLM,
            XLMRobertaXLForMaskedLM,
            XLMRobertaXLForMultipleChoice,
            XLMRobertaXLForQuestionAnswering,
            XLMRobertaXLForSequenceClassification,
            XLMRobertaXLForTokenClassification,
            XLMRobertaXLModel,
            XLMRobertaXLPreTrainedModel,
        )
        from .models.xlnet import (
            XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLNetForMultipleChoice,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForSequenceClassification,
            XLNetForTokenClassification,
            XLNetLMHeadModel,
            XLNetModel,
            XLNetPreTrainedModel,
            load_tf_weights_in_xlnet,
        )
        from .models.xmod import (
            XMOD_PRETRAINED_MODEL_ARCHIVE_LIST,
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodModel,
            XmodPreTrainedModel,
        )
        from .models.yolos import (
            YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST,
            YolosForObjectDetection,
            YolosModel,
            YolosPreTrainedModel,
        )
        from .models.yoso import (
            YOSO_PRETRAINED_MODEL_ARCHIVE_LIST,
            YosoForMaskedLM,
            YosoForMultipleChoice,
            YosoForQuestionAnswering,
            YosoForSequenceClassification,
            YosoForTokenClassification,
            YosoLayer,
            YosoModel,
            YosoPreTrainedModel,
        )

        # Optimization
        from .optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_inverse_sqrt_schedule,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )
        from .pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

        # Trainer
        from .trainer import Trainer
        from .trainer_pt_utils import torch_distributed_zero_first
        from .trainer_seq2seq import Seq2SeqTrainer

    # TensorFlow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_tf_objects import *
    else:
        from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments

        # Benchmarks
        from .benchmark.benchmark_tf import TensorFlowBenchmark
        from .generation import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFGenerationMixin,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
            tf_top_k_top_p_filtering,
        )
        from .keras_callbacks import KerasMetricCallback, PushToHubCallback
        from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, shape_list

        # TensorFlow model imports
        from .models.albert import (
            TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFAlbertForMaskedLM,
            TFAlbertForMultipleChoice,
            TFAlbertForPreTraining,
            TFAlbertForQuestionAnswering,
            TFAlbertForSequenceClassification,
            TFAlbertForTokenClassification,
            TFAlbertMainLayer,
            TFAlbertModel,
            TFAlbertPreTrainedModel,
        )
        from .models.auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForAudioClassification,
            TFAutoModelForCausalLM,
            TFAutoModelForDocumentQuestionAnswering,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedImageModeling,
            TFAutoModelForMaskedLM,
            TFAutoModelForMaskGeneration,
            TFAutoModelForMultipleChoice,
            TFAutoModelForNextSentencePrediction,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSemanticSegmentation,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForSpeechSeq2Seq,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTextEncoding,
            TFAutoModelForTokenClassification,
            TFAutoModelForVision2Seq,
            TFAutoModelForZeroShotImageClassification,
            TFAutoModelWithLMHead,
        )
        from .models.bart import (
            TFBartForConditionalGeneration,
            TFBartForSequenceClassification,
            TFBartModel,
            TFBartPretrainedModel,
        )
        from .models.bert import (
            TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBertEmbeddings,
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )
        from .models.blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            TFBlenderbotSmallForConditionalGeneration,
            TFBlenderbotSmallModel,
            TFBlenderbotSmallPreTrainedModel,
        )
        from .models.blip import (
            TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBlipForConditionalGeneration,
            TFBlipForImageTextRetrieval,
            TFBlipForQuestionAnswering,
            TFBlipModel,
            TFBlipPreTrainedModel,
            TFBlipTextModel,
            TFBlipVisionModel,
        )
        from .models.camembert import (
            TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCamembertForCausalLM,
            TFCamembertForMaskedLM,
            TFCamembertForMultipleChoice,
            TFCamembertForQuestionAnswering,
            TFCamembertForSequenceClassification,
            TFCamembertForTokenClassification,
            TFCamembertModel,
            TFCamembertPreTrainedModel,
        )
        from .models.clip import (
            TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )
        from .models.convbert import (
            TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFConvBertForMaskedLM,
            TFConvBertForMultipleChoice,
            TFConvBertForQuestionAnswering,
            TFConvBertForSequenceClassification,
            TFConvBertForTokenClassification,
            TFConvBertLayer,
            TFConvBertModel,
            TFConvBertPreTrainedModel,
        )
        from .models.convnext import TFConvNextForImageClassification, TFConvNextModel, TFConvNextPreTrainedModel
        from .models.convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )
        from .models.ctrl import (
            TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCTRLForSequenceClassification,
            TFCTRLLMHeadModel,
            TFCTRLModel,
            TFCTRLPreTrainedModel,
        )
        from .models.cvt import (
            TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCvtForImageClassification,
            TFCvtModel,
            TFCvtPreTrainedModel,
        )
        from .models.data2vec import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )
        from .models.deberta import (
            TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDebertaForMaskedLM,
            TFDebertaForQuestionAnswering,
            TFDebertaForSequenceClassification,
            TFDebertaForTokenClassification,
            TFDebertaModel,
            TFDebertaPreTrainedModel,
        )
        from .models.deberta_v2 import (
            TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDebertaV2ForMaskedLM,
            TFDebertaV2ForMultipleChoice,
            TFDebertaV2ForQuestionAnswering,
            TFDebertaV2ForSequenceClassification,
            TFDebertaV2ForTokenClassification,
            TFDebertaV2Model,
            TFDebertaV2PreTrainedModel,
        )
        from .models.deit import (
            TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
            TFDeiTModel,
            TFDeiTPreTrainedModel,
        )
        from .models.distilbert import (
            TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDistilBertForMaskedLM,
            TFDistilBertForMultipleChoice,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertMainLayer,
            TFDistilBertModel,
            TFDistilBertPreTrainedModel,
        )
        from .models.dpr import (
            TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDPRContextEncoder,
            TFDPRPretrainedContextEncoder,
            TFDPRPretrainedQuestionEncoder,
            TFDPRPretrainedReader,
            TFDPRQuestionEncoder,
            TFDPRReader,
        )
        from .models.efficientformer import (
            TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEfficientFormerForImageClassification,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerModel,
            TFEfficientFormerPreTrainedModel,
        )
        from .models.electra import (
            TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )
        from .models.encoder_decoder import TFEncoderDecoderModel
        from .models.esm import (
            ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
            TFEsmModel,
            TFEsmPreTrainedModel,
        )
        from .models.flaubert import (
            TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFFlaubertForMultipleChoice,
            TFFlaubertForQuestionAnsweringSimple,
            TFFlaubertForSequenceClassification,
            TFFlaubertForTokenClassification,
            TFFlaubertModel,
            TFFlaubertPreTrainedModel,
            TFFlaubertWithLMHeadModel,
        )
        from .models.funnel import (
            TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFFunnelBaseModel,
            TFFunnelForMaskedLM,
            TFFunnelForMultipleChoice,
            TFFunnelForPreTraining,
            TFFunnelForQuestionAnswering,
            TFFunnelForSequenceClassification,
            TFFunnelForTokenClassification,
            TFFunnelModel,
            TFFunnelPreTrainedModel,
        )
        from .models.gpt2 import (
            TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )
        from .models.gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )
        from .models.groupvit import (
            TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGroupViTModel,
            TFGroupViTPreTrainedModel,
            TFGroupViTTextModel,
            TFGroupViTVisionModel,
        )
        from .models.hubert import (
            TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFHubertForCTC,
            TFHubertModel,
            TFHubertPreTrainedModel,
        )
        from .models.layoutlm import (
            TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMForMaskedLM,
            TFLayoutLMForQuestionAnswering,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )
        from .models.layoutlmv3 import (
            TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
            TFLayoutLMv3Model,
            TFLayoutLMv3PreTrainedModel,
        )
        from .models.led import TFLEDForConditionalGeneration, TFLEDModel, TFLEDPreTrainedModel
        from .models.longformer import (
            TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLongformerForMaskedLM,
            TFLongformerForMultipleChoice,
            TFLongformerForQuestionAnswering,
            TFLongformerForSequenceClassification,
            TFLongformerForTokenClassification,
            TFLongformerModel,
            TFLongformerPreTrainedModel,
            TFLongformerSelfAttention,
        )
        from .models.lxmert import (
            TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLxmertForPreTraining,
            TFLxmertMainLayer,
            TFLxmertModel,
            TFLxmertPreTrainedModel,
            TFLxmertVisualFeatureEncoder,
        )
        from .models.marian import TFMarianModel, TFMarianMTModel, TFMarianPreTrainedModel
        from .models.mbart import TFMBartForConditionalGeneration, TFMBartModel, TFMBartPreTrainedModel
        from .models.mobilebert import (
            TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMobileBertForMaskedLM,
            TFMobileBertForMultipleChoice,
            TFMobileBertForNextSentencePrediction,
            TFMobileBertForPreTraining,
            TFMobileBertForQuestionAnswering,
            TFMobileBertForSequenceClassification,
            TFMobileBertForTokenClassification,
            TFMobileBertMainLayer,
            TFMobileBertModel,
            TFMobileBertPreTrainedModel,
        )
        from .models.mobilevit import (
            TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMobileViTForImageClassification,
            TFMobileViTForSemanticSegmentation,
            TFMobileViTModel,
            TFMobileViTPreTrainedModel,
        )
        from .models.mpnet import (
            TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMPNetForMaskedLM,
            TFMPNetForMultipleChoice,
            TFMPNetForQuestionAnswering,
            TFMPNetForSequenceClassification,
            TFMPNetForTokenClassification,
            TFMPNetMainLayer,
            TFMPNetModel,
            TFMPNetPreTrainedModel,
        )
        from .models.mt5 import TFMT5EncoderModel, TFMT5ForConditionalGeneration, TFMT5Model
        from .models.openai import (
            TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFOpenAIGPTDoubleHeadsModel,
            TFOpenAIGPTForSequenceClassification,
            TFOpenAIGPTLMHeadModel,
            TFOpenAIGPTMainLayer,
            TFOpenAIGPTModel,
            TFOpenAIGPTPreTrainedModel,
        )
        from .models.opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel
        from .models.pegasus import TFPegasusForConditionalGeneration, TFPegasusModel, TFPegasusPreTrainedModel
        from .models.rag import TFRagModel, TFRagPreTrainedModel, TFRagSequenceForGeneration, TFRagTokenForGeneration
        from .models.regnet import (
            TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )
        from .models.rembert import (
            TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertLayer,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )
        from .models.resnet import (
            TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFResNetForImageClassification,
            TFResNetModel,
            TFResNetPreTrainedModel,
        )
        from .models.roberta import (
            TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaForCausalLM,
            TFRobertaForMaskedLM,
            TFRobertaForMultipleChoice,
            TFRobertaForQuestionAnswering,
            TFRobertaForSequenceClassification,
            TFRobertaForTokenClassification,
            TFRobertaMainLayer,
            TFRobertaModel,
            TFRobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaPreLayerNormForCausalLM,
            TFRobertaPreLayerNormForMaskedLM,
            TFRobertaPreLayerNormForMultipleChoice,
            TFRobertaPreLayerNormForQuestionAnswering,
            TFRobertaPreLayerNormForSequenceClassification,
            TFRobertaPreLayerNormForTokenClassification,
            TFRobertaPreLayerNormMainLayer,
            TFRobertaPreLayerNormModel,
            TFRobertaPreLayerNormPreTrainedModel,
        )
        from .models.roformer import (
            TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRoFormerForCausalLM,
            TFRoFormerForMaskedLM,
            TFRoFormerForMultipleChoice,
            TFRoFormerForQuestionAnswering,
            TFRoFormerForSequenceClassification,
            TFRoFormerForTokenClassification,
            TFRoFormerLayer,
            TFRoFormerModel,
            TFRoFormerPreTrainedModel,
        )
        from .models.sam import (
            TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSamModel,
            TFSamPreTrainedModel,
        )
        from .models.segformer import (
            TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )
        from .models.speech_to_text import (
            TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSpeech2TextForConditionalGeneration,
            TFSpeech2TextModel,
            TFSpeech2TextPreTrainedModel,
        )
        from .models.swin import (
            TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSwinForImageClassification,
            TFSwinForMaskedImageModeling,
            TFSwinModel,
            TFSwinPreTrainedModel,
        )
        from .models.t5 import (
            TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )
        from .models.tapas import (
            TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFTapasForMaskedLM,
            TFTapasForQuestionAnswering,
            TFTapasForSequenceClassification,
            TFTapasModel,
            TFTapasPreTrainedModel,
        )
        from .models.transfo_xl import (
            TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFAdaptiveEmbedding,
            TFTransfoXLForSequenceClassification,
            TFTransfoXLLMHeadModel,
            TFTransfoXLMainLayer,
            TFTransfoXLModel,
            TFTransfoXLPreTrainedModel,
        )
        from .models.vision_encoder_decoder import TFVisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import TFVisionTextDualEncoderModel
        from .models.vit import TFViTForImageClassification, TFViTModel, TFViTPreTrainedModel
        from .models.vit_mae import TFViTMAEForPreTraining, TFViTMAEModel, TFViTMAEPreTrainedModel
        from .models.wav2vec2 import (
            TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWav2Vec2ForCTC,
            TFWav2Vec2ForSequenceClassification,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )
        from .models.whisper import (
            TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWhisperForConditionalGeneration,
            TFWhisperModel,
            TFWhisperPreTrainedModel,
        )
        from .models.xglm import (
            TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXGLMForCausalLM,
            TFXGLMModel,
            TFXGLMPreTrainedModel,
        )
        from .models.xlm import (
            TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLMForMultipleChoice,
            TFXLMForQuestionAnsweringSimple,
            TFXLMForSequenceClassification,
            TFXLMForTokenClassification,
            TFXLMMainLayer,
            TFXLMModel,
            TFXLMPreTrainedModel,
            TFXLMWithLMHeadModel,
        )
        from .models.xlm_roberta import (
            TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLMRobertaForCausalLM,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
            TFXLMRobertaPreTrainedModel,
        )
        from .models.xlnet import (
            TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLNetForMultipleChoice,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetLMHeadModel,
            TFXLNetMainLayer,
            TFXLNetModel,
            TFXLNetPreTrainedModel,
        )

        # Optimization
        from .optimization_tf import AdamWeightDecay, GradientAccumulator, WarmUp, create_optimizer

        # Trainer
        from .trainer_tf import TFTrainer

    try:
        if not (
            is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
            and is_torch_available()
            and is_pretty_midi_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects import *
    else:
        from .models.pop2piano import Pop2PianoFeatureExtractor, Pop2PianoProcessor, Pop2PianoTokenizer

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_flax_objects import *
    else:
        from .generation import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxGenerationMixin,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from .modeling_flax_utils import FlaxPreTrainedModel

        # Flax model imports
        from .models.albert import (
            FlaxAlbertForMaskedLM,
            FlaxAlbertForMultipleChoice,
            FlaxAlbertForPreTraining,
            FlaxAlbertForQuestionAnswering,
            FlaxAlbertForSequenceClassification,
            FlaxAlbertForTokenClassification,
            FlaxAlbertModel,
            FlaxAlbertPreTrainedModel,
        )
        from .models.auto import (
            FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
            FLAX_MODEL_MAPPING,
            FlaxAutoModel,
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForImageClassification,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForMultipleChoice,
            FlaxAutoModelForNextSentencePrediction,
            FlaxAutoModelForPreTraining,
            FlaxAutoModelForQuestionAnswering,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForSequenceClassification,
            FlaxAutoModelForSpeechSeq2Seq,
            FlaxAutoModelForTokenClassification,
            FlaxAutoModelForVision2Seq,
        )
        from .models.bart import (
            FlaxBartDecoderPreTrainedModel,
            FlaxBartForCausalLM,
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
        )
        from .models.beit import (
            FlaxBeitForImageClassification,
            FlaxBeitForMaskedImageModeling,
            FlaxBeitModel,
            FlaxBeitPreTrainedModel,
        )
        from .models.bert import (
            FlaxBertForCausalLM,
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )
        from .models.big_bird import (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
            FlaxBigBirdModel,
            FlaxBigBirdPreTrainedModel,
        )
        from .models.blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            FlaxBlenderbotSmallForConditionalGeneration,
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallPreTrainedModel,
        )
        from .models.bloom import FlaxBloomForCausalLM, FlaxBloomModel, FlaxBloomPreTrainedModel
        from .models.clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )
        from .models.distilbert import (
            FlaxDistilBertForMaskedLM,
            FlaxDistilBertForMultipleChoice,
            FlaxDistilBertForQuestionAnswering,
            FlaxDistilBertForSequenceClassification,
            FlaxDistilBertForTokenClassification,
            FlaxDistilBertModel,
            FlaxDistilBertPreTrainedModel,
        )
        from .models.electra import (
            FlaxElectraForCausalLM,
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )
        from .models.encoder_decoder import FlaxEncoderDecoderModel
        from .models.gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model, FlaxGPT2PreTrainedModel
        from .models.gpt_neo import FlaxGPTNeoForCausalLM, FlaxGPTNeoModel, FlaxGPTNeoPreTrainedModel
        from .models.gptj import FlaxGPTJForCausalLM, FlaxGPTJModel, FlaxGPTJPreTrainedModel
        from .models.longt5 import FlaxLongT5ForConditionalGeneration, FlaxLongT5Model, FlaxLongT5PreTrainedModel
        from .models.marian import FlaxMarianModel, FlaxMarianMTModel, FlaxMarianPreTrainedModel
        from .models.mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )
        from .models.mt5 import FlaxMT5EncoderModel, FlaxMT5ForConditionalGeneration, FlaxMT5Model
        from .models.opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel
        from .models.pegasus import FlaxPegasusForConditionalGeneration, FlaxPegasusModel, FlaxPegasusPreTrainedModel
        from .models.regnet import FlaxRegNetForImageClassification, FlaxRegNetModel, FlaxRegNetPreTrainedModel
        from .models.resnet import FlaxResNetForImageClassification, FlaxResNetModel, FlaxResNetPreTrainedModel
        from .models.roberta import (
            FlaxRobertaForCausalLM,
            FlaxRobertaForMaskedLM,
            FlaxRobertaForMultipleChoice,
            FlaxRobertaForQuestionAnswering,
            FlaxRobertaForSequenceClassification,
            FlaxRobertaForTokenClassification,
            FlaxRobertaModel,
            FlaxRobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormPreTrainedModel,
        )
        from .models.roformer import (
            FlaxRoFormerForMaskedLM,
            FlaxRoFormerForMultipleChoice,
            FlaxRoFormerForQuestionAnswering,
            FlaxRoFormerForSequenceClassification,
            FlaxRoFormerForTokenClassification,
            FlaxRoFormerModel,
            FlaxRoFormerPreTrainedModel,
        )
        from .models.speech_encoder_decoder import FlaxSpeechEncoderDecoderModel
        from .models.t5 import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration, FlaxT5Model, FlaxT5PreTrainedModel
        from .models.vision_encoder_decoder import FlaxVisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import FlaxVisionTextDualEncoderModel
        from .models.vit import FlaxViTForImageClassification, FlaxViTModel, FlaxViTPreTrainedModel
        from .models.wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )
        from .models.whisper import (
            FlaxWhisperForAudioClassification,
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
            FlaxWhisperPreTrainedModel,
        )
        from .models.xglm import FlaxXGLMForCausalLM, FlaxXGLMModel, FlaxXGLMPreTrainedModel
        from .models.xlm_roberta import (
            FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaxXLMRobertaForCausalLM,
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
            FlaxXLMRobertaPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
