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

__version__ = "4.52.0.dev0"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
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
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "agents": [
        "Agent",
        "CodeAgent",
        "HfApiEngine",
        "ManagedAgent",
        "PipelineTool",
        "ReactAgent",
        "ReactCodeAgent",
        "ReactJsonAgent",
        "Tool",
        "Toolbox",
        "ToolCollection",
        "TransformersEngine",
        "launch_gradio_demo",
        "load_tool",
        "stream_to_gradio",
        "tool",
    ],
    "audio_utils": [],
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
        "DataCollatorForMultipleChoice",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithFlattening",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "AsyncTextIteratorStreamer",
        "CompileConfig",
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_dvclive_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_swanlab_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Losses
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    # Models
    "models": [],
    "models.albert": ["AlbertConfig"],
    "models.align": [
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "models.altclip": [
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "models.aria": [
        "AriaConfig",
        "AriaProcessor",
        "AriaTextConfig",
    ],
    "models.audio_spectrogram_transformer": [
        "ASTConfig",
        "ASTFeatureExtractor",
    ],
    "models.auto": [
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
    "models.autoformer": ["AutoformerConfig"],
    "models.aya_vision": ["AyaVisionConfig", "AyaVisionProcessor"],
    "models.bamba": ["BambaConfig"],
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
    "models.beit": ["BeitConfig"],
    "models.bert": [
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": [
        "BertJapaneseTokenizer",
        "CharacterTokenizer",
        "MecabTokenizer",
    ],
    "models.bertweet": ["BertweetTokenizer"],
    "models.big_bird": ["BigBirdConfig"],
    "models.bigbird_pegasus": ["BigBirdPegasusConfig"],
    "models.biogpt": [
        "BioGptConfig",
        "BioGptTokenizer",
    ],
    "models.bit": ["BitConfig"],
    "models.blenderbot": [
        "BlenderbotConfig",
        "BlenderbotTokenizer",
    ],
    "models.blenderbot_small": [
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.blip": [
        "BlipConfig",
        "BlipProcessor",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "models.blip_2": [
        "Blip2Config",
        "Blip2Processor",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "models.bloom": ["BloomConfig"],
    "models.bridgetower": [
        "BridgeTowerConfig",
        "BridgeTowerProcessor",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "models.bros": [
        "BrosConfig",
        "BrosProcessor",
    ],
    "models.byt5": ["ByT5Tokenizer"],
    "models.camembert": ["CamembertConfig"],
    "models.canine": [
        "CanineConfig",
        "CanineTokenizer",
    ],
    "models.chameleon": [
        "ChameleonConfig",
        "ChameleonProcessor",
        "ChameleonVQVAEConfig",
    ],
    "models.chinese_clip": [
        "ChineseCLIPConfig",
        "ChineseCLIPProcessor",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "models.clap": [
        "ClapAudioConfig",
        "ClapConfig",
        "ClapProcessor",
        "ClapTextConfig",
    ],
    "models.clip": [
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.clvp": [
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
        "ClvpFeatureExtractor",
        "ClvpProcessor",
        "ClvpTokenizer",
    ],
    "models.code_llama": [],
    "models.codegen": [
        "CodeGenConfig",
        "CodeGenTokenizer",
    ],
    "models.cohere": ["CohereConfig"],
    "models.cohere2": ["Cohere2Config"],
    "models.colpali": [
        "ColPaliConfig",
        "ColPaliProcessor",
    ],
    "models.conditional_detr": ["ConditionalDetrConfig"],
    "models.convbert": [
        "ConvBertConfig",
        "ConvBertTokenizer",
    ],
    "models.convnext": ["ConvNextConfig"],
    "models.convnextv2": ["ConvNextV2Config"],
    "models.cpm": [],
    "models.cpmant": [
        "CpmAntConfig",
        "CpmAntTokenizer",
    ],
    "models.ctrl": [
        "CTRLConfig",
        "CTRLTokenizer",
    ],
    "models.cvt": ["CvtConfig"],
    "models.dab_detr": ["DabDetrConfig"],
    "models.dac": ["DacConfig", "DacFeatureExtractor"],
    "models.data2vec": [
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.dbrx": ["DbrxConfig"],
    "models.deberta": [
        "DebertaConfig",
        "DebertaTokenizer",
    ],
    "models.deberta_v2": ["DebertaV2Config"],
    "models.decision_transformer": ["DecisionTransformerConfig"],
    "models.deepseek_v3": ["DeepseekV3Config"],
    "models.deformable_detr": ["DeformableDetrConfig"],
    "models.deit": ["DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.deta": ["DetaConfig"],
    "models.deprecated.efficientformer": ["EfficientFormerConfig"],
    "models.deprecated.ernie_m": ["ErnieMConfig"],
    "models.deprecated.gptsan_japanese": [
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    "models.deprecated.graphormer": ["GraphormerConfig"],
    "models.deprecated.jukebox": [
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxTokenizer",
        "JukeboxVQVAEConfig",
    ],
    "models.deprecated.mctct": [
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mega": ["MegaConfig"],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.nat": ["NatConfig"],
    "models.deprecated.nezha": ["NezhaConfig"],
    "models.deprecated.open_llama": ["OpenLlamaConfig"],
    "models.deprecated.qdqbert": ["QDQBertConfig"],
    "models.deprecated.realm": [
        "RealmConfig",
        "RealmTokenizer",
    ],
    "models.deprecated.retribert": [
        "RetriBertConfig",
        "RetriBertTokenizer",
    ],
    "models.deprecated.speech_to_text_2": [
        "Speech2Text2Config",
        "Speech2Text2Processor",
        "Speech2Text2Tokenizer",
    ],
    "models.deprecated.tapex": ["TapexTokenizer"],
    "models.deprecated.trajectory_transformer": ["TrajectoryTransformerConfig"],
    "models.deprecated.transfo_xl": [
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.deprecated.tvlt": [
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    "models.deprecated.van": ["VanConfig"],
    "models.deprecated.vit_hybrid": ["ViTHybridConfig"],
    "models.deprecated.xlm_prophetnet": ["XLMProphetNetConfig"],
    "models.depth_anything": ["DepthAnythingConfig"],
    "models.depth_pro": ["DepthProConfig"],
    "models.detr": ["DetrConfig"],
    "models.dialogpt": [],
    "models.diffllama": ["DiffLlamaConfig"],
    "models.dinat": ["DinatConfig"],
    "models.dinov2": ["Dinov2Config"],
    "models.dinov2_with_registers": ["Dinov2WithRegistersConfig"],
    "models.distilbert": [
        "DistilBertConfig",
        "DistilBertTokenizer",
    ],
    "models.dit": [],
    "models.donut": [
        "DonutProcessor",
        "DonutSwinConfig",
    ],
    "models.dpr": [
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.dpt": ["DPTConfig"],
    "models.efficientnet": ["EfficientNetConfig"],
    "models.electra": [
        "ElectraConfig",
        "ElectraTokenizer",
    ],
    "models.emu3": [
        "Emu3Config",
        "Emu3Processor",
        "Emu3TextConfig",
        "Emu3VQVAEConfig",
    ],
    "models.encodec": [
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.ernie": ["ErnieConfig"],
    "models.esm": ["EsmConfig", "EsmTokenizer"],
    "models.falcon": ["FalconConfig"],
    "models.falcon_mamba": ["FalconMambaConfig"],
    "models.fastspeech2_conformer": [
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerTokenizer",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "models.flaubert": ["FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    "models.fnet": ["FNetConfig"],
    "models.focalnet": ["FocalNetConfig"],
    "models.fsmt": [
        "FSMTConfig",
        "FSMTTokenizer",
    ],
    "models.funnel": [
        "FunnelConfig",
        "FunnelTokenizer",
    ],
    "models.fuyu": ["FuyuConfig"],
    "models.gemma": ["GemmaConfig"],
    "models.gemma2": ["Gemma2Config"],
    "models.gemma3": ["Gemma3Config", "Gemma3Processor", "Gemma3TextConfig"],
    "models.git": [
        "GitConfig",
        "GitProcessor",
        "GitVisionConfig",
    ],
    "models.glm": ["GlmConfig"],
    "models.glpn": ["GLPNConfig"],
    "models.got_ocr2": [
        "GotOcr2Config",
        "GotOcr2Processor",
        "GotOcr2VisionConfig",
    ],
    "models.gpt2": [
        "GPT2Config",
        "GPT2Tokenizer",
    ],
    "models.gpt_bigcode": ["GPTBigCodeConfig"],
    "models.gpt_neo": ["GPTNeoConfig"],
    "models.gpt_neox": ["GPTNeoXConfig"],
    "models.gpt_neox_japanese": ["GPTNeoXJapaneseConfig"],
    "models.gpt_sw3": [],
    "models.gptj": ["GPTJConfig"],
    "models.granite": ["GraniteConfig"],
    "models.granitemoe": ["GraniteMoeConfig"],
    "models.granitemoeshared": ["GraniteMoeSharedConfig"],
    "models.grounding_dino": [
        "GroundingDinoConfig",
        "GroundingDinoProcessor",
    ],
    "models.groupvit": [
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    "models.helium": ["HeliumConfig"],
    "models.herbert": ["HerbertTokenizer"],
    "models.hiera": ["HieraConfig"],
    "models.hubert": ["HubertConfig"],
    "models.ibert": ["IBertConfig"],
    "models.idefics": ["IdeficsConfig"],
    "models.idefics2": ["Idefics2Config"],
    "models.idefics3": ["Idefics3Config"],
    "models.ijepa": ["IJepaConfig"],
    "models.imagegpt": ["ImageGPTConfig"],
    "models.informer": ["InformerConfig"],
    "models.instructblip": [
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "models.instructblipvideo": [
        "InstructBlipVideoConfig",
        "InstructBlipVideoProcessor",
        "InstructBlipVideoQFormerConfig",
        "InstructBlipVideoVisionConfig",
    ],
    "models.jamba": ["JambaConfig"],
    "models.jetmoe": ["JetMoeConfig"],
    "models.kosmos2": [
        "Kosmos2Config",
        "Kosmos2Processor",
    ],
    "models.layoutlm": [
        "LayoutLMConfig",
        "LayoutLMTokenizer",
    ],
    "models.layoutlmv2": [
        "LayoutLMv2Config",
        "LayoutLMv2FeatureExtractor",
        "LayoutLMv2ImageProcessor",
        "LayoutLMv2Processor",
        "LayoutLMv2Tokenizer",
    ],
    "models.layoutlmv3": [
        "LayoutLMv3Config",
        "LayoutLMv3FeatureExtractor",
        "LayoutLMv3ImageProcessor",
        "LayoutLMv3Processor",
        "LayoutLMv3Tokenizer",
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],
    "models.led": ["LEDConfig", "LEDTokenizer"],
    "models.levit": ["LevitConfig"],
    "models.lilt": ["LiltConfig"],
    "models.llama": ["LlamaConfig"],
    "models.llama4": [
        "Llama4Config",
        "Llama4Processor",
        "Llama4TextConfig",
        "Llama4VisionConfig",
    ],
    "models.llava": [
        "LlavaConfig",
        "LlavaProcessor",
    ],
    "models.llava_next": [
        "LlavaNextConfig",
        "LlavaNextProcessor",
    ],
    "models.llava_next_video": [
        "LlavaNextVideoConfig",
        "LlavaNextVideoProcessor",
    ],
    "models.llava_onevision": ["LlavaOnevisionConfig", "LlavaOnevisionProcessor"],
    "models.longformer": [
        "LongformerConfig",
        "LongformerTokenizer",
    ],
    "models.longt5": ["LongT5Config"],
    "models.luke": [
        "LukeConfig",
        "LukeTokenizer",
    ],
    "models.lxmert": [
        "LxmertConfig",
        "LxmertTokenizer",
    ],
    "models.m2m_100": ["M2M100Config"],
    "models.mamba": ["MambaConfig"],
    "models.mamba2": ["Mamba2Config"],
    "models.marian": ["MarianConfig"],
    "models.markuplm": [
        "MarkupLMConfig",
        "MarkupLMFeatureExtractor",
        "MarkupLMProcessor",
        "MarkupLMTokenizer",
    ],
    "models.mask2former": ["Mask2FormerConfig"],
    "models.maskformer": [
        "MaskFormerConfig",
        "MaskFormerSwinConfig",
    ],
    "models.mbart": ["MBartConfig"],
    "models.mbart50": [],
    "models.megatron_bert": ["MegatronBertConfig"],
    "models.megatron_gpt2": [],
    "models.mgp_str": [
        "MgpstrConfig",
        "MgpstrProcessor",
        "MgpstrTokenizer",
    ],
    "models.mimi": ["MimiConfig"],
    "models.mistral": ["MistralConfig"],
    "models.mistral3": ["Mistral3Config"],
    "models.mixtral": ["MixtralConfig"],
    "models.mllama": [
        "MllamaConfig",
        "MllamaProcessor",
    ],
    "models.mluke": [],
    "models.mobilebert": [
        "MobileBertConfig",
        "MobileBertTokenizer",
    ],
    "models.mobilenet_v1": ["MobileNetV1Config"],
    "models.mobilenet_v2": ["MobileNetV2Config"],
    "models.mobilevit": ["MobileViTConfig"],
    "models.mobilevitv2": ["MobileViTV2Config"],
    "models.modernbert": ["ModernBertConfig"],
    "models.moonshine": ["MoonshineConfig"],
    "models.moshi": [
        "MoshiConfig",
        "MoshiDepthConfig",
    ],
    "models.mpnet": [
        "MPNetConfig",
        "MPNetTokenizer",
    ],
    "models.mpt": ["MptConfig"],
    "models.mra": ["MraConfig"],
    "models.mt5": ["MT5Config"],
    "models.musicgen": [
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "models.musicgen_melody": [
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],
    "models.myt5": ["MyT5Tokenizer"],
    "models.nemotron": ["NemotronConfig"],
    "models.nllb": [],
    "models.nllb_moe": ["NllbMoeConfig"],
    "models.nougat": ["NougatProcessor"],
    "models.nystromformer": ["NystromformerConfig"],
    "models.olmo": ["OlmoConfig"],
    "models.olmo2": ["Olmo2Config"],
    "models.olmoe": ["OlmoeConfig"],
    "models.omdet_turbo": [
        "OmDetTurboConfig",
        "OmDetTurboProcessor",
    ],
    "models.oneformer": [
        "OneFormerConfig",
        "OneFormerProcessor",
    ],
    "models.openai": [
        "OpenAIGPTConfig",
        "OpenAIGPTTokenizer",
    ],
    "models.opt": ["OPTConfig"],
    "models.owlv2": [
        "Owlv2Config",
        "Owlv2Processor",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "models.owlvit": [
        "OwlViTConfig",
        "OwlViTProcessor",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "models.paligemma": ["PaliGemmaConfig"],
    "models.patchtsmixer": ["PatchTSMixerConfig"],
    "models.patchtst": ["PatchTSTConfig"],
    "models.pegasus": [
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    "models.pegasus_x": ["PegasusXConfig"],
    "models.perceiver": [
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    "models.persimmon": ["PersimmonConfig"],
    "models.phi": ["PhiConfig"],
    "models.phi3": ["Phi3Config"],
    "models.phi4_multimodal": [
        "Phi4MultimodalAudioConfig",
        "Phi4MultimodalConfig",
        "Phi4MultimodalFeatureExtractor",
        "Phi4MultimodalProcessor",
        "Phi4MultimodalVisionConfig",
    ],
    "models.phimoe": ["PhimoeConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.pixtral": ["PixtralProcessor", "PixtralVisionConfig"],
    "models.plbart": ["PLBartConfig"],
    "models.poolformer": ["PoolFormerConfig"],
    "models.pop2piano": ["Pop2PianoConfig"],
    "models.prompt_depth_anything": ["PromptDepthAnythingConfig"],
    "models.prophetnet": [
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    "models.pvt": ["PvtConfig"],
    "models.pvt_v2": ["PvtV2Config"],
    "models.qwen2": [
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    "models.qwen2_5_vl": [
        "Qwen2_5_VLConfig",
        "Qwen2_5_VLProcessor",
    ],
    "models.qwen2_audio": [
        "Qwen2AudioConfig",
        "Qwen2AudioEncoderConfig",
        "Qwen2AudioProcessor",
    ],
    "models.qwen2_moe": ["Qwen2MoeConfig"],
    "models.qwen2_vl": [
        "Qwen2VLConfig",
        "Qwen2VLProcessor",
    ],
    "models.qwen3": ["Qwen3Config"],
    "models.qwen3_moe": ["Qwen3MoeConfig"],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.recurrent_gemma": ["RecurrentGemmaConfig"],
    "models.reformer": ["ReformerConfig"],
    "models.regnet": ["RegNetConfig"],
    "models.rembert": ["RemBertConfig"],
    "models.resnet": ["ResNetConfig"],
    "models.roberta": [
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    "models.roberta_prelayernorm": ["RobertaPreLayerNormConfig"],
    "models.roc_bert": [
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    "models.roformer": [
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    "models.rt_detr": ["RTDetrConfig", "RTDetrResNetConfig"],
    "models.rt_detr_v2": ["RTDetrV2Config"],
    "models.rwkv": ["RwkvConfig"],
    "models.sam": [
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "models.seamless_m4t": [
        "SeamlessM4TConfig",
        "SeamlessM4TFeatureExtractor",
        "SeamlessM4TProcessor",
    ],
    "models.seamless_m4t_v2": ["SeamlessM4Tv2Config"],
    "models.segformer": ["SegformerConfig"],
    "models.seggpt": ["SegGptConfig"],
    "models.sew": ["SEWConfig"],
    "models.sew_d": ["SEWDConfig"],
    "models.shieldgemma2": [
        "ShieldGemma2Config",
        "ShieldGemma2Processor",
    ],
    "models.siglip": [
        "SiglipConfig",
        "SiglipProcessor",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "models.siglip2": [
        "Siglip2Config",
        "Siglip2Processor",
        "Siglip2TextConfig",
        "Siglip2VisionConfig",
    ],
    "models.smolvlm": ["SmolVLMConfig"],
    "models.speech_encoder_decoder": ["SpeechEncoderDecoderConfig"],
    "models.speech_to_text": [
        "Speech2TextConfig",
        "Speech2TextFeatureExtractor",
        "Speech2TextProcessor",
    ],
    "models.speecht5": [
        "SpeechT5Config",
        "SpeechT5FeatureExtractor",
        "SpeechT5HifiGanConfig",
        "SpeechT5Processor",
    ],
    "models.splinter": [
        "SplinterConfig",
        "SplinterTokenizer",
    ],
    "models.squeezebert": [
        "SqueezeBertConfig",
        "SqueezeBertTokenizer",
    ],
    "models.stablelm": ["StableLmConfig"],
    "models.starcoder2": ["Starcoder2Config"],
    "models.superglue": ["SuperGlueConfig"],
    "models.superpoint": ["SuperPointConfig"],
    "models.swiftformer": ["SwiftFormerConfig"],
    "models.swin": ["SwinConfig"],
    "models.swin2sr": ["Swin2SRConfig"],
    "models.swinv2": ["Swinv2Config"],
    "models.switch_transformers": ["SwitchTransformersConfig"],
    "models.t5": ["T5Config"],
    "models.table_transformer": ["TableTransformerConfig"],
    "models.tapas": [
        "TapasConfig",
        "TapasTokenizer",
    ],
    "models.textnet": ["TextNetConfig"],
    "models.time_series_transformer": ["TimeSeriesTransformerConfig"],
    "models.timesformer": ["TimesformerConfig"],
    "models.timm_backbone": ["TimmBackboneConfig"],
    "models.timm_wrapper": ["TimmWrapperConfig"],
    "models.trocr": [
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    "models.tvp": [
        "TvpConfig",
        "TvpProcessor",
    ],
    "models.udop": [
        "UdopConfig",
        "UdopProcessor",
    ],
    "models.umt5": ["UMT5Config"],
    "models.unispeech": ["UniSpeechConfig"],
    "models.unispeech_sat": ["UniSpeechSatConfig"],
    "models.univnet": [
        "UnivNetConfig",
        "UnivNetFeatureExtractor",
    ],
    "models.upernet": ["UperNetConfig"],
    "models.video_llava": ["VideoLlavaConfig"],
    "models.videomae": ["VideoMAEConfig"],
    "models.vilt": [
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    "models.vipllava": ["VipLlavaConfig"],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",
        "VisionTextDualEncoderProcessor",
    ],
    "models.visual_bert": ["VisualBertConfig"],
    "models.vit": ["ViTConfig"],
    "models.vit_mae": ["ViTMAEConfig"],
    "models.vit_msn": ["ViTMSNConfig"],
    "models.vitdet": ["VitDetConfig"],
    "models.vitmatte": ["VitMatteConfig"],
    "models.vitpose": ["VitPoseConfig"],
    "models.vitpose_backbone": ["VitPoseBackboneConfig"],
    "models.vits": [
        "VitsConfig",
        "VitsTokenizer",
    ],
    "models.vivit": ["VivitConfig"],
    "models.wav2vec2": [
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_bert": [
        "Wav2Vec2BertConfig",
        "Wav2Vec2BertProcessor",
    ],
    "models.wav2vec2_conformer": ["Wav2Vec2ConformerConfig"],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],
    "models.wavlm": ["WavLMConfig"],
    "models.whisper": [
        "WhisperConfig",
        "WhisperFeatureExtractor",
        "WhisperProcessor",
        "WhisperTokenizer",
    ],
    "models.x_clip": [
        "XCLIPConfig",
        "XCLIPProcessor",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "models.xglm": ["XGLMConfig"],
    "models.xlm": ["XLMConfig", "XLMTokenizer"],
    "models.xlm_roberta": ["XLMRobertaConfig"],
    "models.xlm_roberta_xl": ["XLMRobertaXLConfig"],
    "models.xlnet": ["XLNetConfig"],
    "models.xmod": ["XmodConfig"],
    "models.yolos": ["YolosConfig"],
    "models.yoso": ["YosoConfig"],
    "models.zamba": ["ZambaConfig"],
    "models.zamba2": ["Zamba2Config"],
    "models.zoedepth": ["ZoeDepthConfig"],
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageFeatureExtractionPipeline",
        "ImageSegmentationPipeline",
        "ImageTextToTextPipeline",
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
    "quantizers": [],
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
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": [
        "EvalPrediction",
        "IntervalStrategy",
        "SchedulerType",
        "enable_full_determinism",
        "set_seed",
    ],
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
        "is_av_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_sacremoses_available",
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
        "is_torch_hpu_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AwqConfig",
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "FineGrainedFP8Config",
        "GPTQConfig",
        "HiggsConfig",
        "HqqConfig",
        "QuantoConfig",
        "QuarkConfig",
        "SpQRConfig",
        "TorchAoConfig",
        "VptqConfig",
    ],
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
    _import_structure["models.deprecated.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.deprecated.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gemma"].append("GemmaTokenizer")
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
    _import_structure["models.siglip"].append("SiglipTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.udop"].append("UdopTokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
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
    _import_structure["models.cohere"].append("CohereTokenizerFast")
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    _import_structure["models.deprecated.realm"].append("RealmTokenizerFast")
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gemma"].append("GemmaTokenizerFast")
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
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.udop"].append("UdopTokenizerFast")
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
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

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
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.aria"].extend(["AriaImageProcessor"])
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chameleon"].append("ChameleonImageProcessor")
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
    _import_structure["models.deprecated.deta"].append("DetaImageProcessor")
    _import_structure["models.deprecated.efficientformer"].append("EfficientFormerImageProcessor")
    _import_structure["models.deprecated.tvlt"].append("TvltImageProcessor")
    _import_structure["models.deprecated.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    _import_structure["models.depth_pro"].extend(["DepthProImageProcessor", "DepthProImageProcessorFast"])
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    _import_structure["models.emu3"].append("Emu3ImageProcessor")
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    _import_structure["models.gemma3"].append("Gemma3ImageProcessor")
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    _import_structure["models.got_ocr2"].extend(["GotOcr2ImageProcessor"])
    _import_structure["models.grounding_dino"].extend(["GroundingDinoImageProcessor"])
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    _import_structure["models.idefics2"].extend(["Idefics2ImageProcessor"])
    _import_structure["models.idefics3"].extend(["Idefics3ImageProcessor"])
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    _import_structure["models.instructblipvideo"].extend(["InstructBlipVideoImageProcessor"])
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    _import_structure["models.llava"].append("LlavaImageProcessor")
    _import_structure["models.llava_next"].append("LlavaNextImageProcessor")
    _import_structure["models.llava_next_video"].append("LlavaNextVideoImageProcessor")
    _import_structure["models.llava_onevision"].extend(
        ["LlavaOnevisionImageProcessor", "LlavaOnevisionVideoProcessor"]
    )
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    _import_structure["models.mllama"].extend(["MllamaImageProcessor"])
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    _import_structure["models.nougat"].append("NougatImageProcessor")
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    _import_structure["models.pixtral"].append("PixtralImageProcessor")
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    _import_structure["models.prompt_depth_anything"].extend(["PromptDepthAnythingImageProcessor"])
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    _import_structure["models.qwen2_vl"].extend(["Qwen2VLImageProcessor"])
    _import_structure["models.rt_detr"].extend(["RTDetrImageProcessor"])
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    _import_structure["models.seggpt"].extend(["SegGptImageProcessor"])
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    _import_structure["models.siglip2"].append("Siglip2ImageProcessor")
    _import_structure["models.smolvlm"].extend(["SmolVLMImageProcessor"])
    _import_structure["models.superglue"].extend(["SuperGlueImageProcessor"])
    _import_structure["models.superpoint"].extend(["SuperPointImageProcessor"])
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    _import_structure["models.textnet"].extend(["TextNetImageProcessor"])
    _import_structure["models.tvp"].append("TvpImageProcessor")
    _import_structure["models.video_llava"].append("VideoLlavaImageProcessor")
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    _import_structure["models.vitpose"].append("VitPoseImageProcessor")
    _import_structure["models.vivit"].append("VivitImageProcessor")
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
    _import_structure["models.zoedepth"].append("ZoeDepthImageProcessor")

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torchvision_objects

    _import_structure["utils.dummy_torchvision_objects"] = [
        name for name in dir(dummy_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils_fast"] = ["BaseImageProcessorFast"]
    _import_structure["models.blip"].append("BlipImageProcessorFast")
    _import_structure["models.clip"].append("CLIPImageProcessorFast")
    _import_structure["models.convnext"].append("ConvNextImageProcessorFast")
    _import_structure["models.deformable_detr"].append("DeformableDetrImageProcessorFast")
    _import_structure["models.deit"].append("DeiTImageProcessorFast")
    _import_structure["models.depth_pro"].append("DepthProImageProcessorFast")
    _import_structure["models.detr"].append("DetrImageProcessorFast")
    _import_structure["models.gemma3"].append("Gemma3ImageProcessorFast")
    _import_structure["models.got_ocr2"].append("GotOcr2ImageProcessorFast")
    _import_structure["models.llama4"].append("Llama4ImageProcessorFast")
    _import_structure["models.llava"].append("LlavaImageProcessorFast")
    _import_structure["models.llava_next"].append("LlavaNextImageProcessorFast")
    _import_structure["models.llava_onevision"].append("LlavaOnevisionImageProcessorFast")
    _import_structure["models.phi4_multimodal"].append("Phi4MultimodalImageProcessorFast")
    _import_structure["models.pixtral"].append("PixtralImageProcessorFast")
    _import_structure["models.qwen2_vl"].append("Qwen2VLImageProcessorFast")
    _import_structure["models.rt_detr"].append("RTDetrImageProcessorFast")
    _import_structure["models.siglip"].append("SiglipImageProcessorFast")
    _import_structure["models.siglip2"].append("Siglip2ImageProcessorFast")
    _import_structure["models.vit"].append("ViTImageProcessorFast")

try:
    if not (is_torchvision_available() and is_timm_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_timm_and_torchvision_objects

    _import_structure["utils.dummy_timm_and_torchvision_objects"] = [
        name for name in dir(dummy_timm_and_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.timm_wrapper"].extend(["TimmWrapperImageProcessor"])

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["model_debugging_utils"] = [
        "model_addition_debugger",
        "model_addition_debugger_context",
    ]
    _import_structure["activations"] = []
    _import_structure["cache_utils"] = [
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
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
            "BayesianDetectorConfig",
            "BayesianDetectorModel",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EosTokenCriteria",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "StopStringCriteria",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "SynthIDTextWatermarkDetector",
            "SynthIDTextWatermarkingConfig",
            "SynthIDTextWatermarkLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    # PyTorch domain libraries integration
    _import_structure["integrations.executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS", "dynamic_rope_update"]
    _import_structure["modeling_utils"] = ["PreTrainedModel", "AttentionInterface"]

    # PyTorch models structure

    _import_structure["models.albert"].extend(
        [
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
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    _import_structure["models.altclip"].extend(
        [
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    _import_structure["models.aria"].extend(
        [
            "AriaForConditionalGeneration",
            "AriaPreTrainedModel",
            "AriaTextForCausalLM",
            "AriaTextModel",
            "AriaTextPreTrainedModel",
        ]
    )
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
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
            "MODEL_FOR_IMAGE_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_RETRIEVAL_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
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
            "AutoModelForImageTextToText",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForKeypointDetection",
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
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    _import_structure["models.aya_vision"].extend(["AyaVisionForConditionalGeneration", "AyaVisionPreTrainedModel"])
    _import_structure["models.bamba"].extend(
        [
            "BambaForCausalLM",
            "BambaModel",
            "BambaPreTrainedModel",
        ]
    )
    _import_structure["models.bark"].extend(
        [
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
            "BeitBackbone",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
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
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
            "load_tf_weights_in_big_bird",
        ]
    )
    _import_structure["models.bigbird_pegasus"].extend(
        [
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
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    _import_structure["models.bit"].extend(
        [
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
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
            "Blip2ForConditionalGeneration",
            "Blip2ForImageTextRetrieval",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2TextModelWithProjection",
            "Blip2VisionModel",
            "Blip2VisionModelWithProjection",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
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
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    _import_structure["models.bros"].extend(
        [
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
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineModel",
            "CaninePreTrainedModel",
            "load_tf_weights_in_canine",
        ]
    )
    _import_structure["models.chameleon"].extend(
        [
            "ChameleonForConditionalGeneration",
            "ChameleonModel",
            "ChameleonPreTrainedModel",
            "ChameleonProcessor",
            "ChameleonVQVAE",
        ]
    )
    _import_structure["models.chinese_clip"].extend(
        [
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    _import_structure["models.clap"].extend(
        [
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
            "CLIPForImageClassification",
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
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    _import_structure["models.clvp"].extend(
        [
            "ClvpDecoder",
            "ClvpEncoder",
            "ClvpForCausalLM",
            "ClvpModel",
            "ClvpModelForConditionalGeneration",
            "ClvpPreTrainedModel",
        ]
    )
    _import_structure["models.codegen"].extend(
        [
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    _import_structure["models.cohere"].extend(["CohereForCausalLM", "CohereModel", "CoherePreTrainedModel"])
    _import_structure["models.cohere2"].extend(["Cohere2ForCausalLM", "Cohere2Model", "Cohere2PreTrainedModel"])
    _import_structure["models.colpali"].extend(
        [
            "ColPaliForRetrieval",
            "ColPaliPreTrainedModel",
        ]
    )
    _import_structure["models.conditional_detr"].extend(
        [
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.cpmant"].extend(
        [
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    _import_structure["models.dab_detr"].extend(
        [
            "DabDetrForObjectDetection",
            "DabDetrModel",
            "DabDetrPreTrainedModel",
        ]
    )
    _import_structure["models.dac"].extend(
        [
            "DacModel",
            "DacPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
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
    _import_structure["models.dbrx"].extend(
        [
            "DbrxForCausalLM",
            "DbrxModel",
            "DbrxPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
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
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deepseek_v3"].extend(
        [
            "DeepseekV3ForCausalLM",
            "DeepseekV3Model",
            "DeepseekV3PreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.deta"].extend(
        [
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.ernie_m"].extend(
        [
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.gptsan_japanese"].extend(
        [
            "GPTSanJapaneseForConditionalGeneration",
            "GPTSanJapaneseModel",
            "GPTSanJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.graphormer"].extend(
        [
            "GraphormerForGraphClassification",
            "GraphormerModel",
            "GraphormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.jukebox"].extend(
        [
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mega"].extend(
        [
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
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.deprecated.nat"].extend(
        [
            "NatBackbone",
            "NatForImageClassification",
            "NatModel",
            "NatPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.nezha"].extend(
        [
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
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.qdqbert"].extend(
        [
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
            "load_tf_weights_in_qdqbert",
        ]
    )
    _import_structure["models.deprecated.realm"].extend(
        [
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
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.speech_to_text_2"].extend(
        ["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.deprecated.tvlt"].extend(
        [
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.vit_hybrid"].extend(
        [
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.xlm_prophetnet"].extend(
        [
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.depth_anything"].extend(
        [
            "DepthAnythingForDepthEstimation",
            "DepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.depth_pro"].extend(
        [
            "DepthProForDepthEstimation",
            "DepthProModel",
            "DepthProPreTrainedModel",
        ]
    )
    _import_structure["models.detr"].extend(
        [
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    _import_structure["models.diffllama"].extend(
        [
            "DiffLlamaForCausalLM",
            "DiffLlamaForQuestionAnswering",
            "DiffLlamaForSequenceClassification",
            "DiffLlamaForTokenClassification",
            "DiffLlamaModel",
            "DiffLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.dinat"].extend(
        [
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    _import_structure["models.dinov2_with_registers"].extend(
        [
            "Dinov2WithRegistersBackbone",
            "Dinov2WithRegistersForImageClassification",
            "Dinov2WithRegistersModel",
            "Dinov2WithRegistersPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
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
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
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
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    _import_structure["models.efficientnet"].extend(
        [
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
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
    _import_structure["models.emu3"].extend(
        [
            "Emu3ForCausalLM",
            "Emu3ForConditionalGeneration",
            "Emu3PreTrainedModel",
            "Emu3TextModel",
            "Emu3VQVAE",
        ]
    )
    _import_structure["models.encodec"].extend(
        [
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.ernie"].extend(
        [
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
    _import_structure["models.esm"].extend(
        [
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
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    _import_structure["models.falcon_mamba"].extend(
        [
            "FalconMambaForCausalLM",
            "FalconMambaModel",
            "FalconMambaPreTrainedModel",
        ]
    )
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
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
            "FNetForMaskedLM",
            "FNetForMultipleChoice",
            "FNetForNextSentencePrediction",
            "FNetForPreTraining",
            "FNetForQuestionAnswering",
            "FNetForSequenceClassification",
            "FNetForTokenClassification",
            "FNetModel",
            "FNetPreTrainedModel",
        ]
    )
    _import_structure["models.focalnet"].extend(
        [
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
    _import_structure["models.gemma"].extend(
        [
            "GemmaForCausalLM",
            "GemmaForSequenceClassification",
            "GemmaForTokenClassification",
            "GemmaModel",
            "GemmaPreTrainedModel",
        ]
    )
    _import_structure["models.gemma2"].extend(
        [
            "Gemma2ForCausalLM",
            "Gemma2ForSequenceClassification",
            "Gemma2ForTokenClassification",
            "Gemma2Model",
            "Gemma2PreTrainedModel",
        ]
    )
    _import_structure["models.gemma3"].extend(
        [
            "Gemma3ForCausalLM",
            "Gemma3ForConditionalGeneration",
            "Gemma3PreTrainedModel",
            "Gemma3TextModel",
        ]
    )
    _import_structure["models.git"].extend(
        [
            "GitForCausalLM",
            "GitModel",
            "GitPreTrainedModel",
            "GitVisionModel",
        ]
    )
    _import_structure["models.glm"].extend(
        [
            "GlmForCausalLM",
            "GlmForSequenceClassification",
            "GlmForTokenClassification",
            "GlmModel",
            "GlmPreTrainedModel",
        ]
    )
    _import_structure["models.llama4"].extend(
        [
            "Llama4ForCausalLM",
            "Llama4ForConditionalGeneration",
            "Llama4TextModel",
            "Llama4VisionModel",
            "Llama4PreTrainedModel",
        ]
    )
    _import_structure["models.glpn"].extend(
        [
            "GLPNForDepthEstimation",
            "GLPNModel",
            "GLPNPreTrainedModel",
        ]
    )
    _import_structure["models.got_ocr2"].extend(
        [
            "GotOcr2ForConditionalGeneration",
            "GotOcr2PreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
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
            "GPTBigCodeForCausalLM",
            "GPTBigCodeForSequenceClassification",
            "GPTBigCodeForTokenClassification",
            "GPTBigCodeModel",
            "GPTBigCodePreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neo"].extend(
        [
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
            "GPTNeoXForCausalLM",
            "GPTNeoXForQuestionAnswering",
            "GPTNeoXForSequenceClassification",
            "GPTNeoXForTokenClassification",
            "GPTNeoXModel",
            "GPTNeoXPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPTNeoXJapaneseForCausalLM",
            "GPTNeoXJapaneseModel",
            "GPTNeoXJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "GPTJForCausalLM",
            "GPTJForQuestionAnswering",
            "GPTJForSequenceClassification",
            "GPTJModel",
            "GPTJPreTrainedModel",
        ]
    )
    _import_structure["models.granite"].extend(
        [
            "GraniteForCausalLM",
            "GraniteModel",
            "GranitePreTrainedModel",
        ]
    )
    _import_structure["models.granitemoe"].extend(
        [
            "GraniteMoeForCausalLM",
            "GraniteMoeModel",
            "GraniteMoePreTrainedModel",
        ]
    )

    _import_structure["models.granitemoeshared"].extend(
        [
            "GraniteMoeSharedForCausalLM",
            "GraniteMoeSharedModel",
            "GraniteMoeSharedPreTrainedModel",
        ]
    )
    _import_structure["models.grounding_dino"].extend(
        [
            "GroundingDinoForObjectDetection",
            "GroundingDinoModel",
            "GroundingDinoPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "GroupViTModel",
            "GroupViTPreTrainedModel",
            "GroupViTTextModel",
            "GroupViTVisionModel",
        ]
    )
    _import_structure["models.helium"].extend(
        [
            "HeliumForCausalLM",
            "HeliumForSequenceClassification",
            "HeliumForTokenClassification",
            "HeliumModel",
            "HeliumPreTrainedModel",
        ]
    )
    _import_structure["models.hiera"].extend(
        [
            "HieraBackbone",
            "HieraForImageClassification",
            "HieraForPreTraining",
            "HieraModel",
            "HieraPreTrainedModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
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
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    _import_structure["models.idefics2"].extend(
        [
            "Idefics2ForConditionalGeneration",
            "Idefics2Model",
            "Idefics2PreTrainedModel",
            "Idefics2Processor",
        ]
    )
    _import_structure["models.idefics3"].extend(
        [
            "Idefics3ForConditionalGeneration",
            "Idefics3Model",
            "Idefics3PreTrainedModel",
            "Idefics3Processor",
            "Idefics3VisionConfig",
            "Idefics3VisionTransformer",
        ]
    )
    _import_structure["models.ijepa"].extend(
        [
            "IJepaForImageClassification",
            "IJepaModel",
            "IJepaPreTrainedModel",
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
            "load_tf_weights_in_imagegpt",
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    _import_structure["models.instructblipvideo"].extend(
        [
            "InstructBlipVideoForConditionalGeneration",
            "InstructBlipVideoPreTrainedModel",
            "InstructBlipVideoQFormerModel",
            "InstructBlipVideoVisionModel",
        ]
    )
    _import_structure["models.jamba"].extend(
        [
            "JambaForCausalLM",
            "JambaForSequenceClassification",
            "JambaModel",
            "JambaPreTrainedModel",
        ]
    )
    _import_structure["models.jetmoe"].extend(
        [
            "JetMoeForCausalLM",
            "JetMoeForSequenceClassification",
            "JetMoeModel",
            "JetMoePreTrainedModel",
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
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
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    _import_structure["models.levit"].extend(
        [
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    _import_structure["models.lilt"].extend(
        [
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",
            "LlamaForQuestionAnswering",
            "LlamaForSequenceClassification",
            "LlamaForTokenClassification",
            "LlamaModel",
            "LlamaPreTrainedModel",
        ]
    )
    _import_structure["models.llava"].extend(
        [
            "LlavaForConditionalGeneration",
            "LlavaPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next"].extend(
        [
            "LlavaNextForConditionalGeneration",
            "LlavaNextPreTrainedModel",
        ]
    )
    _import_structure["models.phi4_multimodal"].extend(
        [
            "Phi4MultimodalForCausalLM",
            "Phi4MultimodalPreTrainedModel",
            "Phi4MultimodalAudioModel",
            "Phi4MultimodalAudioPreTrainedModel",
            "Phi4MultimodalModel",
            "Phi4MultimodalVisionModel",
            "Phi4MultimodalVisionPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next_video"].extend(
        [
            "LlavaNextVideoForConditionalGeneration",
            "LlavaNextVideoPreTrainedModel",
        ]
    )
    _import_structure["models.llava_onevision"].extend(
        [
            "LlavaOnevisionForConditionalGeneration",
            "LlavaOnevisionPreTrainedModel",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
        ]
    )
    _import_structure["models.longt5"].extend(
        [
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    _import_structure["models.luke"].extend(
        [
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
        ]
    )
    _import_structure["models.m2m_100"].extend(
        [
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    _import_structure["models.mamba"].extend(
        [
            "MambaForCausalLM",
            "MambaModel",
            "MambaPreTrainedModel",
        ]
    )
    _import_structure["models.mamba2"].extend(
        [
            "Mamba2ForCausalLM",
            "Mamba2Model",
            "Mamba2PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        ["MarianForCausalLM", "MarianModel", "MarianMTModel", "MarianPreTrainedModel"]
    )
    _import_structure["models.markuplm"].extend(
        [
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    _import_structure["models.mask2former"].extend(
        [
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    _import_structure["models.maskformer"].extend(
        [
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
    _import_structure["models.megatron_bert"].extend(
        [
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
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    _import_structure["models.mimi"].extend(
        [
            "MimiModel",
            "MimiPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",
            "MistralForQuestionAnswering",
            "MistralForSequenceClassification",
            "MistralForTokenClassification",
            "MistralModel",
            "MistralPreTrainedModel",
        ]
    )
    _import_structure["models.mistral3"].extend(
        [
            "Mistral3ForConditionalGeneration",
            "Mistral3PreTrainedModel",
        ]
    )
    _import_structure["models.mixtral"].extend(
        [
            "MixtralForCausalLM",
            "MixtralForQuestionAnswering",
            "MixtralForSequenceClassification",
            "MixtralForTokenClassification",
            "MixtralModel",
            "MixtralPreTrainedModel",
        ]
    )
    _import_structure["models.mllama"].extend(
        [
            "MllamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MllamaPreTrainedModel",
            "MllamaProcessor",
            "MllamaTextModel",
            "MllamaVisionModel",
        ]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
            "load_tf_weights_in_mobilenet_v1",
        ]
    )
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
            "load_tf_weights_in_mobilenet_v2",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevitv2"].extend(
        [
            "MobileViTV2ForImageClassification",
            "MobileViTV2ForSemanticSegmentation",
            "MobileViTV2Model",
            "MobileViTV2PreTrainedModel",
        ]
    )
    _import_structure["models.modernbert"].extend(
        [
            "ModernBertForMaskedLM",
            "ModernBertForQuestionAnswering",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
            "ModernBertModel",
            "ModernBertPreTrainedModel",
        ]
    )
    _import_structure["models.moonshine"].extend(
        [
            "MoonshineForConditionalGeneration",
            "MoonshineModel",
            "MoonshinePreTrainedModel",
        ]
    )
    _import_structure["models.moshi"].extend(
        [
            "MoshiForCausalLM",
            "MoshiForConditionalGeneration",
            "MoshiModel",
            "MoshiPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mpt"].extend(
        [
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
            "MT5ForTokenClassification",
            "MT5Model",
            "MT5PreTrainedModel",
        ]
    )
    _import_structure["models.musicgen"].extend(
        [
            "MusicgenForCausalLM",
            "MusicgenForConditionalGeneration",
            "MusicgenModel",
            "MusicgenPreTrainedModel",
            "MusicgenProcessor",
        ]
    )
    _import_structure["models.musicgen_melody"].extend(
        [
            "MusicgenMelodyForCausalLM",
            "MusicgenMelodyForConditionalGeneration",
            "MusicgenMelodyModel",
            "MusicgenMelodyPreTrainedModel",
        ]
    )
    _import_structure["models.mvp"].extend(
        [
            "MvpForCausalLM",
            "MvpForConditionalGeneration",
            "MvpForQuestionAnswering",
            "MvpForSequenceClassification",
            "MvpModel",
            "MvpPreTrainedModel",
        ]
    )
    _import_structure["models.nemotron"].extend(
        [
            "NemotronForCausalLM",
            "NemotronForQuestionAnswering",
            "NemotronForSequenceClassification",
            "NemotronForTokenClassification",
            "NemotronModel",
            "NemotronPreTrainedModel",
        ]
    )
    _import_structure["models.nllb_moe"].extend(
        [
            "NllbMoeForConditionalGeneration",
            "NllbMoeModel",
            "NllbMoePreTrainedModel",
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NystromformerForMaskedLM",
            "NystromformerForMultipleChoice",
            "NystromformerForQuestionAnswering",
            "NystromformerForSequenceClassification",
            "NystromformerForTokenClassification",
            "NystromformerModel",
            "NystromformerPreTrainedModel",
        ]
    )
    _import_structure["models.olmo"].extend(
        [
            "OlmoForCausalLM",
            "OlmoModel",
            "OlmoPreTrainedModel",
        ]
    )
    _import_structure["models.olmo2"].extend(
        [
            "Olmo2ForCausalLM",
            "Olmo2Model",
            "Olmo2PreTrainedModel",
        ]
    )
    _import_structure["models.olmoe"].extend(
        [
            "OlmoeForCausalLM",
            "OlmoeModel",
            "OlmoePreTrainedModel",
        ]
    )
    _import_structure["models.omdet_turbo"].extend(
        [
            "OmDetTurboForObjectDetection",
            "OmDetTurboPreTrainedModel",
        ]
    )
    _import_structure["models.oneformer"].extend(
        [
            "OneFormerForUniversalSegmentation",
            "OneFormerModel",
            "OneFormerPreTrainedModel",
        ]
    )
    _import_structure["models.openai"].extend(
        [
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
            "OPTForCausalLM",
            "OPTForQuestionAnswering",
            "OPTForSequenceClassification",
            "OPTModel",
            "OPTPreTrainedModel",
        ]
    )
    _import_structure["models.owlv2"].extend(
        [
            "Owlv2ForObjectDetection",
            "Owlv2Model",
            "Owlv2PreTrainedModel",
            "Owlv2TextModel",
            "Owlv2VisionModel",
        ]
    )
    _import_structure["models.owlvit"].extend(
        [
            "OwlViTForObjectDetection",
            "OwlViTModel",
            "OwlViTPreTrainedModel",
            "OwlViTTextModel",
            "OwlViTVisionModel",
        ]
    )
    _import_structure["models.paligemma"].extend(
        [
            "PaliGemmaForConditionalGeneration",
            "PaliGemmaPreTrainedModel",
            "PaliGemmaProcessor",
        ]
    )
    _import_structure["models.patchtsmixer"].extend(
        [
            "PatchTSMixerForPrediction",
            "PatchTSMixerForPretraining",
            "PatchTSMixerForRegression",
            "PatchTSMixerForTimeSeriesClassification",
            "PatchTSMixerModel",
            "PatchTSMixerPreTrainedModel",
        ]
    )
    _import_structure["models.patchtst"].extend(
        [
            "PatchTSTForClassification",
            "PatchTSTForPrediction",
            "PatchTSTForPretraining",
            "PatchTSTForRegression",
            "PatchTSTModel",
            "PatchTSTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",
            "PegasusForConditionalGeneration",
            "PegasusModel",
            "PegasusPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus_x"].extend(
        [
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    _import_structure["models.perceiver"].extend(
        [
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonForTokenClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    _import_structure["models.phi"].extend(
        [
            "PhiForCausalLM",
            "PhiForSequenceClassification",
            "PhiForTokenClassification",
            "PhiModel",
            "PhiPreTrainedModel",
        ]
    )
    _import_structure["models.phi3"].extend(
        [
            "Phi3ForCausalLM",
            "Phi3ForSequenceClassification",
            "Phi3ForTokenClassification",
            "Phi3Model",
            "Phi3PreTrainedModel",
        ]
    )
    _import_structure["models.phimoe"].extend(
        [
            "PhimoeForCausalLM",
            "PhimoeForSequenceClassification",
            "PhimoeModel",
            "PhimoePreTrainedModel",
        ]
    )
    _import_structure["models.pix2struct"].extend(
        [
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    _import_structure["models.pixtral"].extend(["PixtralPreTrainedModel", "PixtralVisionModel"])
    _import_structure["models.plbart"].extend(
        [
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    _import_structure["models.poolformer"].extend(
        [
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    _import_structure["models.pop2piano"].extend(
        [
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    _import_structure["models.prompt_depth_anything"].extend(
        [
            "PromptDepthAnythingForDepthEstimation",
            "PromptDepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.prophetnet"].extend(
        [
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
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    _import_structure["models.pvt_v2"].extend(
        [
            "PvtV2Backbone",
            "PvtV2ForImageClassification",
            "PvtV2Model",
            "PvtV2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",
            "Qwen2ForQuestionAnswering",
            "Qwen2ForSequenceClassification",
            "Qwen2ForTokenClassification",
            "Qwen2Model",
            "Qwen2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_5_vl"].extend(
        [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5_VLModel",
            "Qwen2_5_VLPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_audio"].extend(
        [
            "Qwen2AudioEncoder",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2AudioPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_moe"].extend(
        [
            "Qwen2MoeForCausalLM",
            "Qwen2MoeForQuestionAnswering",
            "Qwen2MoeForSequenceClassification",
            "Qwen2MoeForTokenClassification",
            "Qwen2MoeModel",
            "Qwen2MoePreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_vl"].extend(
        [
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLModel",
            "Qwen2VLPreTrainedModel",
        ]
    )
    _import_structure["models.qwen3"].extend(
        [
            "Qwen3ForCausalLM",
            "Qwen3ForQuestionAnswering",
            "Qwen3ForSequenceClassification",
            "Qwen3ForTokenClassification",
            "Qwen3Model",
            "Qwen3PreTrainedModel",
        ]
    )
    _import_structure["models.qwen3_moe"].extend(
        [
            "Qwen3MoeForCausalLM",
            "Qwen3MoeForQuestionAnswering",
            "Qwen3MoeForSequenceClassification",
            "Qwen3MoeForTokenClassification",
            "Qwen3MoeModel",
            "Qwen3MoePreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "RagModel",
            "RagPreTrainedModel",
            "RagSequenceForGeneration",
            "RagTokenForGeneration",
        ]
    )
    _import_structure["models.recurrent_gemma"].extend(
        [
            "RecurrentGemmaForCausalLM",
            "RecurrentGemmaModel",
            "RecurrentGemmaPreTrainedModel",
        ]
    )
    _import_structure["models.reformer"].extend(
        [
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerModel",
            "ReformerModelWithLMHead",
            "ReformerPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "RegNetForImageClassification",
            "RegNetModel",
            "RegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "RemBertForCausalLM",
            "RemBertForMaskedLM",
            "RemBertForMultipleChoice",
            "RemBertForQuestionAnswering",
            "RemBertForSequenceClassification",
            "RemBertForTokenClassification",
            "RemBertModel",
            "RemBertPreTrainedModel",
            "load_tf_weights_in_rembert",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "ResNetBackbone",
            "ResNetForImageClassification",
            "ResNetModel",
            "ResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
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
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
            "load_tf_weights_in_roc_bert",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
            "load_tf_weights_in_roformer",
        ]
    )
    _import_structure["models.rt_detr"].extend(
        [
            "RTDetrForObjectDetection",
            "RTDetrModel",
            "RTDetrPreTrainedModel",
            "RTDetrResNetBackbone",
            "RTDetrResNetPreTrainedModel",
        ]
    )
    _import_structure["models.rt_detr_v2"].extend(
        ["RTDetrV2ForObjectDetection", "RTDetrV2Model", "RTDetrV2PreTrainedModel"]
    )
    _import_structure["models.rwkv"].extend(
        [
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "SamModel",
            "SamPreTrainedModel",
            "SamVisionModel",
        ]
    )
    _import_structure["models.seamless_m4t"].extend(
        [
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
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SeamlessM4Tv2ForSpeechToSpeech",
            "SeamlessM4Tv2ForSpeechToText",
            "SeamlessM4Tv2ForTextToSpeech",
            "SeamlessM4Tv2ForTextToText",
            "SeamlessM4Tv2Model",
            "SeamlessM4Tv2PreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "SegformerDecodeHead",
            "SegformerForImageClassification",
            "SegformerForSemanticSegmentation",
            "SegformerModel",
            "SegformerPreTrainedModel",
        ]
    )
    _import_structure["models.seggpt"].extend(
        [
            "SegGptForImageSegmentation",
            "SegGptModel",
            "SegGptPreTrainedModel",
        ]
    )
    _import_structure["models.sew"].extend(
        [
            "SEWForCTC",
            "SEWForSequenceClassification",
            "SEWModel",
            "SEWPreTrainedModel",
        ]
    )
    _import_structure["models.sew_d"].extend(
        [
            "SEWDForCTC",
            "SEWDForSequenceClassification",
            "SEWDModel",
            "SEWDPreTrainedModel",
        ]
    )
    _import_structure["models.shieldgemma2"].append("ShieldGemma2ForImageClassification")
    _import_structure["models.siglip"].extend(
        [
            "SiglipForImageClassification",
            "SiglipModel",
            "SiglipPreTrainedModel",
            "SiglipTextModel",
            "SiglipVisionModel",
        ]
    )
    _import_structure["models.siglip2"].extend(
        [
            "Siglip2ForImageClassification",
            "Siglip2Model",
            "Siglip2PreTrainedModel",
            "Siglip2TextModel",
            "Siglip2VisionModel",
        ]
    )
    _import_structure["models.smolvlm"].extend(
        [
            "SmolVLMForConditionalGeneration",
            "SmolVLMModel",
            "SmolVLMPreTrainedModel",
            "SmolVLMProcessor",
            "SmolVLMVisionConfig",
            "SmolVLMVisionTransformer",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    _import_structure["models.speech_to_text"].extend(
        [
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.speecht5"].extend(
        [
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
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.stablelm"].extend(
        [
            "StableLmForCausalLM",
            "StableLmForSequenceClassification",
            "StableLmForTokenClassification",
            "StableLmModel",
            "StableLmPreTrainedModel",
        ]
    )
    _import_structure["models.starcoder2"].extend(
        [
            "Starcoder2ForCausalLM",
            "Starcoder2ForSequenceClassification",
            "Starcoder2ForTokenClassification",
            "Starcoder2Model",
            "Starcoder2PreTrainedModel",
        ]
    )
    _import_structure["models.superglue"].extend(
        [
            "SuperGlueForKeypointMatching",
            "SuperGluePreTrainedModel",
        ]
    )
    _import_structure["models.superpoint"].extend(
        [
            "SuperPointForKeypointDetection",
            "SuperPointPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "SwinBackbone",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinModel",
            "SwinPreTrainedModel",
        ]
    )
    _import_structure["models.swin2sr"].extend(
        [
            "Swin2SRForImageSuperResolution",
            "Swin2SRModel",
            "Swin2SRPreTrainedModel",
        ]
    )
    _import_structure["models.swinv2"].extend(
        [
            "Swinv2Backbone",
            "Swinv2ForImageClassification",
            "Swinv2ForMaskedImageModeling",
            "Swinv2Model",
            "Swinv2PreTrainedModel",
        ]
    )
    _import_structure["models.switch_transformers"].extend(
        [
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
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5ForQuestionAnswering",
            "T5ForSequenceClassification",
            "T5ForTokenClassification",
            "T5Model",
            "T5PreTrainedModel",
            "load_tf_weights_in_t5",
        ]
    )
    _import_structure["models.table_transformer"].extend(
        [
            "TableTransformerForObjectDetection",
            "TableTransformerModel",
            "TableTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
            "TapasPreTrainedModel",
            "load_tf_weights_in_tapas",
        ]
    )
    _import_structure["models.textnet"].extend(
        [
            "TextNetBackbone",
            "TextNetForImageClassification",
            "TextNetModel",
            "TextNetPreTrainedModel",
        ]
    )
    _import_structure["models.time_series_transformer"].extend(
        [
            "TimeSeriesTransformerForPrediction",
            "TimeSeriesTransformerModel",
            "TimeSeriesTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.timesformer"].extend(
        [
            "TimesformerForVideoClassification",
            "TimesformerModel",
            "TimesformerPreTrainedModel",
        ]
    )
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])
    _import_structure["models.timm_wrapper"].extend(
        ["TimmWrapperForImageClassification", "TimmWrapperModel", "TimmWrapperPreTrainedModel"]
    )
    _import_structure["models.trocr"].extend(
        [
            "TrOCRForCausalLM",
            "TrOCRPreTrainedModel",
        ]
    )
    _import_structure["models.tvp"].extend(
        [
            "TvpForVideoGrounding",
            "TvpModel",
            "TvpPreTrainedModel",
        ]
    )
    _import_structure["models.udop"].extend(
        [
            "UdopEncoderModel",
            "UdopForConditionalGeneration",
            "UdopModel",
            "UdopPreTrainedModel",
        ],
    )
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5ForTokenClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    _import_structure["models.unispeech"].extend(
        [
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    _import_structure["models.unispeech_sat"].extend(
        [
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    _import_structure["models.univnet"].extend(
        [
            "UnivNetModel",
        ]
    )
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    _import_structure["models.video_llava"].extend(
        [
            "VideoLlavaForConditionalGeneration",
            "VideoLlavaPreTrainedModel",
            "VideoLlavaProcessor",
        ]
    )
    _import_structure["models.videomae"].extend(
        [
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vilt"].extend(
        [
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    _import_structure["models.vipllava"].extend(
        [
            "VipLlavaForConditionalGeneration",
            "VipLlavaPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    _import_structure["models.visual_bert"].extend(
        [
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    _import_structure["models.vit"].extend(
        [
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "ViTMAEForPreTraining",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vit_msn"].extend(
        [
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    _import_structure["models.vitdet"].extend(
        [
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    _import_structure["models.vitmatte"].extend(
        [
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    _import_structure["models.vitpose"].extend(
        [
            "VitPoseForPoseEstimation",
            "VitPosePreTrainedModel",
        ]
    )
    _import_structure["models.vitpose_backbone"].extend(
        [
            "VitPoseBackbone",
            "VitPoseBackbonePreTrainedModel",
        ]
    )
    _import_structure["models.vits"].extend(
        [
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    _import_structure["models.vivit"].extend(
        [
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
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
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "Wav2Vec2BertForAudioFrameClassification",
            "Wav2Vec2BertForCTC",
            "Wav2Vec2BertForSequenceClassification",
            "Wav2Vec2BertForXVector",
            "Wav2Vec2BertModel",
            "Wav2Vec2BertPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
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
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
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
    _import_structure["models.xlm_roberta"].extend(
        [
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
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    _import_structure["models.zamba"].extend(
        [
            "ZambaForCausalLM",
            "ZambaForSequenceClassification",
            "ZambaModel",
            "ZambaPreTrainedModel",
        ]
    )
    _import_structure["models.zamba2"].extend(
        [
            "Zamba2ForCausalLM",
            "Zamba2ForSequenceClassification",
            "Zamba2Model",
            "Zamba2PreTrainedModel",
        ]
    )
    _import_structure["models.zoedepth"].extend(
        [
            "ZoeDepthForDepthEstimation",
            "ZoeDepthPreTrainedModel",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
        "get_wsd_schedule",
    ]
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
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
        ]
    )
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
        [
            "TFBartForConditionalGeneration",
            "TFBartForSequenceClassification",
            "TFBartModel",
            "TFBartPretrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
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
        [
            "TFBlenderbotForConditionalGeneration",
            "TFBlenderbotModel",
            "TFBlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "TFBlenderbotSmallForConditionalGeneration",
            "TFBlenderbotSmallModel",
            "TFBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
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
            "TFCLIPModel",
            "TFCLIPPreTrainedModel",
            "TFCLIPTextModel",
            "TFCLIPVisionModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
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
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
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
            "TFDeiTForImageClassification",
            "TFDeiTForImageClassificationWithTeacher",
            "TFDeiTForMaskedImageModeling",
            "TFDeiTModel",
            "TFDeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "TFEfficientFormerForImageClassification",
            "TFEfficientFormerForImageClassificationWithTeacher",
            "TFEfficientFormerModel",
            "TFEfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
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
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    _import_structure["models.electra"].extend(
        [
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
            "TFEsmForMaskedLM",
            "TFEsmForSequenceClassification",
            "TFEsmForTokenClassification",
            "TFEsmModel",
            "TFEsmPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
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
            "TFGroupViTModel",
            "TFGroupViTPreTrainedModel",
            "TFGroupViTTextModel",
            "TFGroupViTVisionModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "TFHubertForCTC",
            "TFHubertModel",
            "TFHubertPreTrainedModel",
        ]
    )

    _import_structure["models.idefics"].extend(
        [
            "TFIdeficsForVisionText2Text",
            "TFIdeficsModel",
            "TFIdeficsPreTrainedModel",
        ]
    )

    _import_structure["models.layoutlm"].extend(
        [
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
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerPreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
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
    _import_structure["models.mistral"].extend(
        ["TFMistralForCausalLM", "TFMistralForSequenceClassification", "TFMistralModel", "TFMistralPreTrainedModel"]
    )
    _import_structure["models.mobilebert"].extend(
        [
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
            "TFMobileViTForImageClassification",
            "TFMobileViTForSemanticSegmentation",
            "TFMobileViTModel",
            "TFMobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
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
        [
            "TFPegasusForConditionalGeneration",
            "TFPegasusModel",
            "TFPegasusPreTrainedModel",
        ]
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
            "TFRegNetForImageClassification",
            "TFRegNetModel",
            "TFRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "TFRemBertForCausalLM",
            "TFRemBertForMaskedLM",
            "TFRemBertForMultipleChoice",
            "TFRemBertForQuestionAnswering",
            "TFRemBertForSequenceClassification",
            "TFRemBertForTokenClassification",
            "TFRemBertModel",
            "TFRemBertPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "TFResNetForImageClassification",
            "TFResNetModel",
            "TFResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
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
            "TFRoFormerForCausalLM",
            "TFRoFormerForMaskedLM",
            "TFRoFormerForMultipleChoice",
            "TFRoFormerForQuestionAnswering",
            "TFRoFormerForSequenceClassification",
            "TFRoFormerForTokenClassification",
            "TFRoFormerModel",
            "TFRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "TFSamModel",
            "TFSamPreTrainedModel",
            "TFSamVisionModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "TFSegformerDecodeHead",
            "TFSegformerForImageClassification",
            "TFSegformerForSemanticSegmentation",
            "TFSegformerModel",
            "TFSegformerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_to_text"].extend(
        [
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "TFSwiftFormerForImageClassification",
            "TFSwiftFormerModel",
            "TFSwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TFTapasForMaskedLM",
            "TFTapasForQuestionAnswering",
            "TFTapasForSequenceClassification",
            "TFTapasModel",
            "TFTapasPreTrainedModel",
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
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
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
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",
        "GradientAccumulator",
        "WarmUp",
        "create_optimizer",
    ]
    _import_structure["tf_utils"] = []


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
    from .utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )

    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import (
        dummy_torchaudio_objects,
    )

    _import_structure["utils.dummy_torchaudio_objects"] = [
        name for name in dir(dummy_torchaudio_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyFeatureExtractor")
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyProcessor")


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
        [
            "FlaxBlenderbotForConditionalGeneration",
            "FlaxBlenderbotModel",
            "FlaxBlenderbotPreTrainedModel",
        ]
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
    _import_structure["models.dinov2"].extend(
        [
            "FlaxDinov2Model",
            "FlaxDinov2ForImageClassification",
            "FlaxDinov2PreTrainedModel",
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
    _import_structure["models.llama"].extend(["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"])
    _import_structure["models.gemma"].extend(["FlaxGemmaForCausalLM", "FlaxGemmaModel", "FlaxGemmaPreTrainedModel"])
    _import_structure["models.longt5"].extend(
        [
            "FlaxLongT5ForConditionalGeneration",
            "FlaxLongT5Model",
            "FlaxLongT5PreTrainedModel",
        ]
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
    _import_structure["models.mistral"].extend(
        [
            "FlaxMistralForCausalLM",
            "FlaxMistralModel",
            "FlaxMistralPreTrainedModel",
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
        [
            "FlaxRegNetForImageClassification",
            "FlaxRegNetModel",
            "FlaxRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "FlaxResNetForImageClassification",
            "FlaxResNetModel",
            "FlaxResNetPreTrainedModel",
        ]
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
        [
            "FlaxT5EncoderModel",
            "FlaxT5ForConditionalGeneration",
            "FlaxT5Model",
            "FlaxT5PreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].append("FlaxVisionEncoderDecoderModel")
    _import_structure["models.vision_text_dual_encoder"].extend(["FlaxVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    _import_structure["models.wav2vec2"].extend(
        [
            "FlaxWav2Vec2ForCTC",
            "FlaxWav2Vec2ForPreTraining",
            "FlaxWav2Vec2Model",
            "FlaxWav2Vec2PreTrainedModel",
        ]
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
    # Agents
    from .agents import (
        Agent,
        CodeAgent,
        HfApiEngine,
        ManagedAgent,
        PipelineTool,
        ReactAgent,
        ReactCodeAgent,
        ReactJsonAgent,
        Tool,
        Toolbox,
        ToolCollection,
        TransformersEngine,
        launch_gradio_demo,
        load_tool,
        stream_to_gradio,
        tool,
    )

    # Modeling
    # Debugging
    from .cache_utils import (
        Cache,
        CacheConfig,
        DynamicCache,
        EncoderDecoderCache,
        HQQQuantizedCache,
        HybridCache,
        MambaCache,
        OffloadedCache,
        OffloadedStaticCache,
        QuantizedCache,
        QuantizedCacheConfig,
        QuantoQuantizedCache,
        SinkCache,
        SlidingWindowCache,
        StaticCache,
    )
    from .configuration_utils import PretrainedConfig
    from .convert_slow_tokenizer import (
        SLOW_TO_FAST_CONVERTERS,
        convert_slow_tokenizer,
    )

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
        DataCollatorForMultipleChoice,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithFlattening,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
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
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    # TensorFlow
    from .generation import (
        AlternatingCodebooksLogitsProcessor,
        AsyncTextIteratorStreamer,
        BayesianDetectorConfig,
        BayesianDetectorModel,
        BeamScorer,
        BeamSearchScorer,
        ClassifierFreeGuidanceLogitsProcessor,
        CompileConfig,
        ConstrainedBeamSearchScorer,
        Constraint,
        ConstraintListState,
        DisjunctiveConstraint,
        EncoderNoRepeatNGramLogitsProcessor,
        EncoderRepetitionPenaltyLogitsProcessor,
        EosTokenCriteria,
        EpsilonLogitsWarper,
        EtaLogitsWarper,
        ExponentialDecayLengthPenalty,
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
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        GenerationConfig,
        GenerationMixin,
        HammingDiversityLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitNormalization,
        LogitsProcessor,
        LogitsProcessorList,
        MaxLengthCriteria,
        MaxTimeCriteria,
        MinLengthLogitsProcessor,
        MinNewTokensLengthLogitsProcessor,
        MinPLogitsWarper,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PhrasalConstraint,
        PrefixConstrainedLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        SequenceBiasLogitsProcessor,
        StoppingCriteria,
        StoppingCriteriaList,
        StopStringCriteria,
        SuppressTokensAtBeginLogitsProcessor,
        SuppressTokensLogitsProcessor,
        SynthIDTextWatermarkDetector,
        SynthIDTextWatermarkingConfig,
        SynthIDTextWatermarkLogitsProcessor,
        TemperatureLogitsWarper,
        TextIteratorStreamer,
        TextStreamer,
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
        TopKLogitsWarper,
        TopPLogitsWarper,
        TypicalLogitsWarper,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
        WatermarkDetector,
        WatermarkingConfig,
        WatermarkLogitsProcessor,
        WhisperTimeStampLogitsProcessor,
    )
    from .hf_argparser import HfArgumentParser
    from .image_processing_base import ImageProcessingMixin
    from .image_processing_utils import BaseImageProcessor
    from .image_processing_utils_fast import BaseImageProcessorFast
    from .image_utils import ImageFeatureExtractionMixin

    # Integrations
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_swanlab_available,
        is_tensorboard_available,
        is_wandb_available,
    )
    from .integrations.executorch import (
        TorchExportableModuleWithStaticCache,
        convert_and_export_with_cache,
    )
    from .keras_callbacks import KerasMetricCallback, PushToHubCallback
    from .model_debugging_utils import (
        model_addition_debugger,
        model_addition_debugger_context,
    )

    # Model Cards
    from .modelcard import ModelCard
    from .modeling_flax_utils import FlaxPreTrainedModel
    from .modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

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
    from .modeling_tf_utils import (
        TFPreTrainedModel,
        TFSequenceSummary,
        TFSharedEmbeddings,
        shape_list,
    )
    from .modeling_utils import AttentionInterface, PreTrainedModel

    # Fast tokenizers imports
    # TensorFlow model imports
    # Flax model imports
    from .models.albert import (
        AlbertConfig,
        AlbertForMaskedLM,
        AlbertForMultipleChoice,
        AlbertForPreTraining,
        AlbertForQuestionAnswering,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertModel,
        AlbertPreTrainedModel,
        AlbertTokenizer,
        AlbertTokenizerFast,
        FlaxAlbertForMaskedLM,
        FlaxAlbertForMultipleChoice,
        FlaxAlbertForPreTraining,
        FlaxAlbertForQuestionAnswering,
        FlaxAlbertForSequenceClassification,
        FlaxAlbertForTokenClassification,
        FlaxAlbertModel,
        FlaxAlbertPreTrainedModel,
        TFAlbertForMaskedLM,
        TFAlbertForMultipleChoice,
        TFAlbertForPreTraining,
        TFAlbertForQuestionAnswering,
        TFAlbertForSequenceClassification,
        TFAlbertForTokenClassification,
        TFAlbertMainLayer,
        TFAlbertModel,
        TFAlbertPreTrainedModel,
        load_tf_weights_in_albert,
    )
    from .models.align import (
        AlignConfig,
        AlignModel,
        AlignPreTrainedModel,
        AlignProcessor,
        AlignTextConfig,
        AlignTextModel,
        AlignVisionConfig,
        AlignVisionModel,
    )
    from .models.altclip import (
        AltCLIPConfig,
        AltCLIPModel,
        AltCLIPPreTrainedModel,
        AltCLIPProcessor,
        AltCLIPTextConfig,
        AltCLIPTextModel,
        AltCLIPVisionConfig,
        AltCLIPVisionModel,
    )
    from .models.aria import (
        AriaConfig,
        AriaForConditionalGeneration,
        AriaImageProcessor,
        AriaPreTrainedModel,
        AriaProcessor,
        AriaTextConfig,
        AriaTextForCausalLM,
        AriaTextModel,
        AriaTextPreTrainedModel,
    )
    from .models.audio_spectrogram_transformer import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTForAudioClassification,
        ASTModel,
        ASTPreTrainedModel,
    )
    from .models.auto import (
        CONFIG_MAPPING,
        FEATURE_EXTRACTOR_MAPPING,
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
        IMAGE_PROCESSOR_MAPPING,
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
        MODEL_FOR_IMAGE_MAPPING,
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
        MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
        MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
        MODEL_FOR_MASK_GENERATION_MAPPING,
        MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        MODEL_FOR_OBJECT_DETECTION_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_RETRIEVAL_MAPPING,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TEXT_ENCODING_MAPPING,
        MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
        MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
        MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
        MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
        MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
        MODEL_FOR_VISION_2_SEQ_MAPPING,
        MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
        MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
        MODEL_MAPPING,
        MODEL_NAMES_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        PROCESSOR_MAPPING,
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
        TOKENIZER_MAPPING,
        AutoBackbone,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
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
        AutoModelForImageTextToText,
        AutoModelForImageToImage,
        AutoModelForInstanceSegmentation,
        AutoModelForKeypointDetection,
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
        AutoProcessor,
        AutoTokenizer,
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
    from .models.autoformer import (
        AutoformerConfig,
        AutoformerForPrediction,
        AutoformerModel,
        AutoformerPreTrainedModel,
    )
    from .models.aya_vision import (
        AyaVisionConfig,
        AyaVisionForConditionalGeneration,
        AyaVisionPreTrainedModel,
        AyaVisionProcessor,
    )
    from .models.bamba import BambaConfig, BambaForCausalLM, BambaModel, BambaPreTrainedModel
    from .models.bark import (
        BarkCausalModel,
        BarkCoarseConfig,
        BarkCoarseModel,
        BarkConfig,
        BarkFineConfig,
        BarkFineModel,
        BarkModel,
        BarkPreTrainedModel,
        BarkProcessor,
        BarkSemanticConfig,
        BarkSemanticModel,
    )
    from .models.bart import (
        BartConfig,
        BartForCausalLM,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BartForSequenceClassification,
        BartModel,
        BartPreTrainedModel,
        BartPretrainedModel,
        BartTokenizer,
        BartTokenizerFast,
        FlaxBartDecoderPreTrainedModel,
        FlaxBartForCausalLM,
        FlaxBartForConditionalGeneration,
        FlaxBartForQuestionAnswering,
        FlaxBartForSequenceClassification,
        FlaxBartModel,
        FlaxBartPreTrainedModel,
        PretrainedBartModel,
        TFBartForConditionalGeneration,
        TFBartForSequenceClassification,
        TFBartModel,
        TFBartPretrainedModel,
    )
    from .models.barthez import BarthezTokenizer, BarthezTokenizerFast
    from .models.bartpho import BartphoTokenizer
    from .models.beit import (
        BeitBackbone,
        BeitConfig,
        BeitFeatureExtractor,
        BeitForImageClassification,
        BeitForMaskedImageModeling,
        BeitForSemanticSegmentation,
        BeitImageProcessor,
        BeitModel,
        BeitPreTrainedModel,
        FlaxBeitForImageClassification,
        FlaxBeitForMaskedImageModeling,
        FlaxBeitModel,
        FlaxBeitPreTrainedModel,
    )
    from .models.bert import (
        BasicTokenizer,
        BertConfig,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertLMHeadModel,
        BertModel,
        BertPreTrainedModel,
        BertTokenizer,
        BertTokenizerFast,
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
        TFBertTokenizer,
        WordpieceTokenizer,
        load_tf_weights_in_bert,
    )
    from .models.bert_generation import (
        BertGenerationConfig,
        BertGenerationDecoder,
        BertGenerationEncoder,
        BertGenerationPreTrainedModel,
        BertGenerationTokenizer,
        load_tf_weights_in_bert_generation,
    )
    from .models.bert_japanese import (
        BertJapaneseTokenizer,
        CharacterTokenizer,
        MecabTokenizer,
    )
    from .models.bertweet import BertweetTokenizer
    from .models.big_bird import (
        BigBirdConfig,
        BigBirdForCausalLM,
        BigBirdForMaskedLM,
        BigBirdForMultipleChoice,
        BigBirdForPreTraining,
        BigBirdForQuestionAnswering,
        BigBirdForSequenceClassification,
        BigBirdForTokenClassification,
        BigBirdModel,
        BigBirdPreTrainedModel,
        BigBirdTokenizer,
        BigBirdTokenizerFast,
        FlaxBigBirdForCausalLM,
        FlaxBigBirdForMaskedLM,
        FlaxBigBirdForMultipleChoice,
        FlaxBigBirdForPreTraining,
        FlaxBigBirdForQuestionAnswering,
        FlaxBigBirdForSequenceClassification,
        FlaxBigBirdForTokenClassification,
        FlaxBigBirdModel,
        FlaxBigBirdPreTrainedModel,
        load_tf_weights_in_big_bird,
    )
    from .models.bigbird_pegasus import (
        BigBirdPegasusConfig,
        BigBirdPegasusForCausalLM,
        BigBirdPegasusForConditionalGeneration,
        BigBirdPegasusForQuestionAnswering,
        BigBirdPegasusForSequenceClassification,
        BigBirdPegasusModel,
        BigBirdPegasusPreTrainedModel,
    )
    from .models.biogpt import (
        BioGptConfig,
        BioGptForCausalLM,
        BioGptForSequenceClassification,
        BioGptForTokenClassification,
        BioGptModel,
        BioGptPreTrainedModel,
        BioGptTokenizer,
    )
    from .models.bit import (
        BitBackbone,
        BitConfig,
        BitForImageClassification,
        BitImageProcessor,
        BitModel,
        BitPreTrainedModel,
    )
    from .models.blenderbot import (
        BlenderbotConfig,
        BlenderbotForCausalLM,
        BlenderbotForConditionalGeneration,
        BlenderbotModel,
        BlenderbotPreTrainedModel,
        BlenderbotTokenizer,
        BlenderbotTokenizerFast,
        FlaxBlenderbotForConditionalGeneration,
        FlaxBlenderbotModel,
        FlaxBlenderbotPreTrainedModel,
        TFBlenderbotForConditionalGeneration,
        TFBlenderbotModel,
        TFBlenderbotPreTrainedModel,
    )
    from .models.blenderbot_small import (
        BlenderbotSmallConfig,
        BlenderbotSmallForCausalLM,
        BlenderbotSmallForConditionalGeneration,
        BlenderbotSmallModel,
        BlenderbotSmallPreTrainedModel,
        BlenderbotSmallTokenizer,
        BlenderbotSmallTokenizerFast,
        FlaxBlenderbotSmallForConditionalGeneration,
        FlaxBlenderbotSmallModel,
        FlaxBlenderbotSmallPreTrainedModel,
        TFBlenderbotSmallForConditionalGeneration,
        TFBlenderbotSmallModel,
        TFBlenderbotSmallPreTrainedModel,
    )
    from .models.blip import (
        BlipConfig,
        BlipForConditionalGeneration,
        BlipForImageTextRetrieval,
        BlipForQuestionAnswering,
        BlipImageProcessor,
        BlipImageProcessorFast,
        BlipModel,
        BlipPreTrainedModel,
        BlipProcessor,
        BlipTextConfig,
        BlipTextModel,
        BlipVisionConfig,
        BlipVisionModel,
        TFBlipForConditionalGeneration,
        TFBlipForImageTextRetrieval,
        TFBlipForQuestionAnswering,
        TFBlipModel,
        TFBlipPreTrainedModel,
        TFBlipTextModel,
        TFBlipVisionModel,
    )
    from .models.blip_2 import (
        Blip2Config,
        Blip2ForConditionalGeneration,
        Blip2ForImageTextRetrieval,
        Blip2Model,
        Blip2PreTrainedModel,
        Blip2Processor,
        Blip2QFormerConfig,
        Blip2QFormerModel,
        Blip2TextModelWithProjection,
        Blip2VisionConfig,
        Blip2VisionModel,
        Blip2VisionModelWithProjection,
    )
    from .models.bloom import (
        BloomConfig,
        BloomForCausalLM,
        BloomForQuestionAnswering,
        BloomForSequenceClassification,
        BloomForTokenClassification,
        BloomModel,
        BloomPreTrainedModel,
        BloomTokenizerFast,
        FlaxBloomForCausalLM,
        FlaxBloomModel,
        FlaxBloomPreTrainedModel,
    )
    from .models.bridgetower import (
        BridgeTowerConfig,
        BridgeTowerForContrastiveLearning,
        BridgeTowerForImageAndTextRetrieval,
        BridgeTowerForMaskedLM,
        BridgeTowerImageProcessor,
        BridgeTowerModel,
        BridgeTowerPreTrainedModel,
        BridgeTowerProcessor,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from .models.bros import (
        BrosConfig,
        BrosForTokenClassification,
        BrosModel,
        BrosPreTrainedModel,
        BrosProcessor,
        BrosSpadeEEForTokenClassification,
        BrosSpadeELForTokenClassification,
    )
    from .models.byt5 import ByT5Tokenizer
    from .models.camembert import (
        CamembertConfig,
        CamembertForCausalLM,
        CamembertForMaskedLM,
        CamembertForMultipleChoice,
        CamembertForQuestionAnswering,
        CamembertForSequenceClassification,
        CamembertForTokenClassification,
        CamembertModel,
        CamembertPreTrainedModel,
        CamembertTokenizer,
        CamembertTokenizerFast,
        TFCamembertForCausalLM,
        TFCamembertForMaskedLM,
        TFCamembertForMultipleChoice,
        TFCamembertForQuestionAnswering,
        TFCamembertForSequenceClassification,
        TFCamembertForTokenClassification,
        TFCamembertModel,
        TFCamembertPreTrainedModel,
    )
    from .models.canine import (
        CanineConfig,
        CanineForMultipleChoice,
        CanineForQuestionAnswering,
        CanineForSequenceClassification,
        CanineForTokenClassification,
        CanineModel,
        CaninePreTrainedModel,
        CanineTokenizer,
        load_tf_weights_in_canine,
    )
    from .models.chameleon import (
        ChameleonConfig,
        ChameleonForConditionalGeneration,
        ChameleonImageProcessor,
        ChameleonModel,
        ChameleonPreTrainedModel,
        ChameleonProcessor,
        ChameleonVQVAE,
        ChameleonVQVAEConfig,
    )
    from .models.chinese_clip import (
        ChineseCLIPConfig,
        ChineseCLIPFeatureExtractor,
        ChineseCLIPImageProcessor,
        ChineseCLIPModel,
        ChineseCLIPPreTrainedModel,
        ChineseCLIPProcessor,
        ChineseCLIPTextConfig,
        ChineseCLIPTextModel,
        ChineseCLIPVisionConfig,
        ChineseCLIPVisionModel,
    )
    from .models.clap import (
        ClapAudioConfig,
        ClapAudioModel,
        ClapAudioModelWithProjection,
        ClapConfig,
        ClapFeatureExtractor,
        ClapModel,
        ClapPreTrainedModel,
        ClapProcessor,
        ClapTextConfig,
        ClapTextModel,
        ClapTextModelWithProjection,
    )
    from .models.clip import (
        CLIPConfig,
        CLIPFeatureExtractor,
        CLIPForImageClassification,
        CLIPImageProcessor,
        CLIPImageProcessorFast,
        CLIPModel,
        CLIPPreTrainedModel,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPTokenizerFast,
        CLIPVisionConfig,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        FlaxCLIPModel,
        FlaxCLIPPreTrainedModel,
        FlaxCLIPTextModel,
        FlaxCLIPTextModelWithProjection,
        FlaxCLIPTextPreTrainedModel,
        FlaxCLIPVisionModel,
        FlaxCLIPVisionPreTrainedModel,
        TFCLIPModel,
        TFCLIPPreTrainedModel,
        TFCLIPTextModel,
        TFCLIPVisionModel,
    )
    from .models.clipseg import (
        CLIPSegConfig,
        CLIPSegForImageSegmentation,
        CLIPSegModel,
        CLIPSegPreTrainedModel,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegTextModel,
        CLIPSegVisionConfig,
        CLIPSegVisionModel,
    )
    from .models.clvp import (
        ClvpConfig,
        ClvpDecoder,
        ClvpDecoderConfig,
        ClvpEncoder,
        ClvpEncoderConfig,
        ClvpFeatureExtractor,
        ClvpForCausalLM,
        ClvpModel,
        ClvpModelForConditionalGeneration,
        ClvpPreTrainedModel,
        ClvpProcessor,
        ClvpTokenizer,
    )
    from .models.code_llama import CodeLlamaTokenizer, CodeLlamaTokenizerFast
    from .models.codegen import (
        CodeGenConfig,
        CodeGenForCausalLM,
        CodeGenModel,
        CodeGenPreTrainedModel,
        CodeGenTokenizer,
        CodeGenTokenizerFast,
    )
    from .models.cohere import (
        CohereConfig,
        CohereForCausalLM,
        CohereModel,
        CoherePreTrainedModel,
        CohereTokenizerFast,
    )
    from .models.cohere2 import (
        Cohere2Config,
        Cohere2ForCausalLM,
        Cohere2Model,
        Cohere2PreTrainedModel,
    )
    from .models.colpali import (
        ColPaliConfig,
        ColPaliForRetrieval,
        ColPaliPreTrainedModel,
        ColPaliProcessor,
    )
    from .models.conditional_detr import (
        ConditionalDetrConfig,
        ConditionalDetrFeatureExtractor,
        ConditionalDetrForObjectDetection,
        ConditionalDetrForSegmentation,
        ConditionalDetrImageProcessor,
        ConditionalDetrModel,
        ConditionalDetrPreTrainedModel,
    )
    from .models.convbert import (
        ConvBertConfig,
        ConvBertForMaskedLM,
        ConvBertForMultipleChoice,
        ConvBertForQuestionAnswering,
        ConvBertForSequenceClassification,
        ConvBertForTokenClassification,
        ConvBertModel,
        ConvBertPreTrainedModel,
        ConvBertTokenizer,
        ConvBertTokenizerFast,
        TFConvBertForMaskedLM,
        TFConvBertForMultipleChoice,
        TFConvBertForQuestionAnswering,
        TFConvBertForSequenceClassification,
        TFConvBertForTokenClassification,
        TFConvBertModel,
        TFConvBertPreTrainedModel,
        load_tf_weights_in_convbert,
    )
    from .models.convnext import (
        ConvNextBackbone,
        ConvNextConfig,
        ConvNextFeatureExtractor,
        ConvNextForImageClassification,
        ConvNextImageProcessor,
        ConvNextImageProcessorFast,
        ConvNextModel,
        ConvNextPreTrainedModel,
        TFConvNextForImageClassification,
        TFConvNextModel,
        TFConvNextPreTrainedModel,
    )
    from .models.convnextv2 import (
        ConvNextV2Backbone,
        ConvNextV2Config,
        ConvNextV2ForImageClassification,
        ConvNextV2Model,
        ConvNextV2PreTrainedModel,
        TFConvNextV2ForImageClassification,
        TFConvNextV2Model,
        TFConvNextV2PreTrainedModel,
    )
    from .models.cpm import CpmTokenizer, CpmTokenizerFast
    from .models.cpmant import (
        CpmAntConfig,
        CpmAntForCausalLM,
        CpmAntModel,
        CpmAntPreTrainedModel,
        CpmAntTokenizer,
    )
    from .models.ctrl import (
        CTRLConfig,
        CTRLForSequenceClassification,
        CTRLLMHeadModel,
        CTRLModel,
        CTRLPreTrainedModel,
        CTRLTokenizer,
        TFCTRLForSequenceClassification,
        TFCTRLLMHeadModel,
        TFCTRLModel,
        TFCTRLPreTrainedModel,
    )
    from .models.cvt import (
        CvtConfig,
        CvtForImageClassification,
        CvtModel,
        CvtPreTrainedModel,
        TFCvtForImageClassification,
        TFCvtModel,
        TFCvtPreTrainedModel,
    )
    from .models.dab_detr import (
        DabDetrConfig,
        DabDetrForObjectDetection,
        DabDetrModel,
        DabDetrPreTrainedModel,
    )
    from .models.dac import (
        DacConfig,
        DacFeatureExtractor,
        DacModel,
        DacPreTrainedModel,
    )
    from .models.data2vec import (
        Data2VecAudioConfig,
        Data2VecAudioForAudioFrameClassification,
        Data2VecAudioForCTC,
        Data2VecAudioForSequenceClassification,
        Data2VecAudioForXVector,
        Data2VecAudioModel,
        Data2VecAudioPreTrainedModel,
        Data2VecTextConfig,
        Data2VecTextForCausalLM,
        Data2VecTextForMaskedLM,
        Data2VecTextForMultipleChoice,
        Data2VecTextForQuestionAnswering,
        Data2VecTextForSequenceClassification,
        Data2VecTextForTokenClassification,
        Data2VecTextModel,
        Data2VecTextPreTrainedModel,
        Data2VecVisionConfig,
        Data2VecVisionForImageClassification,
        Data2VecVisionForSemanticSegmentation,
        Data2VecVisionModel,
        Data2VecVisionPreTrainedModel,
        TFData2VecVisionForImageClassification,
        TFData2VecVisionForSemanticSegmentation,
        TFData2VecVisionModel,
        TFData2VecVisionPreTrainedModel,
    )

    # PyTorch model imports
    from .models.dbrx import (
        DbrxConfig,
        DbrxForCausalLM,
        DbrxModel,
        DbrxPreTrainedModel,
    )
    from .models.deberta import (
        DebertaConfig,
        DebertaForMaskedLM,
        DebertaForQuestionAnswering,
        DebertaForSequenceClassification,
        DebertaForTokenClassification,
        DebertaModel,
        DebertaPreTrainedModel,
        DebertaTokenizer,
        DebertaTokenizerFast,
        TFDebertaForMaskedLM,
        TFDebertaForQuestionAnswering,
        TFDebertaForSequenceClassification,
        TFDebertaForTokenClassification,
        TFDebertaModel,
        TFDebertaPreTrainedModel,
    )
    from .models.deberta_v2 import (
        DebertaV2Config,
        DebertaV2ForMaskedLM,
        DebertaV2ForMultipleChoice,
        DebertaV2ForQuestionAnswering,
        DebertaV2ForSequenceClassification,
        DebertaV2ForTokenClassification,
        DebertaV2Model,
        DebertaV2PreTrainedModel,
        DebertaV2Tokenizer,
        DebertaV2TokenizerFast,
        TFDebertaV2ForMaskedLM,
        TFDebertaV2ForMultipleChoice,
        TFDebertaV2ForQuestionAnswering,
        TFDebertaV2ForSequenceClassification,
        TFDebertaV2ForTokenClassification,
        TFDebertaV2Model,
        TFDebertaV2PreTrainedModel,
    )
    from .models.decision_transformer import (
        DecisionTransformerConfig,
        DecisionTransformerGPT2Model,
        DecisionTransformerGPT2PreTrainedModel,
        DecisionTransformerModel,
        DecisionTransformerPreTrainedModel,
    )
    from .models.deepseek_v3 import (
        DeepseekV3Config,
        DeepseekV3ForCausalLM,
        DeepseekV3Model,
        DeepseekV3PreTrainedModel,
    )
    from .models.deformable_detr import (
        DeformableDetrConfig,
        DeformableDetrFeatureExtractor,
        DeformableDetrForObjectDetection,
        DeformableDetrImageProcessor,
        DeformableDetrImageProcessorFast,
        DeformableDetrModel,
        DeformableDetrPreTrainedModel,
    )
    from .models.deit import (
        DeiTConfig,
        DeiTFeatureExtractor,
        DeiTForImageClassification,
        DeiTForImageClassificationWithTeacher,
        DeiTForMaskedImageModeling,
        DeiTImageProcessor,
        DeiTImageProcessorFast,
        DeiTModel,
        DeiTPreTrainedModel,
        TFDeiTForImageClassification,
        TFDeiTForImageClassificationWithTeacher,
        TFDeiTForMaskedImageModeling,
        TFDeiTModel,
        TFDeiTPreTrainedModel,
    )
    from .models.deprecated.deta import (
        DetaConfig,
        DetaForObjectDetection,
        DetaImageProcessor,
        DetaModel,
        DetaPreTrainedModel,
    )
    from .models.deprecated.efficientformer import (
        EfficientFormerConfig,
        EfficientFormerForImageClassification,
        EfficientFormerForImageClassificationWithTeacher,
        EfficientFormerImageProcessor,
        EfficientFormerModel,
        EfficientFormerPreTrainedModel,
        TFEfficientFormerForImageClassification,
        TFEfficientFormerForImageClassificationWithTeacher,
        TFEfficientFormerModel,
        TFEfficientFormerPreTrainedModel,
    )
    from .models.deprecated.ernie_m import (
        ErnieMConfig,
        ErnieMForInformationExtraction,
        ErnieMForMultipleChoice,
        ErnieMForQuestionAnswering,
        ErnieMForSequenceClassification,
        ErnieMForTokenClassification,
        ErnieMModel,
        ErnieMPreTrainedModel,
        ErnieMTokenizer,
    )
    from .models.deprecated.gptsan_japanese import (
        GPTSanJapaneseConfig,
        GPTSanJapaneseForConditionalGeneration,
        GPTSanJapaneseModel,
        GPTSanJapanesePreTrainedModel,
        GPTSanJapaneseTokenizer,
    )
    from .models.deprecated.graphormer import (
        GraphormerConfig,
        GraphormerForGraphClassification,
        GraphormerModel,
        GraphormerPreTrainedModel,
    )
    from .models.deprecated.jukebox import (
        JukeboxConfig,
        JukeboxModel,
        JukeboxPreTrainedModel,
        JukeboxPrior,
        JukeboxPriorConfig,
        JukeboxTokenizer,
        JukeboxVQVAE,
        JukeboxVQVAEConfig,
    )
    from .models.deprecated.mctct import (
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTForCTC,
        MCTCTModel,
        MCTCTPreTrainedModel,
        MCTCTProcessor,
    )
    from .models.deprecated.mega import (
        MegaConfig,
        MegaForCausalLM,
        MegaForMaskedLM,
        MegaForMultipleChoice,
        MegaForQuestionAnswering,
        MegaForSequenceClassification,
        MegaForTokenClassification,
        MegaModel,
        MegaPreTrainedModel,
    )
    from .models.deprecated.mmbt import (
        MMBTConfig,
        MMBTForClassification,
        MMBTModel,
        ModalEmbeddings,
    )
    from .models.deprecated.nat import (
        NatBackbone,
        NatConfig,
        NatForImageClassification,
        NatModel,
        NatPreTrainedModel,
    )
    from .models.deprecated.nezha import (
        NezhaConfig,
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
    from .models.deprecated.open_llama import (
        OpenLlamaConfig,
        OpenLlamaForCausalLM,
        OpenLlamaForSequenceClassification,
        OpenLlamaModel,
        OpenLlamaPreTrainedModel,
    )
    from .models.deprecated.qdqbert import (
        QDQBertConfig,
        QDQBertForMaskedLM,
        QDQBertForMultipleChoice,
        QDQBertForNextSentencePrediction,
        QDQBertForQuestionAnswering,
        QDQBertForSequenceClassification,
        QDQBertForTokenClassification,
        QDQBertLMHeadModel,
        QDQBertModel,
        QDQBertPreTrainedModel,
        load_tf_weights_in_qdqbert,
    )
    from .models.deprecated.realm import (
        RealmConfig,
        RealmEmbedder,
        RealmForOpenQA,
        RealmKnowledgeAugEncoder,
        RealmPreTrainedModel,
        RealmReader,
        RealmRetriever,
        RealmScorer,
        RealmTokenizer,
        RealmTokenizerFast,
        load_tf_weights_in_realm,
    )
    from .models.deprecated.retribert import (
        RetriBertConfig,
        RetriBertModel,
        RetriBertPreTrainedModel,
        RetriBertTokenizer,
        RetriBertTokenizerFast,
    )
    from .models.deprecated.speech_to_text_2 import (
        Speech2Text2Config,
        Speech2Text2ForCausalLM,
        Speech2Text2PreTrainedModel,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    from .models.deprecated.tapex import TapexTokenizer
    from .models.deprecated.trajectory_transformer import (
        TrajectoryTransformerConfig,
        TrajectoryTransformerModel,
        TrajectoryTransformerPreTrainedModel,
    )
    from .models.deprecated.transfo_xl import (
        AdaptiveEmbedding,
        TFAdaptiveEmbedding,
        TFTransfoXLForSequenceClassification,
        TFTransfoXLLMHeadModel,
        TFTransfoXLMainLayer,
        TFTransfoXLModel,
        TFTransfoXLPreTrainedModel,
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLForSequenceClassification,
        TransfoXLLMHeadModel,
        TransfoXLModel,
        TransfoXLPreTrainedModel,
        TransfoXLTokenizer,
        load_tf_weights_in_transfo_xl,
    )
    from .models.deprecated.tvlt import (
        TvltConfig,
        TvltFeatureExtractor,
        TvltForAudioVisualClassification,
        TvltForPreTraining,
        TvltImageProcessor,
        TvltModel,
        TvltPreTrainedModel,
        TvltProcessor,
    )
    from .models.deprecated.van import (
        VanConfig,
        VanForImageClassification,
        VanModel,
        VanPreTrainedModel,
    )
    from .models.deprecated.vit_hybrid import (
        ViTHybridConfig,
        ViTHybridForImageClassification,
        ViTHybridImageProcessor,
        ViTHybridModel,
        ViTHybridPreTrainedModel,
    )
    from .models.deprecated.xlm_prophetnet import (
        XLMProphetNetConfig,
        XLMProphetNetDecoder,
        XLMProphetNetEncoder,
        XLMProphetNetForCausalLM,
        XLMProphetNetForConditionalGeneration,
        XLMProphetNetModel,
        XLMProphetNetPreTrainedModel,
        XLMProphetNetTokenizer,
    )
    from .models.depth_anything import (
        DepthAnythingConfig,
        DepthAnythingForDepthEstimation,
        DepthAnythingPreTrainedModel,
    )
    from .models.depth_pro import (
        DepthProConfig,
        DepthProForDepthEstimation,
        DepthProImageProcessor,
        DepthProImageProcessorFast,
        DepthProModel,
        DepthProPreTrainedModel,
    )
    from .models.detr import (
        DetrConfig,
        DetrFeatureExtractor,
        DetrForObjectDetection,
        DetrForSegmentation,
        DetrImageProcessor,
        DetrImageProcessorFast,
        DetrModel,
        DetrPreTrainedModel,
    )
    from .models.diffllama import (
        DiffLlamaConfig,
        DiffLlamaForCausalLM,
        DiffLlamaForQuestionAnswering,
        DiffLlamaForSequenceClassification,
        DiffLlamaForTokenClassification,
        DiffLlamaModel,
        DiffLlamaPreTrainedModel,
    )
    from .models.dinat import (
        DinatBackbone,
        DinatConfig,
        DinatForImageClassification,
        DinatModel,
        DinatPreTrainedModel,
    )
    from .models.dinov2 import (
        Dinov2Backbone,
        Dinov2Config,
        Dinov2ForImageClassification,
        Dinov2Model,
        Dinov2PreTrainedModel,
        FlaxDinov2ForImageClassification,
        FlaxDinov2Model,
        FlaxDinov2PreTrainedModel,
    )
    from .models.dinov2_with_registers import (
        Dinov2WithRegistersBackbone,
        Dinov2WithRegistersConfig,
        Dinov2WithRegistersForImageClassification,
        Dinov2WithRegistersModel,
        Dinov2WithRegistersPreTrainedModel,
    )
    from .models.distilbert import (
        DistilBertConfig,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
        DistilBertPreTrainedModel,
        DistilBertTokenizer,
        DistilBertTokenizerFast,
        FlaxDistilBertForMaskedLM,
        FlaxDistilBertForMultipleChoice,
        FlaxDistilBertForQuestionAnswering,
        FlaxDistilBertForSequenceClassification,
        FlaxDistilBertForTokenClassification,
        FlaxDistilBertModel,
        FlaxDistilBertPreTrainedModel,
        TFDistilBertForMaskedLM,
        TFDistilBertForMultipleChoice,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertMainLayer,
        TFDistilBertModel,
        TFDistilBertPreTrainedModel,
    )
    from .models.donut import (
        DonutFeatureExtractor,
        DonutImageProcessor,
        DonutProcessor,
        DonutSwinConfig,
        DonutSwinModel,
        DonutSwinPreTrainedModel,
    )
    from .models.dpr import (
        DPRConfig,
        DPRContextEncoder,
        DPRContextEncoderTokenizer,
        DPRContextEncoderTokenizerFast,
        DPRPretrainedContextEncoder,
        DPRPreTrainedModel,
        DPRPretrainedQuestionEncoder,
        DPRPretrainedReader,
        DPRQuestionEncoder,
        DPRQuestionEncoderTokenizer,
        DPRQuestionEncoderTokenizerFast,
        DPRReader,
        DPRReaderOutput,
        DPRReaderTokenizer,
        DPRReaderTokenizerFast,
        TFDPRContextEncoder,
        TFDPRPretrainedContextEncoder,
        TFDPRPretrainedQuestionEncoder,
        TFDPRPretrainedReader,
        TFDPRQuestionEncoder,
        TFDPRReader,
    )
    from .models.dpt import (
        DPTConfig,
        DPTFeatureExtractor,
        DPTForDepthEstimation,
        DPTForSemanticSegmentation,
        DPTImageProcessor,
        DPTModel,
        DPTPreTrainedModel,
    )
    from .models.efficientnet import (
        EfficientNetConfig,
        EfficientNetForImageClassification,
        EfficientNetImageProcessor,
        EfficientNetModel,
        EfficientNetPreTrainedModel,
    )
    from .models.electra import (
        ElectraConfig,
        ElectraForCausalLM,
        ElectraForMaskedLM,
        ElectraForMultipleChoice,
        ElectraForPreTraining,
        ElectraForQuestionAnswering,
        ElectraForSequenceClassification,
        ElectraForTokenClassification,
        ElectraModel,
        ElectraPreTrainedModel,
        ElectraTokenizer,
        ElectraTokenizerFast,
        FlaxElectraForCausalLM,
        FlaxElectraForMaskedLM,
        FlaxElectraForMultipleChoice,
        FlaxElectraForPreTraining,
        FlaxElectraForQuestionAnswering,
        FlaxElectraForSequenceClassification,
        FlaxElectraForTokenClassification,
        FlaxElectraModel,
        FlaxElectraPreTrainedModel,
        TFElectraForMaskedLM,
        TFElectraForMultipleChoice,
        TFElectraForPreTraining,
        TFElectraForQuestionAnswering,
        TFElectraForSequenceClassification,
        TFElectraForTokenClassification,
        TFElectraModel,
        TFElectraPreTrainedModel,
        load_tf_weights_in_electra,
    )
    from .models.emu3 import (
        Emu3Config,
        Emu3ForCausalLM,
        Emu3ForConditionalGeneration,
        Emu3ImageProcessor,
        Emu3PreTrainedModel,
        Emu3Processor,
        Emu3TextConfig,
        Emu3TextModel,
        Emu3VQVAE,
        Emu3VQVAEConfig,
    )
    from .models.encodec import (
        EncodecConfig,
        EncodecFeatureExtractor,
        EncodecModel,
        EncodecPreTrainedModel,
    )
    from .models.encoder_decoder import (
        EncoderDecoderConfig,
        EncoderDecoderModel,
        FlaxEncoderDecoderModel,
        TFEncoderDecoderModel,
    )
    from .models.ernie import (
        ErnieConfig,
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
    from .models.esm import (
        EsmConfig,
        EsmFoldPreTrainedModel,
        EsmForMaskedLM,
        EsmForProteinFolding,
        EsmForSequenceClassification,
        EsmForTokenClassification,
        EsmModel,
        EsmPreTrainedModel,
        EsmTokenizer,
        TFEsmForMaskedLM,
        TFEsmForSequenceClassification,
        TFEsmForTokenClassification,
        TFEsmModel,
        TFEsmPreTrainedModel,
    )
    from .models.falcon import (
        FalconConfig,
        FalconForCausalLM,
        FalconForQuestionAnswering,
        FalconForSequenceClassification,
        FalconForTokenClassification,
        FalconModel,
        FalconPreTrainedModel,
    )
    from .models.falcon_mamba import (
        FalconMambaConfig,
        FalconMambaForCausalLM,
        FalconMambaModel,
        FalconMambaPreTrainedModel,
    )
    from .models.fastspeech2_conformer import (
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGan,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerModel,
        FastSpeech2ConformerPreTrainedModel,
        FastSpeech2ConformerTokenizer,
        FastSpeech2ConformerWithHifiGan,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    from .models.flaubert import (
        FlaubertConfig,
        FlaubertForMultipleChoice,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
        FlaubertForTokenClassification,
        FlaubertModel,
        FlaubertPreTrainedModel,
        FlaubertTokenizer,
        FlaubertWithLMHeadModel,
        TFFlaubertForMultipleChoice,
        TFFlaubertForQuestionAnsweringSimple,
        TFFlaubertForSequenceClassification,
        TFFlaubertForTokenClassification,
        TFFlaubertModel,
        TFFlaubertPreTrainedModel,
        TFFlaubertWithLMHeadModel,
    )
    from .models.flava import (
        FlavaConfig,
        FlavaFeatureExtractor,
        FlavaForPreTraining,
        FlavaImageCodebook,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaImageModel,
        FlavaImageProcessor,
        FlavaModel,
        FlavaMultimodalConfig,
        FlavaMultimodalModel,
        FlavaPreTrainedModel,
        FlavaProcessor,
        FlavaTextConfig,
        FlavaTextModel,
    )
    from .models.fnet import (
        FNetConfig,
        FNetForMaskedLM,
        FNetForMultipleChoice,
        FNetForNextSentencePrediction,
        FNetForPreTraining,
        FNetForQuestionAnswering,
        FNetForSequenceClassification,
        FNetForTokenClassification,
        FNetModel,
        FNetPreTrainedModel,
        FNetTokenizer,
        FNetTokenizerFast,
    )
    from .models.focalnet import (
        FocalNetBackbone,
        FocalNetConfig,
        FocalNetForImageClassification,
        FocalNetForMaskedImageModeling,
        FocalNetModel,
        FocalNetPreTrainedModel,
    )
    from .models.fsmt import (
        FSMTConfig,
        FSMTForConditionalGeneration,
        FSMTModel,
        FSMTTokenizer,
        PretrainedFSMTModel,
    )
    from .models.funnel import (
        FunnelBaseModel,
        FunnelConfig,
        FunnelForMaskedLM,
        FunnelForMultipleChoice,
        FunnelForPreTraining,
        FunnelForQuestionAnswering,
        FunnelForSequenceClassification,
        FunnelForTokenClassification,
        FunnelModel,
        FunnelPreTrainedModel,
        FunnelTokenizer,
        FunnelTokenizerFast,
        TFFunnelBaseModel,
        TFFunnelForMaskedLM,
        TFFunnelForMultipleChoice,
        TFFunnelForPreTraining,
        TFFunnelForQuestionAnswering,
        TFFunnelForSequenceClassification,
        TFFunnelForTokenClassification,
        TFFunnelModel,
        TFFunnelPreTrainedModel,
        load_tf_weights_in_funnel,
    )
    from .models.fuyu import (
        FuyuConfig,
        FuyuForCausalLM,
        FuyuImageProcessor,
        FuyuPreTrainedModel,
        FuyuProcessor,
    )
    from .models.gemma import (
        FlaxGemmaForCausalLM,
        FlaxGemmaModel,
        FlaxGemmaPreTrainedModel,
        GemmaConfig,
        GemmaForCausalLM,
        GemmaForSequenceClassification,
        GemmaForTokenClassification,
        GemmaModel,
        GemmaPreTrainedModel,
        GemmaTokenizer,
        GemmaTokenizerFast,
    )
    from .models.gemma2 import (
        Gemma2Config,
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
        Gemma2PreTrainedModel,
    )
    from .models.gemma3 import (
        Gemma3Config,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        Gemma3ImageProcessor,
        Gemma3ImageProcessorFast,
        Gemma3PreTrainedModel,
        Gemma3Processor,
        Gemma3TextConfig,
        Gemma3TextModel,
    )
    from .models.git import (
        GitConfig,
        GitForCausalLM,
        GitModel,
        GitPreTrainedModel,
        GitProcessor,
        GitVisionConfig,
        GitVisionModel,
    )
    from .models.glm import (
        GlmConfig,
        GlmForCausalLM,
        GlmForSequenceClassification,
        GlmForTokenClassification,
        GlmModel,
        GlmPreTrainedModel,
    )
    from .models.glpn import (
        GLPNConfig,
        GLPNFeatureExtractor,
        GLPNForDepthEstimation,
        GLPNImageProcessor,
        GLPNModel,
        GLPNPreTrainedModel,
    )
    from .models.got_ocr2 import (
        GotOcr2Config,
        GotOcr2ForConditionalGeneration,
        GotOcr2ImageProcessor,
        GotOcr2ImageProcessorFast,
        GotOcr2PreTrainedModel,
        GotOcr2Processor,
        GotOcr2VisionConfig,
    )
    from .models.gpt2 import (
        FlaxGPT2LMHeadModel,
        FlaxGPT2Model,
        FlaxGPT2PreTrainedModel,
        GPT2Config,
        GPT2DoubleHeadsModel,
        GPT2ForQuestionAnswering,
        GPT2ForSequenceClassification,
        GPT2ForTokenClassification,
        GPT2LMHeadModel,
        GPT2Model,
        GPT2PreTrainedModel,
        GPT2Tokenizer,
        GPT2TokenizerFast,
        TFGPT2DoubleHeadsModel,
        TFGPT2ForSequenceClassification,
        TFGPT2LMHeadModel,
        TFGPT2MainLayer,
        TFGPT2Model,
        TFGPT2PreTrainedModel,
        TFGPT2Tokenizer,
        load_tf_weights_in_gpt2,
    )
    from .models.gpt_bigcode import (
        GPTBigCodeConfig,
        GPTBigCodeForCausalLM,
        GPTBigCodeForSequenceClassification,
        GPTBigCodeForTokenClassification,
        GPTBigCodeModel,
        GPTBigCodePreTrainedModel,
    )
    from .models.gpt_neo import (
        FlaxGPTNeoForCausalLM,
        FlaxGPTNeoModel,
        FlaxGPTNeoPreTrainedModel,
        GPTNeoConfig,
        GPTNeoForCausalLM,
        GPTNeoForQuestionAnswering,
        GPTNeoForSequenceClassification,
        GPTNeoForTokenClassification,
        GPTNeoModel,
        GPTNeoPreTrainedModel,
        load_tf_weights_in_gpt_neo,
    )
    from .models.gpt_neox import (
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        GPTNeoXForQuestionAnswering,
        GPTNeoXForSequenceClassification,
        GPTNeoXForTokenClassification,
        GPTNeoXModel,
        GPTNeoXPreTrainedModel,
        GPTNeoXTokenizerFast,
    )
    from .models.gpt_neox_japanese import (
        GPTNeoXJapaneseConfig,
        GPTNeoXJapaneseForCausalLM,
        GPTNeoXJapaneseModel,
        GPTNeoXJapanesePreTrainedModel,
        GPTNeoXJapaneseTokenizer,
    )
    from .models.gpt_sw3 import GPTSw3Tokenizer
    from .models.gptj import (
        FlaxGPTJForCausalLM,
        FlaxGPTJModel,
        FlaxGPTJPreTrainedModel,
        GPTJConfig,
        GPTJForCausalLM,
        GPTJForQuestionAnswering,
        GPTJForSequenceClassification,
        GPTJModel,
        GPTJPreTrainedModel,
        TFGPTJForCausalLM,
        TFGPTJForQuestionAnswering,
        TFGPTJForSequenceClassification,
        TFGPTJModel,
        TFGPTJPreTrainedModel,
    )
    from .models.granite import (
        GraniteConfig,
        GraniteForCausalLM,
        GraniteModel,
        GranitePreTrainedModel,
    )
    from .models.granitemoe import (
        GraniteMoeConfig,
        GraniteMoeForCausalLM,
        GraniteMoeModel,
        GraniteMoePreTrainedModel,
    )
    from .models.granitemoeshared import (
        GraniteMoeSharedConfig,
        GraniteMoeSharedForCausalLM,
        GraniteMoeSharedModel,
        GraniteMoeSharedPreTrainedModel,
    )
    from .models.grounding_dino import (
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        GroundingDinoImageProcessor,
        GroundingDinoModel,
        GroundingDinoPreTrainedModel,
        GroundingDinoProcessor,
    )
    from .models.groupvit import (
        GroupViTConfig,
        GroupViTModel,
        GroupViTPreTrainedModel,
        GroupViTTextConfig,
        GroupViTTextModel,
        GroupViTVisionConfig,
        GroupViTVisionModel,
        TFGroupViTModel,
        TFGroupViTPreTrainedModel,
        TFGroupViTTextModel,
        TFGroupViTVisionModel,
    )
    from .models.helium import (
        HeliumConfig,
        HeliumForCausalLM,
        HeliumForSequenceClassification,
        HeliumForTokenClassification,
        HeliumModel,
        HeliumPreTrainedModel,
    )
    from .models.herbert import HerbertTokenizer, HerbertTokenizerFast
    from .models.hiera import (
        HieraBackbone,
        HieraConfig,
        HieraForImageClassification,
        HieraForPreTraining,
        HieraModel,
        HieraPreTrainedModel,
    )
    from .models.hubert import (
        HubertConfig,
        HubertForCTC,
        HubertForSequenceClassification,
        HubertModel,
        HubertPreTrainedModel,
        TFHubertForCTC,
        TFHubertModel,
        TFHubertPreTrainedModel,
    )
    from .models.ibert import (
        IBertConfig,
        IBertForMaskedLM,
        IBertForMultipleChoice,
        IBertForQuestionAnswering,
        IBertForSequenceClassification,
        IBertForTokenClassification,
        IBertModel,
        IBertPreTrainedModel,
    )
    from .models.idefics import (
        IdeficsConfig,
        IdeficsForVisionText2Text,
        IdeficsImageProcessor,
        IdeficsModel,
        IdeficsPreTrainedModel,
        IdeficsProcessor,
        TFIdeficsForVisionText2Text,
        TFIdeficsModel,
        TFIdeficsPreTrainedModel,
    )
    from .models.idefics2 import (
        Idefics2Config,
        Idefics2ForConditionalGeneration,
        Idefics2ImageProcessor,
        Idefics2Model,
        Idefics2PreTrainedModel,
        Idefics2Processor,
    )
    from .models.idefics3 import (
        Idefics3Config,
        Idefics3ForConditionalGeneration,
        Idefics3ImageProcessor,
        Idefics3Model,
        Idefics3PreTrainedModel,
        Idefics3Processor,
        Idefics3VisionConfig,
        Idefics3VisionTransformer,
    )
    from .models.ijepa import (
        IJepaConfig,
        IJepaForImageClassification,
        IJepaModel,
        IJepaPreTrainedModel,
    )
    from .models.imagegpt import (
        ImageGPTConfig,
        ImageGPTFeatureExtractor,
        ImageGPTForCausalImageModeling,
        ImageGPTForImageClassification,
        ImageGPTImageProcessor,
        ImageGPTModel,
        ImageGPTPreTrainedModel,
        load_tf_weights_in_imagegpt,
    )
    from .models.informer import (
        InformerConfig,
        InformerForPrediction,
        InformerModel,
        InformerPreTrainedModel,
    )
    from .models.instructblip import (
        InstructBlipConfig,
        InstructBlipForConditionalGeneration,
        InstructBlipPreTrainedModel,
        InstructBlipProcessor,
        InstructBlipQFormerConfig,
        InstructBlipQFormerModel,
        InstructBlipVisionConfig,
        InstructBlipVisionModel,
    )
    from .models.instructblipvideo import (
        InstructBlipVideoConfig,
        InstructBlipVideoForConditionalGeneration,
        InstructBlipVideoImageProcessor,
        InstructBlipVideoPreTrainedModel,
        InstructBlipVideoProcessor,
        InstructBlipVideoQFormerConfig,
        InstructBlipVideoQFormerModel,
        InstructBlipVideoVisionConfig,
        InstructBlipVideoVisionModel,
    )
    from .models.jamba import (
        JambaConfig,
        JambaForCausalLM,
        JambaForSequenceClassification,
        JambaModel,
        JambaPreTrainedModel,
    )
    from .models.jetmoe import (
        JetMoeConfig,
        JetMoeForCausalLM,
        JetMoeForSequenceClassification,
        JetMoeModel,
        JetMoePreTrainedModel,
    )
    from .models.kosmos2 import (
        Kosmos2Config,
        Kosmos2ForConditionalGeneration,
        Kosmos2Model,
        Kosmos2PreTrainedModel,
        Kosmos2Processor,
    )
    from .models.layoutlm import (
        LayoutLMConfig,
        LayoutLMForMaskedLM,
        LayoutLMForQuestionAnswering,
        LayoutLMForSequenceClassification,
        LayoutLMForTokenClassification,
        LayoutLMModel,
        LayoutLMPreTrainedModel,
        LayoutLMTokenizer,
        LayoutLMTokenizerFast,
        TFLayoutLMForMaskedLM,
        TFLayoutLMForQuestionAnswering,
        TFLayoutLMForSequenceClassification,
        TFLayoutLMForTokenClassification,
        TFLayoutLMMainLayer,
        TFLayoutLMModel,
        TFLayoutLMPreTrainedModel,
    )
    from .models.layoutlmv2 import (
        LayoutLMv2Config,
        LayoutLMv2FeatureExtractor,
        LayoutLMv2ForQuestionAnswering,
        LayoutLMv2ForSequenceClassification,
        LayoutLMv2ForTokenClassification,
        LayoutLMv2ImageProcessor,
        LayoutLMv2Model,
        LayoutLMv2PreTrainedModel,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
        LayoutLMv2TokenizerFast,
    )
    from .models.layoutlmv3 import (
        LayoutLMv3Config,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ForQuestionAnswering,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ImageProcessor,
        LayoutLMv3Model,
        LayoutLMv3PreTrainedModel,
        LayoutLMv3Processor,
        LayoutLMv3Tokenizer,
        LayoutLMv3TokenizerFast,
        TFLayoutLMv3ForQuestionAnswering,
        TFLayoutLMv3ForSequenceClassification,
        TFLayoutLMv3ForTokenClassification,
        TFLayoutLMv3Model,
        TFLayoutLMv3PreTrainedModel,
    )
    from .models.layoutxlm import LayoutXLMProcessor, LayoutXLMTokenizer, LayoutXLMTokenizerFast
    from .models.led import (
        LEDConfig,
        LEDForConditionalGeneration,
        LEDForQuestionAnswering,
        LEDForSequenceClassification,
        LEDModel,
        LEDPreTrainedModel,
        LEDTokenizer,
        LEDTokenizerFast,
        TFLEDForConditionalGeneration,
        TFLEDModel,
        TFLEDPreTrainedModel,
    )
    from .models.levit import (
        LevitConfig,
        LevitFeatureExtractor,
        LevitForImageClassification,
        LevitForImageClassificationWithTeacher,
        LevitImageProcessor,
        LevitModel,
        LevitPreTrainedModel,
    )
    from .models.lilt import (
        LiltConfig,
        LiltForQuestionAnswering,
        LiltForSequenceClassification,
        LiltForTokenClassification,
        LiltModel,
        LiltPreTrainedModel,
    )
    from .models.llama import (
        FlaxLlamaForCausalLM,
        FlaxLlamaModel,
        FlaxLlamaPreTrainedModel,
        LlamaConfig,
        LlamaForCausalLM,
        LlamaForQuestionAnswering,
        LlamaForSequenceClassification,
        LlamaForTokenClassification,
        LlamaModel,
        LlamaPreTrainedModel,
        LlamaTokenizer,
        LlamaTokenizerFast,
    )
    from .models.llama4 import (
        Llama4Config,
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4ImageProcessorFast,
        Llama4PreTrainedModel,
        Llama4Processor,
        Llama4TextConfig,
        Llama4TextModel,
        Llama4VisionConfig,
        Llama4VisionModel,
    )
    from .models.llava import (
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaImageProcessor,
        LlavaImageProcessorFast,
        LlavaPreTrainedModel,
        LlavaProcessor,
    )
    from .models.llava_next import (
        LlavaNextConfig,
        LlavaNextForConditionalGeneration,
        LlavaNextImageProcessor,
        LlavaNextImageProcessorFast,
        LlavaNextPreTrainedModel,
        LlavaNextProcessor,
    )
    from .models.llava_next_video import (
        LlavaNextVideoConfig,
        LlavaNextVideoForConditionalGeneration,
        LlavaNextVideoImageProcessor,
        LlavaNextVideoPreTrainedModel,
        LlavaNextVideoProcessor,
    )
    from .models.llava_onevision import (
        LlavaOnevisionConfig,
        LlavaOnevisionForConditionalGeneration,
        LlavaOnevisionImageProcessor,
        LlavaOnevisionImageProcessorFast,
        LlavaOnevisionPreTrainedModel,
        LlavaOnevisionProcessor,
        LlavaOnevisionVideoProcessor,
    )
    from .models.longformer import (
        LongformerConfig,
        LongformerForMaskedLM,
        LongformerForMultipleChoice,
        LongformerForQuestionAnswering,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerModel,
        LongformerPreTrainedModel,
        LongformerTokenizer,
        LongformerTokenizerFast,
        TFLongformerForMaskedLM,
        TFLongformerForMultipleChoice,
        TFLongformerForQuestionAnswering,
        TFLongformerForSequenceClassification,
        TFLongformerForTokenClassification,
        TFLongformerModel,
        TFLongformerPreTrainedModel,
    )
    from .models.longt5 import (
        FlaxLongT5ForConditionalGeneration,
        FlaxLongT5Model,
        FlaxLongT5PreTrainedModel,
        LongT5Config,
        LongT5EncoderModel,
        LongT5ForConditionalGeneration,
        LongT5Model,
        LongT5PreTrainedModel,
    )
    from .models.luke import (
        LukeConfig,
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
        LukeTokenizer,
    )
    from .models.lxmert import (
        LxmertConfig,
        LxmertEncoder,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
        LxmertPreTrainedModel,
        LxmertTokenizer,
        LxmertTokenizerFast,
        LxmertVisualFeatureEncoder,
        TFLxmertForPreTraining,
        TFLxmertMainLayer,
        TFLxmertModel,
        TFLxmertPreTrainedModel,
        TFLxmertVisualFeatureEncoder,
    )
    from .models.m2m_100 import (
        M2M100Config,
        M2M100ForConditionalGeneration,
        M2M100Model,
        M2M100PreTrainedModel,
        M2M100Tokenizer,
    )
    from .models.mamba import (
        MambaConfig,
        MambaForCausalLM,
        MambaModel,
        MambaPreTrainedModel,
    )
    from .models.mamba2 import (
        Mamba2Config,
        Mamba2ForCausalLM,
        Mamba2Model,
        Mamba2PreTrainedModel,
    )
    from .models.marian import (
        FlaxMarianModel,
        FlaxMarianMTModel,
        FlaxMarianPreTrainedModel,
        MarianConfig,
        MarianForCausalLM,
        MarianModel,
        MarianMTModel,
        MarianPreTrainedModel,
        MarianTokenizer,
        TFMarianModel,
        TFMarianMTModel,
        TFMarianPreTrainedModel,
    )
    from .models.markuplm import (
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMForQuestionAnswering,
        MarkupLMForSequenceClassification,
        MarkupLMForTokenClassification,
        MarkupLMModel,
        MarkupLMPreTrainedModel,
        MarkupLMProcessor,
        MarkupLMTokenizer,
        MarkupLMTokenizerFast,
    )
    from .models.mask2former import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
        Mask2FormerModel,
        Mask2FormerPreTrainedModel,
    )
    from .models.maskformer import (
        MaskFormerConfig,
        MaskFormerFeatureExtractor,
        MaskFormerForInstanceSegmentation,
        MaskFormerImageProcessor,
        MaskFormerModel,
        MaskFormerPreTrainedModel,
        MaskFormerSwinBackbone,
        MaskFormerSwinConfig,
    )
    from .models.mbart import (
        FlaxMBartForConditionalGeneration,
        FlaxMBartForQuestionAnswering,
        FlaxMBartForSequenceClassification,
        FlaxMBartModel,
        FlaxMBartPreTrainedModel,
        MBartConfig,
        MBartForCausalLM,
        MBartForConditionalGeneration,
        MBartForQuestionAnswering,
        MBartForSequenceClassification,
        MBartModel,
        MBartPreTrainedModel,
        MBartTokenizer,
        MBartTokenizerFast,
        TFMBartForConditionalGeneration,
        TFMBartModel,
        TFMBartPreTrainedModel,
    )
    from .models.mbart50 import MBart50Tokenizer, MBart50TokenizerFast
    from .models.megatron_bert import (
        MegatronBertConfig,
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
        MgpstrConfig,
        MgpstrForSceneTextRecognition,
        MgpstrModel,
        MgpstrPreTrainedModel,
        MgpstrProcessor,
        MgpstrTokenizer,
    )
    from .models.mimi import (
        MimiConfig,
        MimiModel,
        MimiPreTrainedModel,
    )
    from .models.mistral import (
        FlaxMistralForCausalLM,
        FlaxMistralModel,
        FlaxMistralPreTrainedModel,
        MistralConfig,
        MistralForCausalLM,
        MistralForQuestionAnswering,
        MistralForSequenceClassification,
        MistralForTokenClassification,
        MistralModel,
        MistralPreTrainedModel,
        TFMistralForCausalLM,
        TFMistralForSequenceClassification,
        TFMistralModel,
        TFMistralPreTrainedModel,
    )
    from .models.mistral3 import (
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        Mistral3PreTrainedModel,
    )
    from .models.mixtral import (
        MixtralConfig,
        MixtralForCausalLM,
        MixtralForQuestionAnswering,
        MixtralForSequenceClassification,
        MixtralForTokenClassification,
        MixtralModel,
        MixtralPreTrainedModel,
    )
    from .models.mllama import (
        MllamaConfig,
        MllamaForCausalLM,
        MllamaForConditionalGeneration,
        MllamaImageProcessor,
        MllamaPreTrainedModel,
        MllamaProcessor,
        MllamaTextModel,
        MllamaVisionModel,
    )
    from .models.mluke import MLukeTokenizer
    from .models.mobilebert import (
        MobileBertConfig,
        MobileBertForMaskedLM,
        MobileBertForMultipleChoice,
        MobileBertForNextSentencePrediction,
        MobileBertForPreTraining,
        MobileBertForQuestionAnswering,
        MobileBertForSequenceClassification,
        MobileBertForTokenClassification,
        MobileBertModel,
        MobileBertPreTrainedModel,
        MobileBertTokenizer,
        MobileBertTokenizerFast,
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
        load_tf_weights_in_mobilebert,
    )
    from .models.mobilenet_v1 import (
        MobileNetV1Config,
        MobileNetV1FeatureExtractor,
        MobileNetV1ForImageClassification,
        MobileNetV1ImageProcessor,
        MobileNetV1Model,
        MobileNetV1PreTrainedModel,
        load_tf_weights_in_mobilenet_v1,
    )
    from .models.mobilenet_v2 import (
        MobileNetV2Config,
        MobileNetV2FeatureExtractor,
        MobileNetV2ForImageClassification,
        MobileNetV2ForSemanticSegmentation,
        MobileNetV2ImageProcessor,
        MobileNetV2Model,
        MobileNetV2PreTrainedModel,
        load_tf_weights_in_mobilenet_v2,
    )
    from .models.mobilevit import (
        MobileViTConfig,
        MobileViTFeatureExtractor,
        MobileViTForImageClassification,
        MobileViTForSemanticSegmentation,
        MobileViTImageProcessor,
        MobileViTModel,
        MobileViTPreTrainedModel,
        TFMobileViTForImageClassification,
        TFMobileViTForSemanticSegmentation,
        TFMobileViTModel,
        TFMobileViTPreTrainedModel,
    )
    from .models.mobilevitv2 import (
        MobileViTV2Config,
        MobileViTV2ForImageClassification,
        MobileViTV2ForSemanticSegmentation,
        MobileViTV2Model,
        MobileViTV2PreTrainedModel,
    )
    from .models.modernbert import (
        ModernBertConfig,
        ModernBertForMaskedLM,
        ModernBertForQuestionAnswering,
        ModernBertForSequenceClassification,
        ModernBertForTokenClassification,
        ModernBertModel,
        ModernBertPreTrainedModel,
    )
    from .models.moonshine import (
        MoonshineConfig,
        MoonshineForConditionalGeneration,
        MoonshineModel,
        MoonshinePreTrainedModel,
    )
    from .models.moshi import (
        MoshiConfig,
        MoshiDepthConfig,
        MoshiForCausalLM,
        MoshiForConditionalGeneration,
        MoshiModel,
        MoshiPreTrainedModel,
    )
    from .models.mpnet import (
        MPNetConfig,
        MPNetForMaskedLM,
        MPNetForMultipleChoice,
        MPNetForQuestionAnswering,
        MPNetForSequenceClassification,
        MPNetForTokenClassification,
        MPNetModel,
        MPNetPreTrainedModel,
        MPNetTokenizer,
        MPNetTokenizerFast,
        TFMPNetForMaskedLM,
        TFMPNetForMultipleChoice,
        TFMPNetForQuestionAnswering,
        TFMPNetForSequenceClassification,
        TFMPNetForTokenClassification,
        TFMPNetMainLayer,
        TFMPNetModel,
        TFMPNetPreTrainedModel,
    )
    from .models.mpt import (
        MptConfig,
        MptForCausalLM,
        MptForQuestionAnswering,
        MptForSequenceClassification,
        MptForTokenClassification,
        MptModel,
        MptPreTrainedModel,
    )
    from .models.mra import (
        MraConfig,
        MraForMaskedLM,
        MraForMultipleChoice,
        MraForQuestionAnswering,
        MraForSequenceClassification,
        MraForTokenClassification,
        MraModel,
        MraPreTrainedModel,
    )
    from .models.mt5 import (
        FlaxMT5EncoderModel,
        FlaxMT5ForConditionalGeneration,
        FlaxMT5Model,
        MT5Config,
        MT5EncoderModel,
        MT5ForConditionalGeneration,
        MT5ForQuestionAnswering,
        MT5ForSequenceClassification,
        MT5ForTokenClassification,
        MT5Model,
        MT5PreTrainedModel,
        MT5Tokenizer,
        MT5TokenizerFast,
        TFMT5EncoderModel,
        TFMT5ForConditionalGeneration,
        TFMT5Model,
    )
    from .models.musicgen import (
        MusicgenConfig,
        MusicgenDecoderConfig,
        MusicgenForCausalLM,
        MusicgenForConditionalGeneration,
        MusicgenModel,
        MusicgenPreTrainedModel,
        MusicgenProcessor,
    )
    from .models.musicgen_melody import (
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
        MusicgenMelodyFeatureExtractor,
        MusicgenMelodyForCausalLM,
        MusicgenMelodyForConditionalGeneration,
        MusicgenMelodyModel,
        MusicgenMelodyPreTrainedModel,
        MusicgenMelodyProcessor,
    )
    from .models.mvp import (
        MvpConfig,
        MvpForCausalLM,
        MvpForConditionalGeneration,
        MvpForQuestionAnswering,
        MvpForSequenceClassification,
        MvpModel,
        MvpPreTrainedModel,
        MvpTokenizer,
        MvpTokenizerFast,
    )
    from .models.myt5 import MyT5Tokenizer
    from .models.nemotron import (
        NemotronConfig,
        NemotronForCausalLM,
        NemotronForQuestionAnswering,
        NemotronForSequenceClassification,
        NemotronForTokenClassification,
        NemotronModel,
        NemotronPreTrainedModel,
    )
    from .models.nllb import NllbTokenizer, NllbTokenizerFast
    from .models.nllb_moe import (
        NllbMoeConfig,
        NllbMoeForConditionalGeneration,
        NllbMoeModel,
        NllbMoePreTrainedModel,
        NllbMoeSparseMLP,
        NllbMoeTop2Router,
    )
    from .models.nougat import NougatImageProcessor, NougatProcessor, NougatTokenizerFast
    from .models.nystromformer import (
        NystromformerConfig,
        NystromformerForMaskedLM,
        NystromformerForMultipleChoice,
        NystromformerForQuestionAnswering,
        NystromformerForSequenceClassification,
        NystromformerForTokenClassification,
        NystromformerModel,
        NystromformerPreTrainedModel,
    )
    from .models.olmo import (
        OlmoConfig,
        OlmoForCausalLM,
        OlmoModel,
        OlmoPreTrainedModel,
    )
    from .models.olmo2 import (
        Olmo2Config,
        Olmo2ForCausalLM,
        Olmo2Model,
        Olmo2PreTrainedModel,
    )
    from .models.olmoe import (
        OlmoeConfig,
        OlmoeForCausalLM,
        OlmoeModel,
        OlmoePreTrainedModel,
    )
    from .models.omdet_turbo import (
        OmDetTurboConfig,
        OmDetTurboForObjectDetection,
        OmDetTurboPreTrainedModel,
        OmDetTurboProcessor,
    )
    from .models.oneformer import (
        OneFormerConfig,
        OneFormerForUniversalSegmentation,
        OneFormerImageProcessor,
        OneFormerModel,
        OneFormerPreTrainedModel,
        OneFormerProcessor,
    )
    from .models.openai import (
        OpenAIGPTConfig,
        OpenAIGPTDoubleHeadsModel,
        OpenAIGPTForSequenceClassification,
        OpenAIGPTLMHeadModel,
        OpenAIGPTModel,
        OpenAIGPTPreTrainedModel,
        OpenAIGPTTokenizer,
        OpenAIGPTTokenizerFast,
        TFOpenAIGPTDoubleHeadsModel,
        TFOpenAIGPTForSequenceClassification,
        TFOpenAIGPTLMHeadModel,
        TFOpenAIGPTMainLayer,
        TFOpenAIGPTModel,
        TFOpenAIGPTPreTrainedModel,
        load_tf_weights_in_openai_gpt,
    )
    from .models.opt import (
        FlaxOPTForCausalLM,
        FlaxOPTModel,
        FlaxOPTPreTrainedModel,
        OPTConfig,
        OPTForCausalLM,
        OPTForQuestionAnswering,
        OPTForSequenceClassification,
        OPTModel,
        OPTPreTrainedModel,
        TFOPTForCausalLM,
        TFOPTModel,
        TFOPTPreTrainedModel,
    )
    from .models.owlv2 import (
        Owlv2Config,
        Owlv2ForObjectDetection,
        Owlv2ImageProcessor,
        Owlv2Model,
        Owlv2PreTrainedModel,
        Owlv2Processor,
        Owlv2TextConfig,
        Owlv2TextModel,
        Owlv2VisionConfig,
        Owlv2VisionModel,
    )
    from .models.owlvit import (
        OwlViTConfig,
        OwlViTFeatureExtractor,
        OwlViTForObjectDetection,
        OwlViTImageProcessor,
        OwlViTModel,
        OwlViTPreTrainedModel,
        OwlViTProcessor,
        OwlViTTextConfig,
        OwlViTTextModel,
        OwlViTVisionConfig,
        OwlViTVisionModel,
    )
    from .models.paligemma import (
        PaliGemmaConfig,
        PaliGemmaForConditionalGeneration,
        PaliGemmaPreTrainedModel,
        PaliGemmaProcessor,
    )
    from .models.patchtsmixer import (
        PatchTSMixerConfig,
        PatchTSMixerForPrediction,
        PatchTSMixerForPretraining,
        PatchTSMixerForRegression,
        PatchTSMixerForTimeSeriesClassification,
        PatchTSMixerModel,
        PatchTSMixerPreTrainedModel,
    )
    from .models.patchtst import (
        PatchTSTConfig,
        PatchTSTForClassification,
        PatchTSTForPrediction,
        PatchTSTForPretraining,
        PatchTSTForRegression,
        PatchTSTModel,
        PatchTSTPreTrainedModel,
    )
    from .models.pegasus import (
        FlaxPegasusForConditionalGeneration,
        FlaxPegasusModel,
        FlaxPegasusPreTrainedModel,
        PegasusConfig,
        PegasusForCausalLM,
        PegasusForConditionalGeneration,
        PegasusModel,
        PegasusPreTrainedModel,
        PegasusTokenizer,
        PegasusTokenizerFast,
        TFPegasusForConditionalGeneration,
        TFPegasusModel,
        TFPegasusPreTrainedModel,
    )
    from .models.pegasus_x import (
        PegasusXConfig,
        PegasusXForConditionalGeneration,
        PegasusXModel,
        PegasusXPreTrainedModel,
    )
    from .models.perceiver import (
        PerceiverConfig,
        PerceiverFeatureExtractor,
        PerceiverForImageClassificationConvProcessing,
        PerceiverForImageClassificationFourier,
        PerceiverForImageClassificationLearned,
        PerceiverForMaskedLM,
        PerceiverForMultimodalAutoencoding,
        PerceiverForOpticalFlow,
        PerceiverForSequenceClassification,
        PerceiverImageProcessor,
        PerceiverModel,
        PerceiverPreTrainedModel,
        PerceiverTokenizer,
    )
    from .models.persimmon import (
        PersimmonConfig,
        PersimmonForCausalLM,
        PersimmonForSequenceClassification,
        PersimmonForTokenClassification,
        PersimmonModel,
        PersimmonPreTrainedModel,
    )
    from .models.phi import (
        PhiConfig,
        PhiForCausalLM,
        PhiForSequenceClassification,
        PhiForTokenClassification,
        PhiModel,
        PhiPreTrainedModel,
    )
    from .models.phi3 import (
        Phi3Config,
        Phi3ForCausalLM,
        Phi3ForSequenceClassification,
        Phi3ForTokenClassification,
        Phi3Model,
        Phi3PreTrainedModel,
    )
    from .models.phi4_multimodal import (
        Phi4MultimodalAudioConfig,
        Phi4MultimodalAudioModel,
        Phi4MultimodalAudioPreTrainedModel,
        Phi4MultimodalConfig,
        Phi4MultimodalFeatureExtractor,
        Phi4MultimodalForCausalLM,
        Phi4MultimodalImageProcessorFast,
        Phi4MultimodalModel,
        Phi4MultimodalPreTrainedModel,
        Phi4MultimodalProcessor,
        Phi4MultimodalVisionConfig,
        Phi4MultimodalVisionModel,
        Phi4MultimodalVisionPreTrainedModel,
    )
    from .models.phimoe import (
        PhimoeConfig,
        PhimoeForCausalLM,
        PhimoeForSequenceClassification,
        PhimoeModel,
        PhimoePreTrainedModel,
    )
    from .models.phobert import PhobertTokenizer
    from .models.pix2struct import (
        Pix2StructConfig,
        Pix2StructForConditionalGeneration,
        Pix2StructImageProcessor,
        Pix2StructPreTrainedModel,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructTextModel,
        Pix2StructVisionConfig,
        Pix2StructVisionModel,
    )
    from .models.pixtral import (
        PixtralImageProcessor,
        PixtralImageProcessorFast,
        PixtralPreTrainedModel,
        PixtralProcessor,
        PixtralVisionConfig,
        PixtralVisionModel,
    )
    from .models.plbart import (
        PLBartConfig,
        PLBartForCausalLM,
        PLBartForConditionalGeneration,
        PLBartForSequenceClassification,
        PLBartModel,
        PLBartPreTrainedModel,
        PLBartTokenizer,
    )
    from .models.poolformer import (
        PoolFormerConfig,
        PoolFormerFeatureExtractor,
        PoolFormerForImageClassification,
        PoolFormerImageProcessor,
        PoolFormerModel,
        PoolFormerPreTrainedModel,
    )
    from .models.pop2piano import (
        Pop2PianoConfig,
        Pop2PianoFeatureExtractor,
        Pop2PianoForConditionalGeneration,
        Pop2PianoPreTrainedModel,
        Pop2PianoProcessor,
        Pop2PianoTokenizer,
    )
    from .models.prompt_depth_anything import (
        PromptDepthAnythingConfig,
        PromptDepthAnythingForDepthEstimation,
        PromptDepthAnythingImageProcessor,
        PromptDepthAnythingPreTrainedModel,
    )
    from .models.prophetnet import (
        ProphetNetConfig,
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetPreTrainedModel,
        ProphetNetTokenizer,
    )
    from .models.pvt import (
        PvtConfig,
        PvtForImageClassification,
        PvtImageProcessor,
        PvtModel,
        PvtPreTrainedModel,
    )
    from .models.pvt_v2 import (
        PvtV2Backbone,
        PvtV2Config,
        PvtV2ForImageClassification,
        PvtV2Model,
        PvtV2PreTrainedModel,
    )
    from .models.qwen2 import (
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen2ForQuestionAnswering,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
        Qwen2Model,
        Qwen2PreTrainedModel,
        Qwen2Tokenizer,
        Qwen2TokenizerFast,
    )
    from .models.qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLPreTrainedModel,
        Qwen2_5_VLProcessor,
    )
    from .models.qwen2_audio import (
        Qwen2AudioConfig,
        Qwen2AudioEncoder,
        Qwen2AudioEncoderConfig,
        Qwen2AudioForConditionalGeneration,
        Qwen2AudioPreTrainedModel,
        Qwen2AudioProcessor,
    )
    from .models.qwen2_moe import (
        Qwen2MoeConfig,
        Qwen2MoeForCausalLM,
        Qwen2MoeForQuestionAnswering,
        Qwen2MoeForSequenceClassification,
        Qwen2MoeForTokenClassification,
        Qwen2MoeModel,
        Qwen2MoePreTrainedModel,
    )
    from .models.qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLForConditionalGeneration,
        Qwen2VLImageProcessor,
        Qwen2VLImageProcessorFast,
        Qwen2VLModel,
        Qwen2VLPreTrainedModel,
        Qwen2VLProcessor,
    )
    from .models.qwen3 import (
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3ForQuestionAnswering,
        Qwen3ForSequenceClassification,
        Qwen3ForTokenClassification,
        Qwen3Model,
        Qwen3PreTrainedModel,
    )
    from .models.qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
        Qwen3MoeForQuestionAnswering,
        Qwen3MoeForSequenceClassification,
        Qwen3MoeForTokenClassification,
        Qwen3MoeModel,
        Qwen3MoePreTrainedModel,
    )
    from .models.rag import (
        RagConfig,
        RagModel,
        RagPreTrainedModel,
        RagRetriever,
        RagSequenceForGeneration,
        RagTokenForGeneration,
        RagTokenizer,
        TFRagModel,
        TFRagPreTrainedModel,
        TFRagSequenceForGeneration,
        TFRagTokenForGeneration,
    )
    from .models.recurrent_gemma import (
        RecurrentGemmaConfig,
        RecurrentGemmaForCausalLM,
        RecurrentGemmaModel,
        RecurrentGemmaPreTrainedModel,
    )
    from .models.reformer import (
        ReformerConfig,
        ReformerForMaskedLM,
        ReformerForQuestionAnswering,
        ReformerForSequenceClassification,
        ReformerModel,
        ReformerModelWithLMHead,
        ReformerPreTrainedModel,
        ReformerTokenizer,
        ReformerTokenizerFast,
    )
    from .models.regnet import (
        FlaxRegNetForImageClassification,
        FlaxRegNetModel,
        FlaxRegNetPreTrainedModel,
        RegNetConfig,
        RegNetForImageClassification,
        RegNetModel,
        RegNetPreTrainedModel,
        TFRegNetForImageClassification,
        TFRegNetModel,
        TFRegNetPreTrainedModel,
    )
    from .models.rembert import (
        RemBertConfig,
        RemBertForCausalLM,
        RemBertForMaskedLM,
        RemBertForMultipleChoice,
        RemBertForQuestionAnswering,
        RemBertForSequenceClassification,
        RemBertForTokenClassification,
        RemBertModel,
        RemBertPreTrainedModel,
        RemBertTokenizer,
        RemBertTokenizerFast,
        TFRemBertForCausalLM,
        TFRemBertForMaskedLM,
        TFRemBertForMultipleChoice,
        TFRemBertForQuestionAnswering,
        TFRemBertForSequenceClassification,
        TFRemBertForTokenClassification,
        TFRemBertModel,
        TFRemBertPreTrainedModel,
        load_tf_weights_in_rembert,
    )
    from .models.resnet import (
        FlaxResNetForImageClassification,
        FlaxResNetModel,
        FlaxResNetPreTrainedModel,
        ResNetBackbone,
        ResNetConfig,
        ResNetForImageClassification,
        ResNetModel,
        ResNetPreTrainedModel,
        TFResNetForImageClassification,
        TFResNetModel,
        TFResNetPreTrainedModel,
    )
    from .models.roberta import (
        FlaxRobertaForCausalLM,
        FlaxRobertaForMaskedLM,
        FlaxRobertaForMultipleChoice,
        FlaxRobertaForQuestionAnswering,
        FlaxRobertaForSequenceClassification,
        FlaxRobertaForTokenClassification,
        FlaxRobertaModel,
        FlaxRobertaPreTrainedModel,
        RobertaConfig,
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaModel,
        RobertaPreTrainedModel,
        RobertaTokenizer,
        RobertaTokenizerFast,
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
        FlaxRobertaPreLayerNormForCausalLM,
        FlaxRobertaPreLayerNormForMaskedLM,
        FlaxRobertaPreLayerNormForMultipleChoice,
        FlaxRobertaPreLayerNormForQuestionAnswering,
        FlaxRobertaPreLayerNormForSequenceClassification,
        FlaxRobertaPreLayerNormForTokenClassification,
        FlaxRobertaPreLayerNormModel,
        FlaxRobertaPreLayerNormPreTrainedModel,
        RobertaPreLayerNormConfig,
        RobertaPreLayerNormForCausalLM,
        RobertaPreLayerNormForMaskedLM,
        RobertaPreLayerNormForMultipleChoice,
        RobertaPreLayerNormForQuestionAnswering,
        RobertaPreLayerNormForSequenceClassification,
        RobertaPreLayerNormForTokenClassification,
        RobertaPreLayerNormModel,
        RobertaPreLayerNormPreTrainedModel,
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
    from .models.roc_bert import (
        RoCBertConfig,
        RoCBertForCausalLM,
        RoCBertForMaskedLM,
        RoCBertForMultipleChoice,
        RoCBertForPreTraining,
        RoCBertForQuestionAnswering,
        RoCBertForSequenceClassification,
        RoCBertForTokenClassification,
        RoCBertModel,
        RoCBertPreTrainedModel,
        RoCBertTokenizer,
        load_tf_weights_in_roc_bert,
    )
    from .models.roformer import (
        FlaxRoFormerForMaskedLM,
        FlaxRoFormerForMultipleChoice,
        FlaxRoFormerForQuestionAnswering,
        FlaxRoFormerForSequenceClassification,
        FlaxRoFormerForTokenClassification,
        FlaxRoFormerModel,
        FlaxRoFormerPreTrainedModel,
        RoFormerConfig,
        RoFormerForCausalLM,
        RoFormerForMaskedLM,
        RoFormerForMultipleChoice,
        RoFormerForQuestionAnswering,
        RoFormerForSequenceClassification,
        RoFormerForTokenClassification,
        RoFormerModel,
        RoFormerPreTrainedModel,
        RoFormerTokenizer,
        RoFormerTokenizerFast,
        TFRoFormerForCausalLM,
        TFRoFormerForMaskedLM,
        TFRoFormerForMultipleChoice,
        TFRoFormerForQuestionAnswering,
        TFRoFormerForSequenceClassification,
        TFRoFormerForTokenClassification,
        TFRoFormerModel,
        TFRoFormerPreTrainedModel,
        load_tf_weights_in_roformer,
    )
    from .models.rt_detr import (
        RTDetrConfig,
        RTDetrForObjectDetection,
        RTDetrImageProcessor,
        RTDetrImageProcessorFast,
        RTDetrModel,
        RTDetrPreTrainedModel,
        RTDetrResNetBackbone,
        RTDetrResNetConfig,
        RTDetrResNetPreTrainedModel,
    )
    from .models.rt_detr_v2 import RTDetrV2Config, RTDetrV2ForObjectDetection, RTDetrV2Model, RTDetrV2PreTrainedModel
    from .models.rwkv import (
        RwkvConfig,
        RwkvForCausalLM,
        RwkvModel,
        RwkvPreTrainedModel,
    )
    from .models.sam import (
        SamConfig,
        SamImageProcessor,
        SamMaskDecoderConfig,
        SamModel,
        SamPreTrainedModel,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
        SamVisionModel,
        TFSamModel,
        TFSamPreTrainedModel,
        TFSamVisionModel,
    )
    from .models.seamless_m4t import (
        SeamlessM4TCodeHifiGan,
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TForSpeechToSpeech,
        SeamlessM4TForSpeechToText,
        SeamlessM4TForTextToSpeech,
        SeamlessM4TForTextToText,
        SeamlessM4THifiGan,
        SeamlessM4TModel,
        SeamlessM4TPreTrainedModel,
        SeamlessM4TProcessor,
        SeamlessM4TTextToUnitForConditionalGeneration,
        SeamlessM4TTextToUnitModel,
        SeamlessM4TTokenizer,
        SeamlessM4TTokenizerFast,
    )
    from .models.seamless_m4t_v2 import (
        SeamlessM4Tv2Config,
        SeamlessM4Tv2ForSpeechToSpeech,
        SeamlessM4Tv2ForSpeechToText,
        SeamlessM4Tv2ForTextToSpeech,
        SeamlessM4Tv2ForTextToText,
        SeamlessM4Tv2Model,
        SeamlessM4Tv2PreTrainedModel,
    )
    from .models.segformer import (
        SegformerConfig,
        SegformerDecodeHead,
        SegformerFeatureExtractor,
        SegformerForImageClassification,
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
        SegformerModel,
        SegformerPreTrainedModel,
        TFSegformerDecodeHead,
        TFSegformerForImageClassification,
        TFSegformerForSemanticSegmentation,
        TFSegformerModel,
        TFSegformerPreTrainedModel,
    )
    from .models.seggpt import (
        SegGptConfig,
        SegGptForImageSegmentation,
        SegGptImageProcessor,
        SegGptModel,
        SegGptPreTrainedModel,
    )
    from .models.sew import (
        SEWConfig,
        SEWForCTC,
        SEWForSequenceClassification,
        SEWModel,
        SEWPreTrainedModel,
    )
    from .models.sew_d import (
        SEWDConfig,
        SEWDForCTC,
        SEWDForSequenceClassification,
        SEWDModel,
        SEWDPreTrainedModel,
    )
    from .models.shieldgemma2 import (
        ShieldGemma2Config,
        ShieldGemma2ForImageClassification,
        ShieldGemma2Processor,
    )
    from .models.siglip import (
        SiglipConfig,
        SiglipForImageClassification,
        SiglipImageProcessor,
        SiglipImageProcessorFast,
        SiglipModel,
        SiglipPreTrainedModel,
        SiglipProcessor,
        SiglipTextConfig,
        SiglipTextModel,
        SiglipTokenizer,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from .models.siglip2 import (
        Siglip2Config,
        Siglip2ForImageClassification,
        Siglip2ImageProcessor,
        Siglip2ImageProcessorFast,
        Siglip2Model,
        Siglip2PreTrainedModel,
        Siglip2Processor,
        Siglip2TextConfig,
        Siglip2TextModel,
        Siglip2VisionConfig,
        Siglip2VisionModel,
    )
    from .models.smolvlm import (
        SmolVLMConfig,
        SmolVLMForConditionalGeneration,
        SmolVLMImageProcessor,
        SmolVLMModel,
        SmolVLMPreTrainedModel,
        SmolVLMProcessor,
        SmolVLMVisionConfig,
        SmolVLMVisionTransformer,
    )
    from .models.speech_encoder_decoder import (
        FlaxSpeechEncoderDecoderModel,
        SpeechEncoderDecoderConfig,
        SpeechEncoderDecoderModel,
    )
    from .models.speech_to_text import (
        Speech2TextConfig,
        Speech2TextFeatureExtractor,
        Speech2TextForConditionalGeneration,
        Speech2TextModel,
        Speech2TextPreTrainedModel,
        Speech2TextProcessor,
        Speech2TextTokenizer,
        TFSpeech2TextForConditionalGeneration,
        TFSpeech2TextModel,
        TFSpeech2TextPreTrainedModel,
    )
    from .models.speecht5 import (
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5ForSpeechToSpeech,
        SpeechT5ForSpeechToText,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5HifiGanConfig,
        SpeechT5Model,
        SpeechT5PreTrainedModel,
        SpeechT5Processor,
        SpeechT5Tokenizer,
    )
    from .models.splinter import (
        SplinterConfig,
        SplinterForPreTraining,
        SplinterForQuestionAnswering,
        SplinterModel,
        SplinterPreTrainedModel,
        SplinterTokenizer,
        SplinterTokenizerFast,
    )
    from .models.squeezebert import (
        SqueezeBertConfig,
        SqueezeBertForMaskedLM,
        SqueezeBertForMultipleChoice,
        SqueezeBertForQuestionAnswering,
        SqueezeBertForSequenceClassification,
        SqueezeBertForTokenClassification,
        SqueezeBertModel,
        SqueezeBertPreTrainedModel,
        SqueezeBertTokenizer,
        SqueezeBertTokenizerFast,
    )
    from .models.stablelm import (
        StableLmConfig,
        StableLmForCausalLM,
        StableLmForSequenceClassification,
        StableLmForTokenClassification,
        StableLmModel,
        StableLmPreTrainedModel,
    )
    from .models.starcoder2 import (
        Starcoder2Config,
        Starcoder2ForCausalLM,
        Starcoder2ForSequenceClassification,
        Starcoder2ForTokenClassification,
        Starcoder2Model,
        Starcoder2PreTrainedModel,
    )
    from .models.superglue import (
        SuperGlueConfig,
        SuperGlueForKeypointMatching,
        SuperGlueImageProcessor,
        SuperGluePreTrainedModel,
    )
    from .models.superpoint import (
        SuperPointConfig,
        SuperPointForKeypointDetection,
        SuperPointImageProcessor,
        SuperPointPreTrainedModel,
    )
    from .models.swiftformer import (
        SwiftFormerConfig,
        SwiftFormerForImageClassification,
        SwiftFormerModel,
        SwiftFormerPreTrainedModel,
        TFSwiftFormerForImageClassification,
        TFSwiftFormerModel,
        TFSwiftFormerPreTrainedModel,
    )
    from .models.swin import (
        SwinBackbone,
        SwinConfig,
        SwinForImageClassification,
        SwinForMaskedImageModeling,
        SwinModel,
        SwinPreTrainedModel,
        TFSwinForImageClassification,
        TFSwinForMaskedImageModeling,
        TFSwinModel,
        TFSwinPreTrainedModel,
    )
    from .models.swin2sr import (
        Swin2SRConfig,
        Swin2SRForImageSuperResolution,
        Swin2SRImageProcessor,
        Swin2SRModel,
        Swin2SRPreTrainedModel,
    )
    from .models.swinv2 import (
        Swinv2Backbone,
        Swinv2Config,
        Swinv2ForImageClassification,
        Swinv2ForMaskedImageModeling,
        Swinv2Model,
        Swinv2PreTrainedModel,
    )
    from .models.switch_transformers import (
        SwitchTransformersConfig,
        SwitchTransformersEncoderModel,
        SwitchTransformersForConditionalGeneration,
        SwitchTransformersModel,
        SwitchTransformersPreTrainedModel,
        SwitchTransformersSparseMLP,
        SwitchTransformersTop1Router,
    )
    from .models.t5 import (
        FlaxT5EncoderModel,
        FlaxT5ForConditionalGeneration,
        FlaxT5Model,
        FlaxT5PreTrainedModel,
        T5Config,
        T5EncoderModel,
        T5ForConditionalGeneration,
        T5ForQuestionAnswering,
        T5ForSequenceClassification,
        T5ForTokenClassification,
        T5Model,
        T5PreTrainedModel,
        T5Tokenizer,
        T5TokenizerFast,
        TFT5EncoderModel,
        TFT5ForConditionalGeneration,
        TFT5Model,
        TFT5PreTrainedModel,
        load_tf_weights_in_t5,
    )
    from .models.table_transformer import (
        TableTransformerConfig,
        TableTransformerForObjectDetection,
        TableTransformerModel,
        TableTransformerPreTrainedModel,
    )
    from .models.tapas import (
        TapasConfig,
        TapasForMaskedLM,
        TapasForQuestionAnswering,
        TapasForSequenceClassification,
        TapasModel,
        TapasPreTrainedModel,
        TapasTokenizer,
        TFTapasForMaskedLM,
        TFTapasForQuestionAnswering,
        TFTapasForSequenceClassification,
        TFTapasModel,
        TFTapasPreTrainedModel,
        load_tf_weights_in_tapas,
    )
    from .models.textnet import (
        TextNetBackbone,
        TextNetConfig,
        TextNetForImageClassification,
        TextNetImageProcessor,
        TextNetModel,
        TextNetPreTrainedModel,
    )
    from .models.time_series_transformer import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerForPrediction,
        TimeSeriesTransformerModel,
        TimeSeriesTransformerPreTrainedModel,
    )
    from .models.timesformer import (
        TimesformerConfig,
        TimesformerForVideoClassification,
        TimesformerModel,
        TimesformerPreTrainedModel,
    )
    from .models.timm_backbone import TimmBackbone, TimmBackboneConfig
    from .models.timm_wrapper import (
        TimmWrapperConfig,
        TimmWrapperForImageClassification,
        TimmWrapperImageProcessor,
        TimmWrapperModel,
        TimmWrapperPreTrainedModel,
    )
    from .models.trocr import (
        TrOCRConfig,
        TrOCRForCausalLM,
        TrOCRPreTrainedModel,
        TrOCRProcessor,
    )
    from .models.tvp import (
        TvpConfig,
        TvpForVideoGrounding,
        TvpImageProcessor,
        TvpModel,
        TvpPreTrainedModel,
        TvpProcessor,
    )
    from .models.udop import (
        UdopConfig,
        UdopEncoderModel,
        UdopForConditionalGeneration,
        UdopModel,
        UdopPreTrainedModel,
        UdopProcessor,
        UdopTokenizer,
        UdopTokenizerFast,
    )
    from .models.umt5 import (
        UMT5Config,
        UMT5EncoderModel,
        UMT5ForConditionalGeneration,
        UMT5ForQuestionAnswering,
        UMT5ForSequenceClassification,
        UMT5ForTokenClassification,
        UMT5Model,
        UMT5PreTrainedModel,
    )
    from .models.unispeech import (
        UniSpeechConfig,
        UniSpeechForCTC,
        UniSpeechForPreTraining,
        UniSpeechForSequenceClassification,
        UniSpeechModel,
        UniSpeechPreTrainedModel,
    )
    from .models.unispeech_sat import (
        UniSpeechSatConfig,
        UniSpeechSatForAudioFrameClassification,
        UniSpeechSatForCTC,
        UniSpeechSatForPreTraining,
        UniSpeechSatForSequenceClassification,
        UniSpeechSatForXVector,
        UniSpeechSatModel,
        UniSpeechSatPreTrainedModel,
    )
    from .models.univnet import (
        UnivNetConfig,
        UnivNetFeatureExtractor,
        UnivNetModel,
    )
    from .models.upernet import (
        UperNetConfig,
        UperNetForSemanticSegmentation,
        UperNetPreTrainedModel,
    )
    from .models.video_llava import (
        VideoLlavaConfig,
        VideoLlavaForConditionalGeneration,
        VideoLlavaImageProcessor,
        VideoLlavaPreTrainedModel,
        VideoLlavaProcessor,
    )
    from .models.videomae import (
        VideoMAEConfig,
        VideoMAEFeatureExtractor,
        VideoMAEForPreTraining,
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
        VideoMAEModel,
        VideoMAEPreTrainedModel,
    )
    from .models.vilt import (
        ViltConfig,
        ViltFeatureExtractor,
        ViltForImageAndTextRetrieval,
        ViltForImagesAndTextClassification,
        ViltForMaskedLM,
        ViltForQuestionAnswering,
        ViltForTokenClassification,
        ViltImageProcessor,
        ViltModel,
        ViltPreTrainedModel,
        ViltProcessor,
    )
    from .models.vipllava import (
        VipLlavaConfig,
        VipLlavaForConditionalGeneration,
        VipLlavaPreTrainedModel,
    )
    from .models.vision_encoder_decoder import (
        FlaxVisionEncoderDecoderModel,
        TFVisionEncoderDecoderModel,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
    )
    from .models.vision_text_dual_encoder import (
        FlaxVisionTextDualEncoderModel,
        TFVisionTextDualEncoderModel,
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderModel,
        VisionTextDualEncoderProcessor,
    )
    from .models.visual_bert import (
        VisualBertConfig,
        VisualBertForMultipleChoice,
        VisualBertForPreTraining,
        VisualBertForQuestionAnswering,
        VisualBertForRegionToPhraseAlignment,
        VisualBertForVisualReasoning,
        VisualBertModel,
        VisualBertPreTrainedModel,
    )
    from .models.vit import (
        FlaxViTForImageClassification,
        FlaxViTModel,
        FlaxViTPreTrainedModel,
        TFViTForImageClassification,
        TFViTModel,
        TFViTPreTrainedModel,
        ViTConfig,
        ViTFeatureExtractor,
        ViTForImageClassification,
        ViTForMaskedImageModeling,
        ViTImageProcessor,
        ViTImageProcessorFast,
        ViTModel,
        ViTPreTrainedModel,
    )
    from .models.vit_mae import (
        TFViTMAEForPreTraining,
        TFViTMAEModel,
        TFViTMAEPreTrainedModel,
        ViTMAEConfig,
        ViTMAEForPreTraining,
        ViTMAEModel,
        ViTMAEPreTrainedModel,
    )
    from .models.vit_msn import (
        ViTMSNConfig,
        ViTMSNForImageClassification,
        ViTMSNModel,
        ViTMSNPreTrainedModel,
    )
    from .models.vitdet import (
        VitDetBackbone,
        VitDetConfig,
        VitDetModel,
        VitDetPreTrainedModel,
    )
    from .models.vitmatte import (
        VitMatteConfig,
        VitMatteForImageMatting,
        VitMatteImageProcessor,
        VitMattePreTrainedModel,
    )
    from .models.vitpose import (
        VitPoseConfig,
        VitPoseForPoseEstimation,
        VitPoseImageProcessor,
        VitPosePreTrainedModel,
    )
    from .models.vitpose_backbone import VitPoseBackbone, VitPoseBackboneConfig, VitPoseBackbonePreTrainedModel
    from .models.vits import (
        VitsConfig,
        VitsModel,
        VitsPreTrainedModel,
        VitsTokenizer,
    )
    from .models.vivit import (
        VivitConfig,
        VivitForVideoClassification,
        VivitImageProcessor,
        VivitModel,
        VivitPreTrainedModel,
    )
    from .models.wav2vec2 import (
        FlaxWav2Vec2ForCTC,
        FlaxWav2Vec2ForPreTraining,
        FlaxWav2Vec2Model,
        FlaxWav2Vec2PreTrainedModel,
        TFWav2Vec2ForCTC,
        TFWav2Vec2ForSequenceClassification,
        TFWav2Vec2Model,
        TFWav2Vec2PreTrainedModel,
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForAudioFrameClassification,
        Wav2Vec2ForCTC,
        Wav2Vec2ForMaskedLM,
        Wav2Vec2ForPreTraining,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2ForXVector,
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    from .models.wav2vec2_bert import (
        Wav2Vec2BertConfig,
        Wav2Vec2BertForAudioFrameClassification,
        Wav2Vec2BertForCTC,
        Wav2Vec2BertForSequenceClassification,
        Wav2Vec2BertForXVector,
        Wav2Vec2BertModel,
        Wav2Vec2BertPreTrainedModel,
        Wav2Vec2BertProcessor,
    )
    from .models.wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
        Wav2Vec2ConformerForAudioFrameClassification,
        Wav2Vec2ConformerForCTC,
        Wav2Vec2ConformerForPreTraining,
        Wav2Vec2ConformerForSequenceClassification,
        Wav2Vec2ConformerForXVector,
        Wav2Vec2ConformerModel,
        Wav2Vec2ConformerPreTrainedModel,
    )
    from .models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    from .models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    from .models.wavlm import (
        WavLMConfig,
        WavLMForAudioFrameClassification,
        WavLMForCTC,
        WavLMForSequenceClassification,
        WavLMForXVector,
        WavLMModel,
        WavLMPreTrainedModel,
    )
    from .models.whisper import (
        FlaxWhisperForAudioClassification,
        FlaxWhisperForConditionalGeneration,
        FlaxWhisperModel,
        FlaxWhisperPreTrainedModel,
        TFWhisperForConditionalGeneration,
        TFWhisperModel,
        TFWhisperPreTrainedModel,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForAudioClassification,
        WhisperForCausalLM,
        WhisperForConditionalGeneration,
        WhisperModel,
        WhisperPreTrainedModel,
        WhisperProcessor,
        WhisperTokenizer,
        WhisperTokenizerFast,
    )
    from .models.x_clip import (
        XCLIPConfig,
        XCLIPModel,
        XCLIPPreTrainedModel,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPTextModel,
        XCLIPVisionConfig,
        XCLIPVisionModel,
    )
    from .models.xglm import (
        FlaxXGLMForCausalLM,
        FlaxXGLMModel,
        FlaxXGLMPreTrainedModel,
        TFXGLMForCausalLM,
        TFXGLMModel,
        TFXGLMPreTrainedModel,
        XGLMConfig,
        XGLMForCausalLM,
        XGLMModel,
        XGLMPreTrainedModel,
        XGLMTokenizer,
        XGLMTokenizerFast,
    )
    from .models.xlm import (
        TFXLMForMultipleChoice,
        TFXLMForQuestionAnsweringSimple,
        TFXLMForSequenceClassification,
        TFXLMForTokenClassification,
        TFXLMMainLayer,
        TFXLMModel,
        TFXLMPreTrainedModel,
        TFXLMWithLMHeadModel,
        XLMConfig,
        XLMForMultipleChoice,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMModel,
        XLMPreTrainedModel,
        XLMTokenizer,
        XLMWithLMHeadModel,
    )
    from .models.xlm_roberta import (
        FlaxXLMRobertaForCausalLM,
        FlaxXLMRobertaForMaskedLM,
        FlaxXLMRobertaForMultipleChoice,
        FlaxXLMRobertaForQuestionAnswering,
        FlaxXLMRobertaForSequenceClassification,
        FlaxXLMRobertaForTokenClassification,
        FlaxXLMRobertaModel,
        FlaxXLMRobertaPreTrainedModel,
        TFXLMRobertaForCausalLM,
        TFXLMRobertaForMaskedLM,
        TFXLMRobertaForMultipleChoice,
        TFXLMRobertaForQuestionAnswering,
        TFXLMRobertaForSequenceClassification,
        TFXLMRobertaForTokenClassification,
        TFXLMRobertaModel,
        TFXLMRobertaPreTrainedModel,
        XLMRobertaConfig,
        XLMRobertaForCausalLM,
        XLMRobertaForMaskedLM,
        XLMRobertaForMultipleChoice,
        XLMRobertaForQuestionAnswering,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaModel,
        XLMRobertaPreTrainedModel,
        XLMRobertaTokenizer,
        XLMRobertaTokenizerFast,
    )
    from .models.xlm_roberta_xl import (
        XLMRobertaXLConfig,
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
        TFXLNetForMultipleChoice,
        TFXLNetForQuestionAnsweringSimple,
        TFXLNetForSequenceClassification,
        TFXLNetForTokenClassification,
        TFXLNetLMHeadModel,
        TFXLNetMainLayer,
        TFXLNetModel,
        TFXLNetPreTrainedModel,
        XLNetConfig,
        XLNetForMultipleChoice,
        XLNetForQuestionAnswering,
        XLNetForQuestionAnsweringSimple,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetLMHeadModel,
        XLNetModel,
        XLNetPreTrainedModel,
        XLNetTokenizer,
        XLNetTokenizerFast,
        load_tf_weights_in_xlnet,
    )
    from .models.xmod import (
        XmodConfig,
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
        YolosConfig,
        YolosFeatureExtractor,
        YolosForObjectDetection,
        YolosImageProcessor,
        YolosModel,
        YolosPreTrainedModel,
    )
    from .models.yoso import (
        YosoConfig,
        YosoForMaskedLM,
        YosoForMultipleChoice,
        YosoForQuestionAnswering,
        YosoForSequenceClassification,
        YosoForTokenClassification,
        YosoModel,
        YosoPreTrainedModel,
    )
    from .models.zamba import (
        ZambaConfig,
        ZambaForCausalLM,
        ZambaForSequenceClassification,
        ZambaModel,
        ZambaPreTrainedModel,
    )
    from .models.zamba2 import (
        Zamba2Config,
        Zamba2ForCausalLM,
        Zamba2ForSequenceClassification,
        Zamba2Model,
        Zamba2PreTrainedModel,
    )
    from .models.zoedepth import (
        ZoeDepthConfig,
        ZoeDepthForDepthEstimation,
        ZoeDepthImageProcessor,
        ZoeDepthPreTrainedModel,
    )

    # Optimization
    from .optimization import (
        Adafactor,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_inverse_sqrt_schedule,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
        get_wsd_schedule,
    )

    # Optimization
    from .optimization_tf import (
        AdamWeightDecay,
        GradientAccumulator,
        WarmUp,
        create_optimizer,
    )

    # Pipelines
    from .pipelines import (
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageFeatureExtractionPipeline,
        ImageSegmentationPipeline,
        ImageTextToTextPipeline,
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
    from .pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

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
    from .tokenization_utils_fast import PreTrainedTokenizerFast

    # Trainer
    from .trainer import Trainer

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
    from .trainer_pt_utils import torch_distributed_zero_first
    from .trainer_seq2seq import Seq2SeqTrainer
    from .trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
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
        is_av_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_sacremoses_available,
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
        is_torch_hpu_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from .utils.quantization_config import (
        AqlmConfig,
        AwqConfig,
        BitNetConfig,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
        EetqConfig,
        FbgemmFp8Config,
        FineGrainedFP8Config,
        GPTQConfig,
        HiggsConfig,
        HqqConfig,
        QuantoConfig,
        QuarkConfig,
        SpQRConfig,
        TorchAoConfig,
        VptqConfig,
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
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
