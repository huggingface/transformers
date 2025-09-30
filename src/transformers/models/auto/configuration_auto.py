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
"""Auto Config class."""

import importlib
import os
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator, KeysView, ValuesView
from typing import Any, TypeVar, Union

from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging


logger = logging.get_logger(__name__)


_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


CONFIG_MAPPING_NAMES = OrderedDict[str, str](
    [
        # Add configs here
        ("aimv2", "Aimv2Config"),
        ("aimv2_vision_model", "Aimv2VisionConfig"),
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("apertus", "ApertusConfig"),
        ("arcee", "ArceeConfig"),
        ("aria", "AriaConfig"),
        ("aria_text", "AriaTextConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("autoformer", "AutoformerConfig"),
        ("aya_vision", "AyaVisionConfig"),
        ("bamba", "BambaConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("bitnet", "BitNetConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("blip_2_qformer", "Blip2QFormerConfig"),
        ("bloom", "BloomConfig"),
        ("blt", "BltConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("clap", "ClapConfig"),
        ("clip", "CLIPConfig"),
        ("clip_text_model", "CLIPTextConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clvp", "ClvpConfig"),
        ("code_llama", "LlamaConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("cohere2", "Cohere2Config"),
        ("cohere2_vision", "Cohere2VisionConfig"),
        ("colpali", "ColPaliConfig"),
        ("colqwen2", "ColQwen2Config"),
        ("conditional_detr", "ConditionalDetrConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("cpmant", "CpmAntConfig"),
        ("csm", "CsmConfig"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("d_fine", "DFineConfig"),
        ("dab-detr", "DabDetrConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("dbrx", "DbrxConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("deepseek_v2", "DeepseekV2Config"),
        ("deepseek_v3", "DeepseekV3Config"),
        ("deepseek_vl", "DeepseekVLConfig"),
        ("deepseek_vl_hybrid", "DeepseekVLHybridConfig"),
        ("deformable_detr", "DeformableDetrConfig"),
        ("deit", "DeiTConfig"),
        ("depth_anything", "DepthAnythingConfig"),
        ("depth_pro", "DepthProConfig"),
        ("deta", "DetaConfig"),
        ("detr", "DetrConfig"),
        ("dia", "DiaConfig"),
        ("diffllama", "DiffLlamaConfig"),
        ("dinat", "DinatConfig"),
        ("dinov2", "Dinov2Config"),
        ("dinov2_with_registers", "Dinov2WithRegistersConfig"),
        ("dinov3_convnext", "DINOv3ConvNextConfig"),
        ("dinov3_vit", "DINOv3ViTConfig"),
        ("distilbert", "DistilBertConfig"),
        ("doge", "DogeConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("dots1", "Dots1Config"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("edgetam", "EdgeTamConfig"),
        ("edgetam_video", "EdgeTamVideoConfig"),
        ("edgetam_vision_model", "EdgeTamVisionConfig"),
        ("efficientformer", "EfficientFormerConfig"),
        ("efficientloftr", "EfficientLoFTRConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("emu3", "Emu3Config"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("eomt", "EomtConfig"),
        ("ernie", "ErnieConfig"),
        ("ernie4_5", "Ernie4_5Config"),
        ("ernie4_5_moe", "Ernie4_5_MoeConfig"),
        ("ernie_m", "ErnieMConfig"),
        ("esm", "EsmConfig"),
        ("evolla", "EvollaConfig"),
        ("exaone4", "Exaone4Config"),
        ("falcon", "FalconConfig"),
        ("falcon_h1", "FalconH1Config"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("fastspeech2_conformer_with_hifigan", "FastSpeech2ConformerWithHifiGanConfig"),
        ("flaubert", "FlaubertConfig"),
        ("flava", "FlavaConfig"),
        ("flex_olmo", "FlexOlmoConfig"),
        ("florence2", "Florence2Config"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("funnel", "FunnelConfig"),
        ("fuyu", "FuyuConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("gemma3", "Gemma3Config"),
        ("gemma3_text", "Gemma3TextConfig"),
        ("gemma3n", "Gemma3nConfig"),
        ("gemma3n_audio", "Gemma3nAudioConfig"),
        ("gemma3n_text", "Gemma3nTextConfig"),
        ("gemma3n_vision", "Gemma3nVisionConfig"),
        ("git", "GitConfig"),
        ("glm", "GlmConfig"),
        ("glm4", "Glm4Config"),
        ("glm4_moe", "Glm4MoeConfig"),
        ("glm4v", "Glm4vConfig"),
        ("glm4v_moe", "Glm4vMoeConfig"),
        ("glm4v_moe_text", "Glm4vMoeTextConfig"),
        ("glm4v_text", "Glm4vTextConfig"),
        ("glpn", "GLPNConfig"),
        ("got_ocr2", "GotOcr2Config"),
        ("gpt-sw3", "GPT2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gpt_oss", "GptOssConfig"),
        ("gptj", "GPTJConfig"),
        ("gptsan-japanese", "GPTSanJapaneseConfig"),
        ("granite", "GraniteConfig"),
        ("granite_speech", "GraniteSpeechConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("granitemoehybrid", "GraniteMoeHybridConfig"),
        ("granitemoeshared", "GraniteMoeSharedConfig"),
        ("granitevision", "LlavaNextConfig"),
        ("graphormer", "GraphormerConfig"),
        ("grounding-dino", "GroundingDinoConfig"),
        ("groupvit", "GroupViTConfig"),
        ("helium", "HeliumConfig"),
        ("hgnet_v2", "HGNetV2Config"),
        ("hiera", "HieraConfig"),
        ("hubert", "HubertConfig"),
        ("hunyuan_v1_dense", "HunYuanDenseV1Config"),
        ("hunyuan_v1_moe", "HunYuanMoEV1Config"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics3", "Idefics3Config"),
        ("idefics3_vision", "Idefics3VisionConfig"),
        ("ijepa", "IJepaConfig"),
        ("imagegpt", "ImageGPTConfig"),
        ("informer", "InformerConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("internvl", "InternVLConfig"),
        ("internvl_vision", "InternVLVisionConfig"),
        ("jamba", "JambaConfig"),
        ("janus", "JanusConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("jukebox", "JukeboxConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("kosmos-2.5", "Kosmos2_5Config"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextConfig"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lfm2", "Lfm2Config"),
        ("lfm2_vl", "Lfm2VlConfig"),
        ("lightglue", "LightGlueConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("llama4", "Llama4Config"),
        ("llama4_text", "Llama4TextConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longcat_flash", "LongcatFlashConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("lxmert", "LxmertConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("marian", "MarianConfig"),
        ("markuplm", "MarkupLMConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("mctct", "MCTCTConfig"),
        ("mega", "MegaConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("metaclip_2", "MetaClip2Config"),
        ("mgp-str", "MgpstrConfig"),
        ("mimi", "MimiConfig"),
        ("minimax", "MiniMaxConfig"),
        ("ministral", "MinistralConfig"),
        ("mistral", "MistralConfig"),
        ("mistral3", "Mistral3Config"),
        ("mixtral", "MixtralConfig"),
        ("mlcd", "MLCDVisionConfig"),
        ("mllama", "MllamaConfig"),
        ("mm-grounding-dino", "MMGroundingDinoConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("modernbert", "ModernBertConfig"),
        ("modernbert-decoder", "ModernBertDecoderConfig"),
        ("moonshine", "MoonshineConfig"),
        ("moshi", "MoshiConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("mvp", "MvpConfig"),
        ("nat", "NatConfig"),
        ("nemotron", "NemotronConfig"),
        ("nezha", "NezhaConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nougat", "VisionEncoderDecoderConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("olmo2", "Olmo2Config"),
        ("olmo3", "Olmo3Config"),
        ("olmoe", "OlmoeConfig"),
        ("omdet-turbo", "OmDetTurboConfig"),
        ("oneformer", "OneFormerConfig"),
        ("open-llama", "OpenLlamaConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("opt", "OPTConfig"),
        ("ovis2", "Ovis2Config"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("parakeet_ctc", "ParakeetCTCConfig"),
        ("parakeet_encoder", "ParakeetEncoderConfig"),
        ("patchtsmixer", "PatchTSMixerConfig"),
        ("patchtst", "PatchTSTConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("perception_encoder", "TimmWrapperConfig"),
        ("perception_lm", "PerceptionLMConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phi4_multimodal", "Phi4MultimodalConfig"),
        ("phimoe", "PhimoeConfig"),
        ("pix2struct", "Pix2StructConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("prompt_depth_anything", "PromptDepthAnythingConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qdqbert", "QDQBertConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_5_omni", "Qwen2_5OmniConfig"),
        ("qwen2_5_vl", "Qwen2_5_VLConfig"),
        ("qwen2_5_vl_text", "Qwen2_5_VLTextConfig"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("qwen2_vl_text", "Qwen2VLTextConfig"),
        ("qwen3", "Qwen3Config"),
        ("qwen3_moe", "Qwen3MoeConfig"),
        ("qwen3_next", "Qwen3NextConfig"),
        ("qwen3_omni_moe", "Qwen3OmniMoeConfig"),
        ("qwen3_vl", "Qwen3VLConfig"),
        ("qwen3_vl_moe", "Qwen3VLMoeConfig"),
        ("qwen3_vl_moe_text", "Qwen3VLMoeTextConfig"),
        ("qwen3_vl_text", "Qwen3VLTextConfig"),
        ("rag", "RagConfig"),
        ("realm", "RealmConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("reformer", "ReformerConfig"),
        ("regnet", "RegNetConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("retribert", "RetriBertConfig"),
        ("roberta", "RobertaConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rt_detr_v2", "RTDetrV2Config"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("sam2", "Sam2Config"),
        ("sam2_hiera_det_model", "Sam2HieraDetConfig"),
        ("sam2_video", "Sam2VideoConfig"),
        ("sam2_vision_model", "Sam2VisionConfig"),
        ("sam_hq", "SamHQConfig"),
        ("sam_hq_vision_model", "SamHQVisionConfig"),
        ("sam_vision_model", "SamVisionConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("seed_oss", "SeedOssConfig"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("shieldgemma2", "ShieldGemma2Config"),
        ("siglip", "SiglipConfig"),
        ("siglip2", "Siglip2Config"),
        ("siglip2_vision_model", "Siglip2VisionConfig"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("smollm3", "SmolLM3Config"),
        ("smolvlm", "SmolVLMConfig"),
        ("smolvlm_vision", "SmolVLMVisionConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speech_to_text_2", "Speech2Text2Config"),
        ("speecht5", "SpeechT5Config"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("superglue", "SuperGlueConfig"),
        ("superpoint", "SuperPointConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("t5", "T5Config"),
        ("t5gemma", "T5GemmaConfig"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("textnet", "TextNetConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesfm", "TimesFmConfig"),
        ("timesformer", "TimesformerConfig"),
        ("timm_backbone", "TimmBackboneConfig"),
        ("timm_wrapper", "TimmWrapperConfig"),
        ("trajectory_transformer", "TrajectoryTransformerConfig"),
        ("transfo-xl", "TransfoXLConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvlt", "TvltConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("van", "VanConfig"),
        ("vaultgemma", "VaultGemmaConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_hybrid", "ViTHybridConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitmatte", "VitMatteConfig"),
        ("vitpose", "VitPoseConfig"),
        ("vitpose_backbone", "VitPoseBackboneConfig"),
        ("vits", "VitsConfig"),
        ("vivit", "VivitConfig"),
        ("vjepa2", "VJEPA2Config"),
        ("voxtral", "VoxtralConfig"),
        ("voxtral_encoder", "VoxtralEncoderConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xcodec", "XcodecConfig"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xlstm", "xLSTMConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("zamba", "ZambaConfig"),
        ("zamba2", "Zamba2Config"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)


MODEL_NAMES_MAPPING = OrderedDict[str, str](
    [
        # Add full (and cased) model names here
        ("aimv2", "AIMv2"),
        ("aimv2_vision_model", "Aimv2VisionModel"),
        ("albert", "ALBERT"),
        ("align", "ALIGN"),
        ("altclip", "AltCLIP"),
        ("apertus", "Apertus"),
        ("arcee", "Arcee"),
        ("aria", "Aria"),
        ("aria_text", "AriaText"),
        ("audio-spectrogram-transformer", "Audio Spectrogram Transformer"),
        ("autoformer", "Autoformer"),
        ("aya_vision", "AyaVision"),
        ("bamba", "Bamba"),
        ("bark", "Bark"),
        ("bart", "BART"),
        ("barthez", "BARThez"),
        ("bartpho", "BARTpho"),
        ("beit", "BEiT"),
        ("bert", "BERT"),
        ("bert-generation", "Bert Generation"),
        ("bert-japanese", "BertJapanese"),
        ("bertweet", "BERTweet"),
        ("big_bird", "BigBird"),
        ("bigbird_pegasus", "BigBird-Pegasus"),
        ("biogpt", "BioGpt"),
        ("bit", "BiT"),
        ("bitnet", "BitNet"),
        ("blenderbot", "Blenderbot"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("blip", "BLIP"),
        ("blip-2", "BLIP-2"),
        ("blip_2_qformer", "BLIP-2 QFormer"),
        ("bloom", "BLOOM"),
        ("blt", "Blt"),
        ("bort", "BORT"),
        ("bridgetower", "BridgeTower"),
        ("bros", "BROS"),
        ("byt5", "ByT5"),
        ("camembert", "CamemBERT"),
        ("canine", "CANINE"),
        ("chameleon", "Chameleon"),
        ("chinese_clip", "Chinese-CLIP"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "CLAP"),
        ("clip", "CLIP"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSeg"),
        ("clvp", "CLVP"),
        ("code_llama", "CodeLlama"),
        ("codegen", "CodeGen"),
        ("cohere", "Cohere"),
        ("cohere2", "Cohere2"),
        ("cohere2_vision", "Cohere2Vision"),
        ("colpali", "ColPali"),
        ("colqwen2", "ColQwen2"),
        ("conditional_detr", "Conditional DETR"),
        ("convbert", "ConvBERT"),
        ("convnext", "ConvNeXT"),
        ("convnextv2", "ConvNeXTV2"),
        ("cpm", "CPM"),
        ("cpmant", "CPM-Ant"),
        ("csm", "CSM"),
        ("ctrl", "CTRL"),
        ("cvt", "CvT"),
        ("d_fine", "D-FINE"),
        ("dab-detr", "DAB-DETR"),
        ("dac", "DAC"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("dbrx", "DBRX"),
        ("deberta", "DeBERTa"),
        ("deberta-v2", "DeBERTa-v2"),
        ("decision_transformer", "Decision Transformer"),
        ("deepseek_v2", "DeepSeek-V2"),
        ("deepseek_v3", "DeepSeek-V3"),
        ("deepseek_vl", "DeepseekVL"),
        ("deepseek_vl_hybrid", "DeepseekVLHybrid"),
        ("deformable_detr", "Deformable DETR"),
        ("deit", "DeiT"),
        ("deplot", "DePlot"),
        ("depth_anything", "Depth Anything"),
        ("depth_anything_v2", "Depth Anything V2"),
        ("depth_pro", "DepthPro"),
        ("deta", "DETA"),
        ("detr", "DETR"),
        ("dia", "Dia"),
        ("dialogpt", "DialoGPT"),
        ("diffllama", "DiffLlama"),
        ("dinat", "DiNAT"),
        ("dinov2", "DINOv2"),
        ("dinov2_with_registers", "DINOv2 with Registers"),
        ("dinov3_convnext", "DINOv3 ConvNext"),
        ("dinov3_vit", "DINOv3 ViT"),
        ("distilbert", "DistilBERT"),
        ("dit", "DiT"),
        ("doge", "Doge"),
        ("donut-swin", "DonutSwin"),
        ("dots1", "dots1"),
        ("dpr", "DPR"),
        ("dpt", "DPT"),
        ("edgetam", "EdgeTAM"),
        ("edgetam_video", "EdgeTamVideo"),
        ("edgetam_vision_model", "EdgeTamVisionModel"),
        ("efficientformer", "EfficientFormer"),
        ("efficientloftr", "EfficientLoFTR"),
        ("efficientnet", "EfficientNet"),
        ("electra", "ELECTRA"),
        ("emu3", "Emu3"),
        ("encodec", "EnCodec"),
        ("encoder-decoder", "Encoder decoder"),
        ("eomt", "EoMT"),
        ("ernie", "ERNIE"),
        ("ernie4_5", "Ernie4_5"),
        ("ernie4_5_moe", "Ernie4_5_MoE"),
        ("ernie_m", "ErnieM"),
        ("esm", "ESM"),
        ("evolla", "Evolla"),
        ("exaone4", "EXAONE-4.0"),
        ("falcon", "Falcon"),
        ("falcon3", "Falcon3"),
        ("falcon_h1", "FalconH1"),
        ("falcon_mamba", "FalconMamba"),
        ("fastspeech2_conformer", "FastSpeech2Conformer"),
        ("fastspeech2_conformer_with_hifigan", "FastSpeech2ConformerWithHifiGan"),
        ("flan-t5", "FLAN-T5"),
        ("flan-ul2", "FLAN-UL2"),
        ("flaubert", "FlauBERT"),
        ("flava", "FLAVA"),
        ("flex_olmo", "FlexOlmo"),
        ("florence2", "Florence2"),
        ("fnet", "FNet"),
        ("focalnet", "FocalNet"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("funnel", "Funnel Transformer"),
        ("fuyu", "Fuyu"),
        ("gemma", "Gemma"),
        ("gemma2", "Gemma2"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("gemma3n", "Gemma3nForConditionalGeneration"),
        ("gemma3n_audio", "Gemma3nAudioEncoder"),
        ("gemma3n_text", "Gemma3nForCausalLM"),
        ("gemma3n_vision", "TimmWrapperModel"),
        ("git", "GIT"),
        ("glm", "GLM"),
        ("glm4", "GLM4"),
        ("glm4_moe", "Glm4MoE"),
        ("glm4v", "GLM4V"),
        ("glm4v_moe", "GLM4VMOE"),
        ("glm4v_moe_text", "GLM4VMOE"),
        ("glm4v_text", "GLM4V"),
        ("glpn", "GLPN"),
        ("got_ocr2", "GOT-OCR2"),
        ("gpt-sw3", "GPT-Sw3"),
        ("gpt2", "OpenAI GPT-2"),
        ("gpt_bigcode", "GPTBigCode"),
        ("gpt_neo", "GPT Neo"),
        ("gpt_neox", "GPT NeoX"),
        ("gpt_neox_japanese", "GPT NeoX Japanese"),
        ("gpt_oss", "GptOss"),
        ("gptj", "GPT-J"),
        ("gptsan-japanese", "GPTSAN-japanese"),
        ("granite", "Granite"),
        ("granite_speech", "GraniteSpeech"),
        ("granitemoe", "GraniteMoeMoe"),
        ("granitemoehybrid", "GraniteMoeHybrid"),
        ("granitemoeshared", "GraniteMoeSharedMoe"),
        ("granitevision", "LLaVA-NeXT"),
        ("graphormer", "Graphormer"),
        ("grounding-dino", "Grounding DINO"),
        ("groupvit", "GroupViT"),
        ("helium", "Helium"),
        ("herbert", "HerBERT"),
        ("hgnet_v2", "HGNet-V2"),
        ("hiera", "Hiera"),
        ("hubert", "Hubert"),
        ("hunyuan_v1_dense", "HunYuanDenseV1"),
        ("hunyuan_v1_moe", "HunYuanMoeV1"),
        ("ibert", "I-BERT"),
        ("idefics", "IDEFICS"),
        ("idefics2", "Idefics2"),
        ("idefics3", "Idefics3"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "I-JEPA"),
        ("imagegpt", "ImageGPT"),
        ("informer", "Informer"),
        ("instructblip", "InstructBLIP"),
        ("instructblipvideo", "InstructBlipVideo"),
        ("internvl", "InternVL"),
        ("internvl_vision", "InternVLVision"),
        ("jamba", "Jamba"),
        ("janus", "Janus"),
        ("jetmoe", "JetMoe"),
        ("jukebox", "Jukebox"),
        ("kosmos-2", "KOSMOS-2"),
        ("kosmos-2.5", "KOSMOS-2.5"),
        ("kyutai_speech_to_text", "KyutaiSpeechToText"),
        ("layoutlm", "LayoutLM"),
        ("layoutlmv2", "LayoutLMv2"),
        ("layoutlmv3", "LayoutLMv3"),
        ("layoutxlm", "LayoutXLM"),
        ("led", "LED"),
        ("levit", "LeViT"),
        ("lfm2", "Lfm2"),
        ("lfm2_vl", "Lfm2Vl"),
        ("lightglue", "LightGlue"),
        ("lilt", "LiLT"),
        ("llama", "LLaMA"),
        ("llama2", "Llama2"),
        ("llama3", "Llama3"),
        ("llama4", "Llama4"),
        ("llama4_text", "Llama4ForCausalLM"),
        ("llava", "LLaVa"),
        ("llava_next", "LLaVA-NeXT"),
        ("llava_next_video", "LLaVa-NeXT-Video"),
        ("llava_onevision", "LLaVA-Onevision"),
        ("longcat_flash", "LongCatFlash"),
        ("longformer", "Longformer"),
        ("longt5", "LongT5"),
        ("luke", "LUKE"),
        ("lxmert", "LXMERT"),
        ("m2m_100", "M2M100"),
        ("madlad-400", "MADLAD-400"),
        ("mamba", "Mamba"),
        ("mamba2", "mamba2"),
        ("marian", "Marian"),
        ("markuplm", "MarkupLM"),
        ("mask2former", "Mask2Former"),
        ("maskformer", "MaskFormer"),
        ("maskformer-swin", "MaskFormerSwin"),
        ("matcha", "MatCha"),
        ("mbart", "mBART"),
        ("mbart50", "mBART-50"),
        ("mctct", "M-CTC-T"),
        ("mega", "MEGA"),
        ("megatron-bert", "Megatron-BERT"),
        ("megatron_gpt2", "Megatron-GPT2"),
        ("metaclip_2", "MetaCLIP 2"),
        ("mgp-str", "MGP-STR"),
        ("mimi", "Mimi"),
        ("minimax", "MiniMax"),
        ("ministral", "Ministral"),
        ("mistral", "Mistral"),
        ("mistral3", "Mistral3"),
        ("mixtral", "Mixtral"),
        ("mlcd", "MLCD"),
        ("mllama", "Mllama"),
        ("mluke", "mLUKE"),
        ("mm-grounding-dino", "MM Grounding DINO"),
        ("mms", "MMS"),
        ("mobilebert", "MobileBERT"),
        ("mobilenet_v1", "MobileNetV1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("mobilevit", "MobileViT"),
        ("mobilevitv2", "MobileViTV2"),
        ("modernbert", "ModernBERT"),
        ("modernbert-decoder", "ModernBertDecoder"),
        ("moonshine", "Moonshine"),
        ("moshi", "Moshi"),
        ("mpnet", "MPNet"),
        ("mpt", "MPT"),
        ("mra", "MRA"),
        ("mt5", "MT5"),
        ("musicgen", "MusicGen"),
        ("musicgen_melody", "MusicGen Melody"),
        ("mvp", "MVP"),
        ("myt5", "myt5"),
        ("nat", "NAT"),
        ("nemotron", "Nemotron"),
        ("nezha", "Nezha"),
        ("nllb", "NLLB"),
        ("nllb-moe", "NLLB-MOE"),
        ("nougat", "Nougat"),
        ("nystromformer", "Nyströmformer"),
        ("olmo", "OLMo"),
        ("olmo2", "OLMo2"),
        ("olmo3", "Olmo3"),
        ("olmoe", "OLMoE"),
        ("omdet-turbo", "OmDet-Turbo"),
        ("oneformer", "OneFormer"),
        ("open-llama", "OpenLlama"),
        ("openai-gpt", "OpenAI GPT"),
        ("opt", "OPT"),
        ("ovis2", "Ovis2"),
        ("owlv2", "OWLv2"),
        ("owlvit", "OWL-ViT"),
        ("paligemma", "PaliGemma"),
        ("parakeet", "Parakeet"),
        ("parakeet_ctc", "Parakeet"),
        ("parakeet_encoder", "ParakeetEncoder"),
        ("patchtsmixer", "PatchTSMixer"),
        ("patchtst", "PatchTST"),
        ("pegasus", "Pegasus"),
        ("pegasus_x", "PEGASUS-X"),
        ("perceiver", "Perceiver"),
        ("perception_encoder", "PerceptionEncoder"),
        ("perception_lm", "PerceptionLM"),
        ("persimmon", "Persimmon"),
        ("phi", "Phi"),
        ("phi3", "Phi3"),
        ("phi4_multimodal", "Phi4Multimodal"),
        ("phimoe", "Phimoe"),
        ("phobert", "PhoBERT"),
        ("pix2struct", "Pix2Struct"),
        ("pixtral", "Pixtral"),
        ("plbart", "PLBart"),
        ("poolformer", "PoolFormer"),
        ("pop2piano", "Pop2Piano"),
        ("prompt_depth_anything", "PromptDepthAnything"),
        ("prophetnet", "ProphetNet"),
        ("pvt", "PVT"),
        ("pvt_v2", "PVTv2"),
        ("qdqbert", "QDQBert"),
        ("qwen2", "Qwen2"),
        ("qwen2_5_omni", "Qwen2_5Omni"),
        ("qwen2_5_vl", "Qwen2_5_VL"),
        ("qwen2_5_vl_text", "Qwen2_5_VL"),
        ("qwen2_audio", "Qwen2Audio"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_moe", "Qwen2MoE"),
        ("qwen2_vl", "Qwen2VL"),
        ("qwen2_vl_text", "Qwen2VL"),
        ("qwen3", "Qwen3"),
        ("qwen3_moe", "Qwen3MoE"),
        ("qwen3_next", "Qwen3Next"),
        ("qwen3_omni_moe", "Qwen3OmniMoE"),
        ("qwen3_vl", "Qwen3VL"),
        ("qwen3_vl_moe", "Qwen3VLMoe"),
        ("qwen3_vl_moe_text", "Qwen3VLMoe"),
        ("qwen3_vl_text", "Qwen3VL"),
        ("rag", "RAG"),
        ("realm", "REALM"),
        ("recurrent_gemma", "RecurrentGemma"),
        ("reformer", "Reformer"),
        ("regnet", "RegNet"),
        ("rembert", "RemBERT"),
        ("resnet", "ResNet"),
        ("retribert", "RetriBERT"),
        ("roberta", "RoBERTa"),
        ("roberta-prelayernorm", "RoBERTa-PreLayerNorm"),
        ("roc_bert", "RoCBert"),
        ("roformer", "RoFormer"),
        ("rt_detr", "RT-DETR"),
        ("rt_detr_resnet", "RT-DETR-ResNet"),
        ("rt_detr_v2", "RT-DETRv2"),
        ("rwkv", "RWKV"),
        ("sam", "SAM"),
        ("sam2", "SAM2"),
        ("sam2_hiera_det_model", "Sam2HieraDetModel"),
        ("sam2_video", "Sam2VideoModel"),
        ("sam2_vision_model", "Sam2VisionModel"),
        ("sam_hq", "SAM-HQ"),
        ("sam_hq_vision_model", "SamHQVisionModel"),
        ("sam_vision_model", "SamVisionModel"),
        ("seamless_m4t", "SeamlessM4T"),
        ("seamless_m4t_v2", "SeamlessM4Tv2"),
        ("seed_oss", "SeedOss"),
        ("segformer", "SegFormer"),
        ("seggpt", "SegGPT"),
        ("sew", "SEW"),
        ("sew-d", "SEW-D"),
        ("shieldgemma2", "Shieldgemma2"),
        ("siglip", "SigLIP"),
        ("siglip2", "SigLIP2"),
        ("siglip2_vision_model", "Siglip2VisionModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("smollm3", "SmolLM3"),
        ("smolvlm", "SmolVLM"),
        ("smolvlm_vision", "SmolVLMVisionTransformer"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("speech_to_text", "Speech2Text"),
        ("speech_to_text_2", "Speech2Text2"),
        ("speecht5", "SpeechT5"),
        ("splinter", "Splinter"),
        ("squeezebert", "SqueezeBERT"),
        ("stablelm", "StableLm"),
        ("starcoder2", "Starcoder2"),
        ("superglue", "SuperGlue"),
        ("superpoint", "SuperPoint"),
        ("swiftformer", "SwiftFormer"),
        ("swin", "Swin Transformer"),
        ("swin2sr", "Swin2SR"),
        ("swinv2", "Swin Transformer V2"),
        ("switch_transformers", "SwitchTransformers"),
        ("t5", "T5"),
        ("t5gemma", "T5Gemma"),
        ("t5v1.1", "T5v1.1"),
        ("table-transformer", "Table Transformer"),
        ("tapas", "TAPAS"),
        ("tapex", "TAPEX"),
        ("textnet", "TextNet"),
        ("time_series_transformer", "Time Series Transformer"),
        ("timesfm", "TimesFm"),
        ("timesformer", "TimeSformer"),
        ("timm_backbone", "TimmBackbone"),
        ("timm_wrapper", "TimmWrapperModel"),
        ("trajectory_transformer", "Trajectory Transformer"),
        ("transfo-xl", "Transformer-XL"),
        ("trocr", "TrOCR"),
        ("tvlt", "TVLT"),
        ("tvp", "TVP"),
        ("udop", "UDOP"),
        ("ul2", "UL2"),
        ("umt5", "UMT5"),
        ("unispeech", "UniSpeech"),
        ("unispeech-sat", "UniSpeechSat"),
        ("univnet", "UnivNet"),
        ("upernet", "UPerNet"),
        ("van", "VAN"),
        ("vaultgemma", "VaultGemma"),
        ("video_llava", "VideoLlava"),
        ("videomae", "VideoMAE"),
        ("vilt", "ViLT"),
        ("vipllava", "VipLlava"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("vision-text-dual-encoder", "VisionTextDualEncoder"),
        ("visual_bert", "VisualBERT"),
        ("vit", "ViT"),
        ("vit_hybrid", "ViT Hybrid"),
        ("vit_mae", "ViTMAE"),
        ("vit_msn", "ViTMSN"),
        ("vitdet", "VitDet"),
        ("vitmatte", "ViTMatte"),
        ("vitpose", "ViTPose"),
        ("vitpose_backbone", "ViTPoseBackbone"),
        ("vits", "VITS"),
        ("vivit", "ViViT"),
        ("vjepa2", "VJEPA2Model"),
        ("voxtral", "Voxtral"),
        ("voxtral_encoder", "Voxtral Encoder"),
        ("wav2vec2", "Wav2Vec2"),
        ("wav2vec2-bert", "Wav2Vec2-BERT"),
        ("wav2vec2-conformer", "Wav2Vec2-Conformer"),
        ("wav2vec2_phoneme", "Wav2Vec2Phoneme"),
        ("wavlm", "WavLM"),
        ("whisper", "Whisper"),
        ("xclip", "X-CLIP"),
        ("xcodec", "X-CODEC"),
        ("xglm", "XGLM"),
        ("xlm", "XLM"),
        ("xlm-prophetnet", "XLM-ProphetNet"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("xlm-v", "XLM-V"),
        ("xlnet", "XLNet"),
        ("xls_r", "XLS-R"),
        ("xlsr_wav2vec2", "XLSR-Wav2Vec2"),
        ("xlstm", "xLSTM"),
        ("xmod", "X-MOD"),
        ("yolos", "YOLOS"),
        ("yoso", "YOSO"),
        ("zamba", "Zamba"),
        ("zamba2", "Zamba2"),
        ("zoedepth", "ZoeDepth"),
    ]
)

# This is tied to the processing `-` -> `_` in `model_type_to_module_name`. For example, instead of putting
# `transfo-xl` (as in `CONFIG_MAPPING_NAMES`), we should use `transfo_xl`.
DEPRECATED_MODELS = [
    "bort",
    "deta",
    "efficientformer",
    "ernie_m",
    "gptsan_japanese",
    "graphormer",
    "jukebox",
    "mctct",
    "mega",
    "mmbt",
    "nat",
    "nezha",
    "open_llama",
    "qdqbert",
    "realm",
    "retribert",
    "speech_to_text_2",
    "tapex",
    "trajectory_transformer",
    "transfo_xl",
    "tvlt",
    "van",
    "vit_hybrid",
    "xlm_prophetnet",
]

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict[str, str](
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2", "kosmos2"),
        ("kosmos-2.5", "kosmos2_5"),
        ("maskformer-swin", "maskformer"),
        ("xclip", "x_clip"),
        ("clip_vision_model", "clip"),
        ("qwen2_audio_encoder", "qwen2_audio"),
        ("voxtral_encoder", "voxtral"),
        ("clip_text_model", "clip"),
        ("aria_text", "aria"),
        ("gemma3_text", "gemma3"),
        ("gemma3n_audio", "gemma3n"),
        ("gemma3n_text", "gemma3n"),
        ("gemma3n_vision", "gemma3n"),
        ("glm4v_text", "glm4v"),
        ("glm4v_moe_text", "glm4v_moe"),
        ("idefics3_vision", "idefics3"),
        ("siglip_vision_model", "siglip"),
        ("siglip2_vision_model", "siglip2"),
        ("aimv2_vision_model", "aimv2"),
        ("smolvlm_vision", "smolvlm"),
        ("chinese_clip_vision_model", "chinese_clip"),
        ("rt_detr_resnet", "rt_detr"),
        ("granitevision", "llava_next"),
        ("internvl_vision", "internvl"),
        ("qwen2_5_vl_text", "qwen2_5_vl"),
        ("qwen2_vl_text", "qwen2_vl"),
        ("qwen3_vl_text", "qwen3_vl"),
        ("qwen3_vl_moe_text", "qwen3_vl_moe"),
        ("sam_vision_model", "sam"),
        ("sam2_vision_model", "sam2"),
        ("edgetam_vision_model", "edgetam"),
        ("sam2_hiera_det_model", "sam2"),
        ("sam_hq_vision_model", "sam_hq"),
        ("llama4_text", "llama4"),
        ("blip_2_qformer", "blip_2"),
        ("fastspeech2_conformer_with_hifigan", "fastspeech2_conformer"),
        ("perception_encoder", "perception_lm"),
        ("parakeet_encoder", "parakeet"),
        ("parakeet_ctc", "parakeet"),
    ]
)


def model_type_to_module_name(key) -> str:
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


def config_class_to_model_type(config) -> Union[str, None]:
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict[str, type[PretrainedConfig]]):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> type[PretrainedConfig]:
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self) -> list[str]:
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self) -> list[type[PretrainedConfig]]:
        return [self[k] for k in self._mapping] + list(self._extra_content.values())

    def items(self) -> list[tuple[str, type[PretrainedConfig]]]:
        return [(k, self[k]) for k in self._mapping] + list(self._extra_content.items())

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item: object) -> bool:
        return item in self._mapping or item in self._extra_content

    def register(self, key: str, value: type[PretrainedConfig], exist_ok=False) -> None:
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict[str, str]):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        if self._initialized:
            return

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self) -> KeysView[str]:
        self._initialize()
        return self._data.keys()

    def values(self) -> ValuesView[str]:
        self._initialize()
        return self._data.values()

    def items(self) -> KeysView[str]:
        self._initialize()
        return self._data.keys()

    def __iter__(self) -> Iterator[str]:
        self._initialize()
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        self._initialize()
        return item in self._data


def _get_class_name(model_class: Union[str, list[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(
    config_to_class=None, use_model_types: bool = True
) -> Callable[[_CallableT], _CallableT]:
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            # Example: -OO
            return fn
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs) -> PretrainedConfig:
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike[str]], **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - A path or url to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token") is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        if has_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            # Apply heuristic: if model_type is mistral but layer_types is present, treat as ministral
            if config_dict["model_type"] == "mistral" and "layer_types" in config_dict:
                logger.info(
                    "Detected mistral model with layer_types, treating as ministral for alternating attention compatibility. "
                )
                config_dict["model_type"] = "ministral"

            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date.\n\n"
                    "You can update Transformers with the command `pip install --upgrade transformers`. If this "
                    "does not work, and the checkpoint is very new, then there may not be a release version "
                    "that supports this model yet. In this case, you can get the most up-to-date code by installing "
                    "Transformers from source with the command "
                    "`pip install git+https://github.com/huggingface/transformers.git`"
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
        else:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False) -> None:
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)


__all__ = ["CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"]
