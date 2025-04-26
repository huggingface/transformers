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
from typing import TYPE_CHECKING

from ..utils import _LazyModule
from ..utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .albert import *
    from .align import *
    from .altclip import *
    from .aria import *
    from .audio_spectrogram_transformer import *
    from .auto import *
    from .autoformer import *
    from .aya_vision import *
    from .bamba import *
    from .bark import *
    from .bart import *
    from .barthez import *
    from .bartpho import *
    from .beit import *
    from .bert import *
    from .bert_generation import *
    from .bert_japanese import *
    from .bertweet import *
    from .big_bird import *
    from .bigbird_pegasus import *
    from .biogpt import *
    from .bit import *
    from .blenderbot import *
    from .blenderbot_small import *
    from .blip import *
    from .blip_2 import *
    from .bloom import *
    from .bridgetower import *
    from .bros import *
    from .byt5 import *
    from .camembert import *
    from .canine import *
    from .chameleon import *
    from .chinese_clip import *
    from .clap import *
    from .clip import *
    from .clipseg import *
    from .clvp import *
    from .code_llama import *
    from .codegen import *
    from .cohere import *
    from .cohere2 import *
    from .colpali import *
    from .conditional_detr import *
    from .convbert import *
    from .convnext import *
    from .convnextv2 import *
    from .cpm import *
    from .cpmant import *
    from .ctrl import *
    from .cvt import *
    from .dab_detr import *
    from .dac import *
    from .data2vec import *
    from .dbrx import *
    from .deberta import *
    from .deberta_v2 import *
    from .decision_transformer import *
    from .deformable_detr import *
    from .deit import *
    from .deprecated import *
    from .depth_anything import *
    from .depth_pro import *
    from .detr import *
    from .dialogpt import *
    from .diffllama import *
    from .dinat import *
    from .dinov2 import *
    from .dinov2_with_registers import *
    from .distilbert import *
    from .dit import *
    from .donut import *
    from .dpr import *
    from .dpt import *
    from .efficientnet import *
    from .electra import *
    from .emu3 import *
    from .encodec import *
    from .encoder_decoder import *
    from .ernie import *
    from .esm import *
    from .falcon import *
    from .falcon_mamba import *
    from .fastspeech2_conformer import *
    from .flaubert import *
    from .flava import *
    from .fnet import *
    from .focalnet import *
    from .fsmt import *
    from .funnel import *
    from .fuyu import *
    from .gemma import *
    from .gemma2 import *
    from .gemma3 import *
    from .git import *
    from .glm import *
    from .glm4 import *
    from .glpn import *
    from .got_ocr2 import *
    from .gpt2 import *
    from .gpt_bigcode import *
    from .gpt_neo import *
    from .gpt_neox import *
    from .gpt_neox_japanese import *
    from .gpt_sw3 import *
    from .gptj import *
    from .granite import *
    from .granite_speech import *
    from .granitemoe import *
    from .granitemoeshared import *
    from .grounding_dino import *
    from .groupvit import *
    from .helium import *
    from .herbert import *
    from .hiera import *

    # --- !!! ADD YOUR MODEL IMPORT HERE (for type checking) !!! ---
    from .hindi_causal_lm import *

    # --- !!! END ADDITION !!! ---
    from .hubert import *
    from .ibert import *
    from .idefics import *
    from .idefics2 import *
    from .idefics3 import *
    from .ijepa import *
    from .imagegpt import *
    from .informer import *
    from .instructblip import *
    from .instructblipvideo import *
    from .internvl import *
    from .jamba import *
    from .janus import *
    from .jetmoe import *
    from .kosmos2 import *
    from .layoutlm import *
    from .layoutlmv2 import *
    from .layoutlmv3 import *
    from .layoutxlm import *
    from .led import *
    from .levit import *
    from .lilt import *
    from .llama import *
    from .llama4 import *
    from .llava import *
    from .llava_next import *
    from .llava_next_video import *
    from .llava_onevision import *
    from .longformer import *
    from .longt5 import *
    from .luke import *
    from .lxmert import *
    from .m2m_100 import *
    from .mamba import *
    from .mamba2 import *
    from .marian import *
    from .markuplm import *
    from .mask2former import *
    from .maskformer import *
    from .mbart import *
    from .mbart50 import *
    from .megatron_bert import *
    from .megatron_gpt2 import *
    from .mgp_str import *
    from .mimi import *
    from .mistral import *
    from .mistral3 import *
    from .mixtral import *
    from .mlcd import *
    from .mllama import *
    from .mluke import *
    from .mobilebert import *
    from .mobilenet_v1 import *
    from .mobilenet_v2 import *
    from .mobilevit import *
    from .mobilevitv2 import *
    from .modernbert import *
    from .moonshine import *
    from .moshi import *
    from .mpnet import *
    from .mpt import *
    from .mra import *
    from .mt5 import *
    from .musicgen import *
    from .musicgen_melody import *
    from .mvp import *
    from .myt5 import *
    from .nemotron import *
    from .nllb import *
    from .nllb_moe import *
    from .nougat import *
    from .nystromformer import *
    from .olmo import *
    from .olmo2 import *
    from .olmoe import *
    from .omdet_turbo import *
    from .oneformer import *
    from .openai import *
    from .opt import *
    from .owlv2 import *
    from .owlvit import *
    from .paligemma import *
    from .patchtsmixer import *
    from .patchtst import *
    from .pegasus import *
    from .pegasus_x import *
    from .perceiver import *
    from .persimmon import *
    from .phi import *
    from .phi3 import *
    from .phi4_multimodal import *
    from .phimoe import *
    from .phobert import *
    from .pix2struct import *
    from .pixtral import *
    from .plbart import *
    from .poolformer import *
    from .pop2piano import *
    from .prophetnet import *
    from .pvt import *
    from .pvt_v2 import *
    from .qwen2 import *
    from .qwen2_5_vl import *
    from .qwen2_audio import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .rag import *
    from .recurrent_gemma import *
    from .reformer import *
    from .regnet import *
    from .rembert import *
    from .resnet import *
    from .roberta import *
    from .roberta_prelayernorm import *
    from .roc_bert import *
    from .roformer import *
    from .rt_detr import *
    from .rt_detr_v2 import *
    from .rwkv import *
    from .sam import *
    from .seamless_m4t import *
    from .seamless_m4t_v2 import *
    from .segformer import *
    from .seggpt import *
    from .sew import *
    from .sew_d import *
    from .siglip import *
    from .siglip2 import *
    from .smolvlm import *
    from .speech_encoder_decoder import *
    from .speech_to_text import *
    from .speecht5 import *
    from .splinter import *
    from .squeezebert import *
    from .stablelm import *
    from .starcoder2 import *
    from .superglue import *
    from .superpoint import *
    from .swiftformer import *
    from .swin import *
    from .swin2sr import *
    from .swinv2 import *
    from .switch_transformers import *
    from .t5 import *
    from .table_transformer import *
    from .tapas import *
    from .textnet import *
    from .time_series_transformer import *
    from .timesfm import *
    from .timesformer import *
    from .timm_backbone import *
    from .timm_wrapper import *
    from .trocr import *
    from .tvp import *
    from .udop import *
    from .umt5 import *
    from .unispeech import *
    from .unispeech_sat import *
    from .univnet import *
    from .upernet import *
    from .video_llava import *
    from .videomae import *
    from .vilt import *
    from .vipllava import *
    from .vision_encoder_decoder import *
    from .vision_text_dual_encoder import *
    from .visual_bert import *
    from .vit import *
    from .vit_mae import *
    from .vit_msn import *
    from .vitdet import *
    from .vitmatte import *
    from .vitpose import *
    from .vitpose_backbone import *
    from .vits import *
    from .vivit import *
    from .wav2vec2 import *
    from .wav2vec2_bert import *
    from .wav2vec2_conformer import *
    from .wav2vec2_phoneme import *
    from .wav2vec2_with_lm import *
    from .wavlm import *
    from .whisper import *
    from .x_clip import *
    from .xglm import *
    from .xlm import *
    from .xlm_roberta import *
    from .xlm_roberta_xl import *
    from .xlnet import *
    from .xmod import *
    from .yolos import *
    from .yoso import *
    from .zamba import *
    from .zamba2 import *
    from .zoedepth import *
else:
    import sys

    _SUBMODELS = {  # This dictionary links model_type to the module directory name
        "albert": "albert",
        "align": "align",
        "altclip": "altclip",
        "aria": "aria",
        "audio-spectrogram-transformer": "audio_spectrogram_transformer",
        "autoformer": "autoformer",
        "aya_vision": "aya_vision",
        "bamba": "bamba",
        "bark": "bark",
        "bart": "bart",
        "beit": "beit",
        "bert": "bert",
        "bert-generation": "bert_generation",
        "big_bird": "big_bird",
        "bigbird_pegasus": "bigbird_pegasus",
        "biogpt": "biogpt",
        "bit": "bit",
        "blenderbot": "blenderbot",
        "blenderbot-small": "blenderbot_small",
        "blip": "blip",
        "blip-2": "blip_2",
        "bloom": "bloom",
        "bridgetower": "bridgetower",
        "bros": "bros",
        "camembert": "camembert",
        "canine": "canine",
        "chameleon": "chameleon",
        "chinese_clip": "chinese_clip",
        "clap": "clap",
        "clip": "clip",
        "clipseg": "clipseg",
        "clvp": "clvp",
        "code_llama": "llama",  # Uses llama module
        "codegen": "codegen",
        "cohere": "cohere",
        "cohere2": "cohere2",
        "colpali": "colpali",
        "conditional_detr": "conditional_detr",
        "convbert": "convbert",
        "convnext": "convnext",
        "convnextv2": "convnextv2",
        "cpmant": "cpmant",
        "ctrl": "ctrl",
        "cvt": "cvt",
        "dab-detr": "dab_detr",
        "dac": "dac",
        "data2vec-audio": "data2vec",  # Uses data2vec module
        "data2vec-text": "data2vec",  # Uses data2vec module
        "data2vec-vision": "data2vec",  # Uses data2vec module
        "dbrx": "dbrx",
        "deberta": "deberta",
        "deberta-v2": "deberta_v2",
        "decision_transformer": "decision_transformer",
        "deepseek_v3": "deepseek_v3",
        "deformable_detr": "deformable_detr",
        "deit": "deit",
        "depth_anything": "depth_anything",
        "depth_pro": "depth_pro",
        "deta": "deta",
        "detr": "detr",
        "diffllama": "diffllama",
        "dinat": "dinat",
        "dinov2": "dinov2",
        "dinov2_with_registers": "dinov2_with_registers",
        "distilbert": "distilbert",
        "donut": "donut",  # Special case handled below
        "dpr": "dpr",
        "dpt": "dpt",
        "efficientformer": "efficientformer",
        "efficientnet": "efficientnet",
        "electra": "electra",
        "emu3": "emu3",
        "encodec": "encodec",
        "encoder-decoder": "encoder_decoder",
        "ernie": "ernie",
        "ernie_m": "ernie_m",  # In deprecated
        "esm": "esm",
        "falcon": "falcon",
        "falcon_mamba": "falcon_mamba",
        "fastspeech2_conformer": "fastspeech2_conformer",
        "flaubert": "flaubert",
        "flava": "flava",
        "fnet": "fnet",
        "focalnet": "focalnet",
        "fsmt": "fsmt",
        "funnel": "funnel",
        "fuyu": "fuyu",
        "gemma": "gemma",
        "gemma2": "gemma2",
        "gemma3": "gemma3",
        "git": "git",
        "glm": "glm",
        "glm4": "glm4",
        "glpn": "glpn",
        "got_ocr2": "got_ocr2",
        "gpt-sw3": "gpt_sw3",
        "gpt2": "gpt2",
        "gpt_bigcode": "gpt_bigcode",
        "gpt_neo": "gpt_neo",
        "gpt_neox": "gpt_neox",
        "gpt_neox_japanese": "gpt_neox_japanese",
        "gptj": "gptj",
        "gptsan-japanese": "gptsan_japanese",  # In deprecated
        "granite": "granite",
        "granite_speech": "granite_speech",
        "granitemoe": "granitemoe",
        "granitemoeshared": "granitemoeshared",
        "graphormer": "graphormer",  # In deprecated
        "grounding-dino": "grounding_dino",
        "groupvit": "groupvit",
        "helium": "helium",
        "herbert": "herbert",
        "hiera": "hiera",
        # --- !!! ADD YOUR MODEL TYPE HERE !!! ---
        "hindi_causal_lm": "hindi_causal_lm",
        # --- !!! END ADDITION !!! ---
        "hubert": "hubert",
        "ibert": "ibert",
        "idefics": "idefics",
        "idefics2": "idefics2",
        "idefics3": "idefics3",
        "ijepa": "ijepa",
        "imagegpt": "imagegpt",
        "informer": "informer",
        "instructblip": "instructblip",
        "instructblipvideo": "instructblipvideo",
        "internvl": "internvl",
        "jamba": "jamba",
        "janus": "janus",
        "jetmoe": "jetmoe",
        "jukebox": "jukebox",  # In deprecated
        "kosmos-2": "kosmos2",  # Special case handled below
        "layoutlm": "layoutlm",
        "layoutlmv2": "layoutlmv2",
        "layoutlmv3": "layoutlmv3",
        "led": "led",
        "levit": "levit",
        "lilt": "lilt",
        "llama": "llama",
        "llama4": "llama4",
        "llava": "llava",
        "llava_next": "llava_next",
        "llava_next_video": "llava_next_video",
        "llava_onevision": "llava_onevision",
        "longformer": "longformer",
        "longt5": "longt5",
        "luke": "luke",
        "lxmert": "lxmert",
        "m2m_100": "m2m_100",
        "mamba": "mamba",
        "mamba2": "mamba2",
        "marian": "marian",
        "markuplm": "markuplm",
        "mask2former": "mask2former",
        "maskformer": "maskformer",
        "mbart": "mbart",
        "mctct": "mctct",  # In deprecated
        "mega": "mega",  # In deprecated
        "megatron-bert": "megatron_bert",
        "mgp-str": "mgp_str",
        "mimi": "mimi",
        "mistral": "mistral",
        "mistral3": "mistral3",
        "mixtral": "mixtral",
        "mlcd": "mlcd",
        "mllama": "mllama",
        "mluke": "mluke",
        "mobilebert": "mobilebert",
        "mobilenet_v1": "mobilenet_v1",
        "mobilenet_v2": "mobilenet_v2",
        "mobilevit": "mobilevit",
        "mobilevitv2": "mobilevitv2",
        "modernbert": "modernbert",
        "moonshine": "moonshine",
        "moshi": "moshi",
        "mpnet": "mpnet",
        "mpt": "mpt",
        "mra": "mra",
        "mt5": "mt5",
        "musicgen": "musicgen",
        "musicgen_melody": "musicgen_melody",
        "mvp": "mvp",
        "nat": "nat",  # In deprecated
        "nemotron": "nemotron",
        "nezha": "nezha",  # In deprecated
        "nllb": "nllb",  # Special case handled below
        "nllb-moe": "nllb_moe",
        "nougat": "nougat",
        "nystromformer": "nystromformer",
        "olmo": "olmo",
        "olmo2": "olmo2",
        "olmoe": "olmoe",
        "omdet-turbo": "omdet_turbo",
        "oneformer": "oneformer",
        "openai-gpt": "openai",  # Special case handled below
        "opt": "opt",
        "owlv2": "owlv2",
        "owlvit": "owlvit",
        "paligemma": "paligemma",
        "patchtsmixer": "patchtsmixer",
        "patchtst": "patchtst",
        "pegasus": "pegasus",
        "pegasus_x": "pegasus_x",
        "perceiver": "perceiver",
        "persimmon": "persimmon",
        "phi": "phi",
        "phi3": "phi3",
        "phi4_multimodal": "phi4_multimodal",
        "phimoe": "phimoe",
        "phobert": "phobert",
        "pix2struct": "pix2struct",
        "pixtral": "pixtral",
        "plbart": "plbart",
        "poolformer": "poolformer",
        "pop2piano": "pop2piano",
        "prophetnet": "prophetnet",
        "pvt": "pvt",
        "pvt_v2": "pvt_v2",
        "qdqbert": "qdqbert",  # In deprecated
        "qwen2": "qwen2",
        "qwen2_5_vl": "qwen2_5_vl",
        "qwen2_audio": "qwen2_audio",
        "qwen2_moe": "qwen2_moe",  # Special case handled below
        "qwen2_vl": "qwen2_vl",  # Special case handled below
        "rag": "rag",
        "realm": "realm",  # In deprecated
        "recurrent_gemma": "recurrent_gemma",  # Special case handled below
        "reformer": "reformer",
        "regnet": "regnet",
        "rembert": "rembert",
        "resnet": "resnet",
        "retribert": "retribert",  # In deprecated
        "roberta": "roberta",
        "roberta-prelayernorm": "roberta_prelayernorm",
        "roc_bert": "roc_bert",
        "roformer": "roformer",
        "rt-detr": "rt_detr",  # Special case handled below
        "rt_detr_v2": "rt_detr_v2",
        "rwkv": "rwkv",
        "sam": "sam",
        "seamless_m4t": "seamless_m4t",
        "seamless_m4t_v2": "seamless_m4t_v2",
        "segformer": "segformer",  # Special case handled below
        "seggpt": "seggpt",  # Special case handled below
        "sew": "sew",
        "sew-d": "sew_d",
        "shieldgemma2": "shieldgemma2",
        "siglip": "siglip",
        "siglip2": "siglip2",  # Special case handled below
        "smolvlm": "smolvlm",
        "speech-encoder-decoder": "speech_encoder_decoder",
        "speech_to_text": "speech_to_text",
        "speech_to_text_2": "speech_to_text_2",  # In deprecated
        "speecht5": "speecht5",  # Special case handled below
        "splinter": "splinter",
        "squeezebert": "squeezebert",
        "stablelm": "stablelm",  # Special case handled below
        "starcoder2": "starcoder2",
        "superglue": "superglue",
        "superpoint": "superpoint",
        "swiftformer": "swiftformer",
        "swin": "swin",
        "swin2sr": "swin2sr",
        "swinv2": "swinv2",
        "switch_transformers": "switch_transformers",
        "t5": "t5",
        "table-transformer": "table_transformer",
        "tapas": "tapas",
        "tapex": "tapex",  # In deprecated
        "textnet": "textnet",
        "time_series_transformer": "time_series_transformer",
        "timesfm": "timesfm",  # Special case handled below
        "timesformer": "timesformer",
        "timm_backbone": "timm_backbone",
        "timm_wrapper": "timm_wrapper",  # Special case handled below
        "trajectory_transformer": "trajectory_transformer",  # In deprecated
        "transfo-xl": "transfo_xl",  # Special case handled below & in deprecated
        "trocr": "trocr",
        "tvlt": "tvlt",  # In deprecated
        "tvp": "tvp",
        "udop": "udop",  # Special case handled below
        "umt5": "umt5",
        "unispeech": "unispeech",
        "unispeech-sat": "unispeech_sat",
        "univnet": "univnet",
        "upernet": "upernet",
        "van": "van",  # In deprecated
        "video_llava": "video_llava",
        "videomae": "videomae",
        "vilt": "vilt",
        "vipllava": "vipllava",
        "vision-encoder-decoder": "vision_encoder_decoder",
        "vision-text-dual-encoder": "vision_text_dual_encoder",
        "visual_bert": "visual_bert",
        "vit": "vit",
        "vit_hybrid": "vit_hybrid",  # In deprecated
        "vit_mae": "vit_mae",
        "vit_msn": "vit_msn",
        "vitdet": "vitdet",
        "vitmatte": "vitmatte",
        "vitpose": "vitpose",
        "vitpose_backbone": "vitpose_backbone",
        "vits": "vits",
        "vivit": "vivit",
        "wav2vec2": "wav2vec2",
        "wav2vec2-bert": "wav2vec2_bert",
        "wav2vec2-conformer": "wav2vec2_conformer",
        "wavlm": "wavlm",
        "whisper": "whisper",
        "x-clip": "x_clip",  # Special case handled below
        "xglm": "xglm",
        "xlm": "xlm",
        "xlm-prophetnet": "xlm_prophetnet",  # In deprecated
        "xlm-roberta": "xlm_roberta",
        "xlm-roberta-xl": "xlm_roberta_xl",
        "xlnet": "xlnet",
        "yolos": "yolos",
        "yoso": "yoso",
        "zamba": "zamba",  # Special case handled below
        "zamba2": "zamba2",  # Special case handled below
        "zoedepth": "zoedepth",
    }
    # This dictionary is intentionally incomplete. Existing entries are those where the model_type differs
    # from the module name. Only add new entries here if the model_type has a `-` or relates
    # to a different existing model module.
    # For example -> `donut-swin` uses the `donut` module.
    # `data2vec-audio` uses the `data2vec` module.
    _SUBMODELS["blip_2"] = "blip_2"
    _SUBMODELS["data2vec"] = "data2vec"
    _SUBMODELS["donut"] = "donut"
    _SUBMODELS["kosmos2"] = "kosmos2"
    _SUBMODELS["nllb"] = "nllb"
    _SUBMODELS["openai"] = "openai"
    _SUBMODELS["qwen2_moe"] = "qwen2_moe"
    _SUBMODELS["qwen2_vl"] = "qwen2_vl"
    _SUBMODELS["qwen3"] = "qwen3"
    _SUBMODELS["qwen3_moe"] = "qwen3_moe"
    _SUBMODELS["recurrent_gemma"] = "recurrent_gemma"
    _SUBMODELS["rt_detr"] = "rt_detr"
    _SUBMODELS["segformer"] = "segformer"
    _SUBMODELS["seggpt"] = "seggpt"
    _SUBMODELS["siglip2"] = "siglip2"
    _SUBMODELS["speecht5"] = "speecht5"
    _SUBMODELS["stablelm"] = "stablelm"
    _SUBMODELS["timesfm"] = "timesfm"
    _SUBMODELS["timm_wrapper"] = "timm_wrapper"
    _SUBMODELS["transfo_xl"] = "transfo_xl"
    _SUBMODELS["udop"] = "udop"
    _SUBMODELS["x_clip"] = "x_clip"
    _SUBMODELS["zamba"] = "zamba"
    _SUBMODELS["zamba2"] = "zamba2"

    _file = globals()["__file__"]
    _import_structure = define_import_structure(_file, _SUBMODELS)
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)

# Make sure that all architectures are characterized by the constant typing in TESTING_WITHOUT_PT/TF/FLAX + ONNX_EXPORT
# _SUPPORTED_MODELS = set(value for model_list in _SUPPORTED_MODELS_PT + _SUPPORTED_MODELS_TF + _SUPPORTED_MODELS_FLAX for value in model_list)
__all__ = list(_SUBMODELS.keys())


def model_info(model_type: str):
    """ """
    if model_type in _SUBMODELS:
        module_name = _SUBMODELS[model_type]
        return module_name

    raise KeyError(model_type)


__all__.append("model_info")
