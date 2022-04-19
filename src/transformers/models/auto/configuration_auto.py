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
""" Auto Config class."""
import importlib
import re
import warnings
from collections import OrderedDict
from typing import List, Union

from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module
from ...utils import CONFIG_NAME, logging


logger = logging.get_logger(__name__)

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("tapex", "BartConfig"),
        ("dpt", "DPTConfig"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("glpn", "GLPNConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("convnext", "ConvNextConfig"),
        ("van", "VanConfig"),
        ("resnet", "ResNetConfig"),
        ("regnet", "RegNetConfig"),
        ("yoso", "YosoConfig"),
        ("swin", "SwinConfig"),
        ("vilt", "ViltConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("realm", "RealmConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("xglm", "XGLMConfig"),
        ("imagegpt", "ImageGPTConfig"),
        ("qdqbert", "QDQBertConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("trocr", "TrOCRConfig"),
        ("fnet", "FNetConfig"),
        ("segformer", "SegformerConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("perceiver", "PerceiverConfig"),
        ("gptj", "GPTJConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("plbart", "PLBartConfig"),
        ("beit", "BeitConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("rembert", "RemBertConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("canine", "CanineConfig"),
        ("roformer", "RoFormerConfig"),
        ("clip", "CLIPConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("deit", "DeiTConfig"),
        ("luke", "LukeConfig"),
        ("detr", "DetrConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("big_bird", "BigBirdConfig"),
        ("speech_to_text_2", "Speech2Text2Config"),
        ("speech_to_text", "Speech2TextConfig"),
        ("vit", "ViTConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("m2m_100", "M2M100Config"),
        ("convbert", "ConvBertConfig"),
        ("led", "LEDConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("retribert", "RetriBertConfig"),
        ("ibert", "IBertConfig"),
        ("mt5", "MT5Config"),
        ("t5", "T5Config"),
        ("mobilebert", "MobileBertConfig"),
        ("distilbert", "DistilBertConfig"),
        ("albert", "AlbertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("camembert", "CamembertConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("pegasus", "PegasusConfig"),
        ("marian", "MarianConfig"),
        ("mbart", "MBartConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mpnet", "MPNetConfig"),
        ("bart", "BartConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("reformer", "ReformerConfig"),
        ("longformer", "LongformerConfig"),
        ("roberta", "RobertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("deberta", "DebertaConfig"),
        ("flaubert", "FlaubertConfig"),
        ("fsmt", "FSMTConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("hubert", "HubertConfig"),
        ("bert", "BertConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("gpt2", "GPT2Config"),
        ("transfo-xl", "TransfoXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("xlm", "XLMConfig"),
        ("ctrl", "CTRLConfig"),
        ("electra", "ElectraConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("funnel", "FunnelConfig"),
        ("lxmert", "LxmertConfig"),
        ("dpr", "DPRConfig"),
        ("layoutlm", "LayoutLMConfig"),
        ("rag", "RagConfig"),
        ("tapas", "TapasConfig"),
        ("splinter", "SplinterConfig"),
        ("sew-d", "SEWDConfig"),
        ("sew", "SEWConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("unispeech", "UniSpeechConfig"),
        ("wavlm", "WavLMConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
    ]
)

CONFIG_ARCHIVE_MAP_MAPPING_NAMES = OrderedDict(
    [
        # Add archive maps here)
        ("dpt", "DPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("glpn", "GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("maskformer", "MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("poolformer", "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convnext", "CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("van", "VAN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("resnet", "RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("regnet", "REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("yoso", "YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swin", "SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vilt", "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_mae", "VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("realm", "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nystromformer", "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xglm", "XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("imagegpt", "IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("qdqbert", "QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fnet", "FNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pegasus", "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("segformer", "SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("perceiver", "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gptj", "GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlmv2", "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("plbart", "PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("beit", "BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-vision", "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("rembert", "REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("visual_bert", "VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("canine", "CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roformer", "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("clip", "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bigbird_pegasus", "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deit", "DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("luke", "LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("detr", "DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neo", "GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("big_bird", "BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("megatron-bert", "MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text", "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text_2", "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit", "VIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2", "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("m2m_100", "M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convbert", "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("led", "LED_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot-small", "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bert", "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bart", "BART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot", "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mbart", "MBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("openai-gpt", "OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("transfo-xl", "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt2", "GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ctrl", "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlnet", "XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm", "XLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roberta", "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-text", "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-audio", "DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("distilbert", "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("albert", "ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("camembert", "CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("t5", "T5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-roberta", "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("flaubert", "FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fsmt", "FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("electra", "ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("longformer", "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("retribert", "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("funnel", "FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("lxmert", "LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlm", "LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dpr", "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta", "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta-v2", "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("squeezebert", "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-prophetnet", "XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("prophetnet", "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mpnet", "MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tapas", "TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ibert", "IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("hubert", "HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("splinter", "SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew-d", "SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew", "SEW_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech-sat", "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech", "UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP"),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("tapex", "TAPEX"),
        ("dpt", "DPT"),
        ("decision_transformer", "Decision Transformer"),
        ("glpn", "GLPN"),
        ("maskformer", "MaskFormer"),
        ("poolformer", "PoolFormer"),
        ("convnext", "ConvNext"),
        ("van", "VAN"),
        ("resnet", "ResNet"),
        ("regnet", "RegNet"),
        ("yoso", "YOSO"),
        ("swin", "Swin"),
        ("vilt", "ViLT"),
        ("vit_mae", "ViTMAE"),
        ("realm", "Realm"),
        ("nystromformer", "Nystromformer"),
        ("xglm", "XGLM"),
        ("imagegpt", "ImageGPT"),
        ("qdqbert", "QDQBert"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("trocr", "TrOCR"),
        ("fnet", "FNet"),
        ("segformer", "SegFormer"),
        ("vision-text-dual-encoder", "VisionTextDualEncoder"),
        ("perceiver", "Perceiver"),
        ("gptj", "GPT-J"),
        ("beit", "BEiT"),
        ("plbart", "PLBart"),
        ("rembert", "RemBERT"),
        ("layoutlmv2", "LayoutLMv2"),
        ("visual_bert", "VisualBert"),
        ("canine", "Canine"),
        ("roformer", "RoFormer"),
        ("clip", "CLIP"),
        ("bigbird_pegasus", "BigBirdPegasus"),
        ("deit", "DeiT"),
        ("luke", "LUKE"),
        ("detr", "DETR"),
        ("gpt_neo", "GPT Neo"),
        ("big_bird", "BigBird"),
        ("speech_to_text_2", "Speech2Text2"),
        ("speech_to_text", "Speech2Text"),
        ("vit", "ViT"),
        ("wav2vec2", "Wav2Vec2"),
        ("m2m_100", "M2M100"),
        ("convbert", "ConvBERT"),
        ("led", "LED"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("retribert", "RetriBERT"),
        ("ibert", "I-BERT"),
        ("t5", "T5"),
        ("mobilebert", "MobileBERT"),
        ("distilbert", "DistilBERT"),
        ("albert", "ALBERT"),
        ("bert-generation", "Bert Generation"),
        ("camembert", "CamemBERT"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("pegasus", "Pegasus"),
        ("blenderbot", "Blenderbot"),
        ("marian", "Marian"),
        ("mbart", "mBART"),
        ("megatron-bert", "MegatronBert"),
        ("bart", "BART"),
        ("reformer", "Reformer"),
        ("longformer", "Longformer"),
        ("roberta", "RoBERTa"),
        ("flaubert", "FlauBERT"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("squeezebert", "SqueezeBERT"),
        ("bert", "BERT"),
        ("openai-gpt", "OpenAI GPT"),
        ("gpt2", "OpenAI GPT-2"),
        ("transfo-xl", "Transformer-XL"),
        ("xlnet", "XLNet"),
        ("xlm", "XLM"),
        ("ctrl", "CTRL"),
        ("electra", "ELECTRA"),
        ("encoder-decoder", "Encoder decoder"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("funnel", "Funnel Transformer"),
        ("lxmert", "LXMERT"),
        ("deberta-v2", "DeBERTa-v2"),
        ("deberta", "DeBERTa"),
        ("layoutlm", "LayoutLM"),
        ("dpr", "DPR"),
        ("rag", "RAG"),
        ("xlm-prophetnet", "XLMProphetNet"),
        ("prophetnet", "ProphetNet"),
        ("mt5", "mT5"),
        ("mpnet", "MPNet"),
        ("tapas", "TAPAS"),
        ("hubert", "Hubert"),
        ("barthez", "BARThez"),
        ("phobert", "PhoBERT"),
        ("bartpho", "BARTpho"),
        ("cpm", "CPM"),
        ("bertweet", "Bertweet"),
        ("bert-japanese", "BertJapanese"),
        ("byt5", "ByT5"),
        ("mbart50", "mBART-50"),
        ("splinter", "Splinter"),
        ("sew-d", "SEW-D"),
        ("sew", "SEW"),
        ("unispeech-sat", "UniSpeechSat"),
        ("unispeech", "UniSpeech"),
        ("wavlm", "WavLM"),
        ("bort", "BORT"),
        ("dialogpt", "DialoGPT"),
        ("xls_r", "XLS-R"),
        ("t5v1.1", "T5v1.1"),
        ("herbert", "HerBERT"),
        ("wav2vec2_phoneme", "Wav2Vec2Phoneme"),
        ("megatron_gpt2", "MegatronGPT2"),
        ("xlsr_wav2vec2", "XLSR-Wav2Vec2"),
        ("mluke", "mLUKE"),
        ("layoutxlm", "LayoutXLM"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("dit", "DiT"),
    ]
)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    return key.replace("-", "_")


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
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

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys():
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict):
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
        warnings.warn(
            "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. "
            "It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.",
            FutureWarning,
        )

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self):
        self._initialize()
        return self._data.keys()

    def values(self):
        self._initialize()
        return self._data.values()

    def items(self):
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        self._initialize()
        return item in self._data


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = _LazyLoadAllMappings(CONFIG_ARCHIVE_MAP_MAPPING_NAMES)


def _get_class_name(model_class: Union[str, List[str]]):
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
            f"{indent}- [`{config_name}`] configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
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
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}"
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

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                      namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
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
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
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
        >>> config = AutoConfig.from_pretrained("bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> config.unused_kwargs
        {'foo': False}
        ```"""
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the configuration file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warning(
                    "Explicitly passing a `revision` is encouraged when loading a configuration with custom code to "
                    "ensure no malicious code has been contributed in a newer revision."
                )
            class_ref = config_dict["auto_map"]["AutoConfig"]
            module_file, class_name = class_ref.split(".")
            config_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **kwargs
            )
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in str(pretrained_model_name_or_path):
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config):
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
        CONFIG_MAPPING.register(model_type, config)
