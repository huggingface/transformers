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
""" Auto Config class. """

import re
from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ..albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from ..bart.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig
from ..bert.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from ..bert_generation.configuration_bert_generation import BertGenerationConfig
from ..big_bird.configuration_big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig
from ..bigbird_pegasus.configuration_bigbird_pegasus import (
    BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BigBirdPegasusConfig,
)
from ..blenderbot.configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from ..blenderbot_small.configuration_blenderbot_small import (
    BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BlenderbotSmallConfig,
)
from ..camembert.configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from ..clip.configuration_clip import CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP, CLIPConfig
from ..convbert.configuration_convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig
from ..ctrl.configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from ..deberta.configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from ..deberta_v2.configuration_deberta_v2 import DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaV2Config
from ..deit.configuration_deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig
from ..distilbert.configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from ..dpr.configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
from ..electra.configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from ..encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from ..flaubert.configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from ..fsmt.configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
from ..funnel.configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from ..gpt2.configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from ..gpt_neo.configuration_gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig
from ..ibert.configuration_ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig
from ..layoutlm.configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from ..led.configuration_led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig
from ..longformer.configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from ..luke.configuration_luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig
from ..lxmert.configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from ..m2m_100.configuration_m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config
from ..marian.configuration_marian import MarianConfig
from ..mbart.configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig
from ..megatron_bert.configuration_megatron_bert import MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MegatronBertConfig
from ..mobilebert.configuration_mobilebert import MobileBertConfig
from ..mpnet.configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
from ..mt5.configuration_mt5 import MT5Config
from ..openai.configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from ..pegasus.configuration_pegasus import PegasusConfig
from ..prophetnet.configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from ..rag.configuration_rag import RagConfig
from ..reformer.configuration_reformer import ReformerConfig
from ..retribert.configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from ..roberta.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from ..roformer.configuration_roformer import ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RoFormerConfig
from ..speech_to_text.configuration_speech_to_text import (
    SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    Speech2TextConfig,
)
from ..squeezebert.configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from ..t5.configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from ..tapas.configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
from ..transfo_xl.configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from ..vit.configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig
from ..wav2vec2.configuration_wav2vec2 import WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2Config
from ..xlm.configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from ..xlm_prophetnet.configuration_xlm_prophetnet import (
    XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMProphetNetConfig,
)
from ..xlm_roberta.configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from ..xlnet.configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        # Add archive maps here
        ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LED_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MBART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)


CONFIG_MAPPING = OrderedDict(
    [
        # Add configs here
        ("roformer", RoFormerConfig),
        ("clip", CLIPConfig),
        ("bigbird_pegasus", BigBirdPegasusConfig),
        ("deit", DeiTConfig),
        ("luke", LukeConfig),
        ("gpt_neo", GPTNeoConfig),
        ("big_bird", BigBirdConfig),
        ("speech_to_text", Speech2TextConfig),
        ("vit", ViTConfig),
        ("wav2vec2", Wav2Vec2Config),
        ("m2m_100", M2M100Config),
        ("convbert", ConvBertConfig),
        ("led", LEDConfig),
        ("blenderbot-small", BlenderbotSmallConfig),
        ("retribert", RetriBertConfig),
        ("ibert", IBertConfig),
        ("mt5", MT5Config),
        ("t5", T5Config),
        ("mobilebert", MobileBertConfig),
        ("distilbert", DistilBertConfig),
        ("albert", AlbertConfig),
        ("bert-generation", BertGenerationConfig),
        ("camembert", CamembertConfig),
        ("xlm-roberta", XLMRobertaConfig),
        ("pegasus", PegasusConfig),
        ("marian", MarianConfig),
        ("mbart", MBartConfig),
        ("megatron_bert", MegatronBertConfig),
        ("mpnet", MPNetConfig),
        ("bart", BartConfig),
        ("blenderbot", BlenderbotConfig),
        ("reformer", ReformerConfig),
        ("longformer", LongformerConfig),
        ("roberta", RobertaConfig),
        ("deberta-v2", DebertaV2Config),
        ("deberta", DebertaConfig),
        ("flaubert", FlaubertConfig),
        ("fsmt", FSMTConfig),
        ("squeezebert", SqueezeBertConfig),
        ("bert", BertConfig),
        ("openai-gpt", OpenAIGPTConfig),
        ("gpt2", GPT2Config),
        ("transfo-xl", TransfoXLConfig),
        ("xlnet", XLNetConfig),
        ("xlm-prophetnet", XLMProphetNetConfig),
        ("prophetnet", ProphetNetConfig),
        ("xlm", XLMConfig),
        ("ctrl", CTRLConfig),
        ("electra", ElectraConfig),
        ("encoder-decoder", EncoderDecoderConfig),
        ("funnel", FunnelConfig),
        ("lxmert", LxmertConfig),
        ("dpr", DPRConfig),
        ("layoutlm", LayoutLMConfig),
        ("rag", RagConfig),
        ("tapas", TapasConfig),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("roformer", "RoFormer"),
        ("clip", "CLIP"),
        ("bigbird_pegasus", "BigBirdPegasus"),
        ("deit", "DeiT"),
        ("luke", "LUKE"),
        ("gpt_neo", "GPT Neo"),
        ("big_bird", "BigBird"),
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
        ("pegasus", "Pegasus"),
        ("blenderbot", "Blenderbot"),
        ("marian", "Marian"),
        ("mbart", "mBART"),
        ("megatron_bert", "MegatronBert"),
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
    ]
)


def _get_class_name(model_class):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f":class:`~transformers.{c.__name__}`" for c in model_class])
    return f":class:`~transformers.{model_class.__name__}`"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {
                model_type: f":class:`~transformers.{config.__name__}`"
                for model_type, config in CONFIG_MAPPING.items()
            }
        else:
            model_type_to_name = {
                model_type: _get_class_name(config_to_class[config])
                for model_type, config in CONFIG_MAPPING.items()
                if config in config_to_class
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {config.__name__: _get_class_name(clas) for config, clas in config_to_class.items()}
        config_to_model_name = {
            config.__name__: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING.items()
        }
        lines = [
            f"{indent}- :class:`~transformers.{config_name}` configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
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
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
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

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config object
        that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                      namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        Examples::

            >>> from transformers import AutoConfig

            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')

            >>> # Download configuration from huggingface.co (user-uploaded) and cache.
            >>> config = AutoConfig.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If configuration file is in a directory (e.g., was saved using `save_pretrained('./test/saved_model/')`).
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/')

            >>> # Load a specific configuration file.
            >>> config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')

            >>> # Change some config attributes when loading a pretrained config.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            >>> config.output_attentions
            True
            >>> config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
            >>> config.output_attentions
            True
            >>> config.unused_kwargs
            {'foo': False}
        """
        kwargs["_from_auto"] = True
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in str(pretrained_model_name_or_path):
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            "Should have a `model_type` key in its config.json, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )
