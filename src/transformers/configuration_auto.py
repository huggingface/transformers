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


import logging
from collections import OrderedDict

from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig, MBartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_marian import MarianConfig
from .configuration_mobilebert import MobileBertConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_reformer import ReformerConfig
from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig


logger = logging.getLogger(__name__)


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BART_PRETRAINED_CONFIG_ARCHIVE_MAP,
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
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)


CONFIG_MAPPING = OrderedDict(
    [
        ("retribert", RetriBertConfig,),
        ("t5", T5Config,),
        ("mobilebert", MobileBertConfig,),
        ("distilbert", DistilBertConfig,),
        ("albert", AlbertConfig,),
        ("camembert", CamembertConfig,),
        ("xlm-roberta", XLMRobertaConfig,),
        ("marian", MarianConfig,),
        ("mbart", MBartConfig,),
        ("bart", BartConfig,),
        ("reformer", ReformerConfig,),
        ("longformer", LongformerConfig,),
        ("roberta", RobertaConfig,),
        ("flaubert", FlaubertConfig,),
        ("bert", BertConfig,),
        ("openai-gpt", OpenAIGPTConfig,),
        ("gpt2", GPT2Config,),
        ("transfo-xl", TransfoXLConfig,),
        ("xlnet", XLNetConfig,),
        ("xlm", XLMConfig,),
        ("ctrl", CTRLConfig,),
        ("electra", ElectraConfig,),
        ("encoder-decoder", EncoderDecoderConfig,),
    ]
)


class AutoConfig:
    r"""
        :class:`~transformers.AutoConfig` is a generic configuration class
        that will be instantiated as one of the configuration classes of the library
        when created with the :func:`~transformers.AutoConfig.from_pretrained` class method.

        The :func:`~transformers.AutoConfig.from_pretrained` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.
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
            "Unrecognized model identifier: {}. Should contain one of {}".format(
                model_type, ", ".join(CONFIG_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiates one of the configuration classes of the library
        from a pre-trained model configuration.

        The configuration class to instantiate is selected
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

            - `t5`: :class:`~transformers.T5Config` (T5 model)
            - `distilbert`: :class:`~transformers.DistilBertConfig` (DistilBERT model)
            - `albert`: :class:`~transformers.AlbertConfig` (ALBERT model)
            - `camembert`: :class:`~transformers.CamembertConfig` (CamemBERT model)
            - `xlm-roberta`: :class:`~transformers.XLMRobertaConfig` (XLM-RoBERTa model)
            - `longformer`: :class:`~transformers.LongformerConfig` (Longformer model)
            - `roberta`: :class:`~transformers.RobertaConfig` (RoBERTa model)
            - `reformer`: :class:`~transformers.ReformerConfig` (Reformer model)
            - `bert`: :class:`~transformers.BertConfig` (Bert model)
            - `openai-gpt`: :class:`~transformers.OpenAIGPTConfig` (OpenAI GPT model)
            - `gpt2`: :class:`~transformers.GPT2Config` (OpenAI GPT-2 model)
            - `transfo-xl`: :class:`~transformers.TransfoXLConfig` (Transformer-XL model)
            - `xlnet`: :class:`~transformers.XLNetConfig` (XLNet model)
            - `xlm`: :class:`~transformers.XLMConfig` (XLM model)
            - `ctrl` : :class:`~transformers.CTRLConfig` (CTRL model)
            - `flaubert` : :class:`~transformers.FlaubertConfig` (Flaubert model)
            - `electra` : :class:`~transformers.ElectraConfig` (ELECTRA model)

        Args:
            pretrained_model_name_or_path (:obj:`string`):
                Is either: \
                    - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                    - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                    - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                    - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir (:obj:`string`, optional, defaults to `None`):
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download (:obj:`boolean`, optional, defaults to `False`):
                Force to (re-)download the model weights and configuration files and override the cached versions if they exist.

            resume_download (:obj:`boolean`, optional, defaults to `False`):
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.

            proxies (:obj:`Dict[str, str]`, optional, defaults to `None`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`.
                The proxies are used on each request. See `the requests documentation <https://requests.readthedocs.io/en/master/user/advanced/#proxies>`__ for usage.

            return_unused_kwargs (:obj:`boolean`, optional, defaults to `False`):
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

            kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`): key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.


        Examples::

            config = AutoConfig.from_pretrained('bert-base-uncased')  # Download configuration from S3 and cache.
            config = AutoConfig.from_pretrained('./test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')
            config = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in pretrained_model_name_or_path:
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            "Unrecognized model in {}. "
            "Should have a `model_type` key in its config.json, or contain one of the following strings "
            "in its name: {}".format(pretrained_model_name_or_path, ", ".join(CONFIG_MAPPING.keys()))
        )
