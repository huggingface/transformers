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

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .configuration_bert import BertConfig
from .configuration_openai import OpenAIGPTConfig
from .configuration_gpt2 import GPT2Config
from .configuration_transfo_xl import TransfoXLConfig
from .configuration_xlnet import XLNetConfig
from .configuration_xlm import XLMConfig
from .configuration_roberta import RobertaConfig
from .configuration_distilbert import DistilBertConfig
from .configuration_ctrl import CTRLConfig
from .configuration_camembert import CamembertConfig
from .configuration_albert import AlbertConfig

logger = logging.getLogger(__name__)


class AutoConfig(object):
    r""":class:`~transformers.AutoConfig` is a generic configuration class
        that will be instantiated as one of the configuration classes of the library
        when created with the `AutoConfig.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertConfig (DistilBERT model)
            - contains `albert`: AlbertConfig (ALBERT model)
            - contains `camembert`: CamembertConfig (CamemBERT model)
            - contains `roberta`: RobertaConfig (RoBERTa model)
            - contains `bert`: BertConfig (Bert model)
            - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
            - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
            - contains `xlnet`: XLNetConfig (XLNet model)
            - contains `xlm`: XLMConfig (XLM model)
            - contains `ctrl` : CTRLConfig (CTRL model)
        This class cannot be instantiated using `__init__()` (throw an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a one of the configuration classes of the library
        from a pre-trained model configuration.

        The configuration class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertConfig (DistilBERT model)
            - contains `albert`: AlbertConfig (ALBERT model)
            - contains `camembert`: CamembertConfig (CamemBERT model)
            - contains `roberta`: RobertaConfig (RoBERTa model)
            - contains `bert`: BertConfig (Bert model)
            - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
            - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
            - contains `xlnet`: XLNetConfig (XLNet model)
            - contains `xlm`: XLMConfig (XLM model)
            - contains `ctrl` : CTRLConfig (CTRL model)
        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

        Examples::

            config = AutoConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = AutoConfig.from_pretrained('./test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')
            config = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return AlbertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return OpenAIGPTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return GPT2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TransfoXLConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return CTRLConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta', 'distilbert', 'camembert', 'ctrl', 'albert'".format(pretrained_model_name_or_path))
