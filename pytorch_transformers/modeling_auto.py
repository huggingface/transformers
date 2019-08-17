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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.parameter import Parameter

from .modeling_bert import BertConfig, BertModel
from .modeling_openai import OpenAIGPTConfig, OpenAIGPTModel
from .modeling_gpt2 import GPT2Config, GPT2Model
from .modeling_transfo_xl import TransfoXLConfig, TransfoXLModel
from .modeling_xlnet import XLNetConfig, XLNetModel
from .modeling_xlm import XLMConfig, XLMModel
from .modeling_roberta import RobertaConfig, RobertaModel

from .modeling_utils import PreTrainedModel, SequenceSummary

logger = logging.getLogger(__name__)

class AutoConfig(object):
    r""":class:`~pytorch_transformers.AutoConfig` is a generic configuration class
        that will be instantiated as one of the configuration classes of the library
        when created with the `AutoConfig.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `bert`: BertConfig (Bert model)
            - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
            - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
            - contains `xlnet`: XLNetConfig (XLNet model)
            - contains `xlm`: XLMConfig (XLM model)
            - contains `roberta`: RobertaConfig (RoBERTa model)

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
            - contains `bert`: BertConfig (Bert model)
            - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
            - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
            - contains `xlnet`: XLNetConfig (XLNet model)
            - contains `xlm`: XLMConfig (XLM model)
            - contains `roberta`: RobertaConfig (RoBERTa model)

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load from cache
                    or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **return_unused_kwargs**: (`optional`) bool:
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes:
                ie the part of kwargs which has not been used to update `config` and is otherwise ignored.
            **kwargs**: (`optional`) dict:
                Dictionary of key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used
                to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

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
        if 'roberta' in pretrained_model_name_or_path:
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

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta'".format(pretrained_model_name_or_path))


class AutoModel(object):
    r"""
        :class:`~pytorch_transformers.AutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method take care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)
            - contains `roberta`: RobertaModel (RoBERTa model)

        This class cannot be instantiated using `__init__()` (throw an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiate a one of the base model classes of the library
        from a pre-trained model configuration.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)
            - contains `roberta`: RobertaModel (RoBERTa model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache
                    or download and cache if not already stored in cache (e.g. 'bert-base-uncased').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a tensorflow index checkpoint `file` (e.g. `./tf_model/model.ckpt.index`).
                    In this case, ``from_tf`` should be set to True and a configuration object should be
                    provided as `config` argument. This loading option is slower than converting the TensorFlow
                    checkpoint in a PyTorch model using the provided conversion scripts and loading
                    the PyTorch model afterwards.
            **model_args**: (`optional`) Sequence:
                All remaning positional arguments will be passed to the underlying model's __init__ function
            **config**: an optional configuration for the model to use instead of an automatically loaded configuation.
                Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with a `shortcut name` of a pre-trained model), or
                - the model was saved using the `save_pretrained(save_directory)` (loaded by suppling the save directory).
            **state_dict**: an optional state dictionnary for the model to use instead of a state dictionary loaded
                from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using `save_pretrained(dir)` and `from_pretrained(save_directory)` is not
                a simpler option.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **output_loading_info**: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            **kwargs**: (`optional`) dict:
                Dictionary of key, values to update the configuration object after loading.
                Can be used to override selected configuration parameters. E.g. ``output_attention=True``.

               - If a configuration is provided with `config`, **kwargs will be directly passed
                 to the underlying model's __init__ method.
               - If a configuration is not provided, **kwargs will be first passed to the pretrained
                 model configuration class loading function (`PretrainedConfig.from_pretrained`).
                 Each key of **kwargs that corresponds to a configuration attribute
                 will be used to override said attribute with the supplied **kwargs value.
                 Remaining keys that do not correspond to any configuration attribute will
                 be passed to the underlying model's __init__ function.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'roberta' in pretrained_model_name_or_path:
            return RobertaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return OpenAIGPTModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return GPT2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TransfoXLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta'".format(pretrained_model_name_or_path))
