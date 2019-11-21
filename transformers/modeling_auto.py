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

from .modeling_bert import BertModel, BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering
from .modeling_openai import OpenAIGPTModel, OpenAIGPTLMHeadModel
from .modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from .modeling_ctrl import CTRLModel, CTRLLMHeadModel
from .modeling_transfo_xl import TransfoXLModel, TransfoXLLMHeadModel
from .modeling_xlnet import XLNetModel, XLNetLMHeadModel, XLNetForSequenceClassification, XLNetForQuestionAnswering
from .modeling_xlm import XLMModel, XLMWithLMHeadModel, XLMForSequenceClassification, XLMForQuestionAnswering
from .modeling_roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification
from .modeling_distilbert import DistilBertModel, DistilBertForQuestionAnswering, DistilBertForMaskedLM, DistilBertForSequenceClassification
from .modeling_camembert import CamembertModel, CamembertForMaskedLM, CamembertForSequenceClassification, CamembertForMultipleChoice

from .modeling_utils import PreTrainedModel, SequenceSummary

from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)


class AutoModel(object):
    r"""
        :class:`~transformers.AutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `camembert`: CamembertModel (CamemBERT model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `camembert`: CamembertModel (CamemBERT model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
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
        elif 'ctrl' in pretrained_model_name_or_path:
            return CTRLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta, 'ctrl'".format(pretrained_model_name_or_path))


class AutoModelWithLMHead(object):
    r"""
        :class:`~transformers.AutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForMaskedLM (DistilBERT model)
            - contains `camembert`: CamembertForMaskedLM (CamemBERT model)
            - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
            - contains `bert`: BertForMaskedLM (Bert model)
            - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `ctrl`: CTRLLMModel (Salesforce CTRL model)
            - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: XLNetLMHeadModel (XLNet model)
            - contains `xlm`: XLMWithLMHeadModel (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForMaskedLM (DistilBERT model)
            - contains `camembert`: CamembertForMaskedLM (CamemBERT model)
            - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
            - contains `bert`: BertForMaskedLM (Bert model)
            - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: XLNetLMHeadModel (XLNet model)
            - contains `xlm`: XLMWithLMHeadModel (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return OpenAIGPTLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TransfoXLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMWithLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return CTRLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta','ctrl'".format(pretrained_model_name_or_path))


class AutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers.AutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForSequenceClassification (DistilBERT model)
            - contains `camembert`: CamembertForSequenceClassification (CamemBERT model)
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForSequenceClassification (DistilBERT model)
            - contains `camembert`: CamembertForSequenceClassification (CamemBERT model)
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'xlm', 'roberta'".format(pretrained_model_name_or_path))


class AutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers.AutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: BertForQuestionAnswering (Bert model)
            - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: XLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` method.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: BertForQuestionAnswering (Bert model)
            - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: XLMForQuestionAnswering (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'xlm'".format(pretrained_model_name_or_path))
