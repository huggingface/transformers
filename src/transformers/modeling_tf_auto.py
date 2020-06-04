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


import logging
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BertConfig,
    CTRLConfig,
    DistilBertConfig,
    GPT2Config,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLNetConfig,
)
from .configuration_utils import PretrainedConfig
from .modeling_tf_albert import (
    TFAlbertForMaskedLM,
    TFAlbertForMultipleChoice,
    TFAlbertForPreTraining,
    TFAlbertForQuestionAnswering,
    TFAlbertForSequenceClassification,
    TFAlbertModel,
)
from .modeling_tf_bert import (
    TFBertForMaskedLM,
    TFBertForMultipleChoice,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFBertForTokenClassification,
    TFBertModel,
)
from .modeling_tf_ctrl import TFCTRLLMHeadModel, TFCTRLModel
from .modeling_tf_distilbert import (
    TFDistilBertForMaskedLM,
    TFDistilBertForQuestionAnswering,
    TFDistilBertForSequenceClassification,
    TFDistilBertForTokenClassification,
    TFDistilBertModel,
)
from .modeling_tf_gpt2 import TFGPT2LMHeadModel, TFGPT2Model
from .modeling_tf_openai import TFOpenAIGPTLMHeadModel, TFOpenAIGPTModel
from .modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from .modeling_tf_t5 import TFT5ForConditionalGeneration, TFT5Model
from .modeling_tf_transfo_xl import TFTransfoXLLMHeadModel, TFTransfoXLModel
from .modeling_tf_xlm import (
    TFXLMForQuestionAnsweringSimple,
    TFXLMForSequenceClassification,
    TFXLMModel,
    TFXLMWithLMHeadModel,
)
from .modeling_tf_xlnet import (
    TFXLNetForQuestionAnsweringSimple,
    TFXLNetForSequenceClassification,
    TFXLNetForTokenClassification,
    TFXLNetLMHeadModel,
    TFXLNetModel,
)


logger = logging.getLogger(__name__)


TF_MODEL_MAPPING = OrderedDict(
    [
        (T5Config, TFT5Model),
        (DistilBertConfig, TFDistilBertModel),
        (AlbertConfig, TFAlbertModel),
        (RobertaConfig, TFRobertaModel),
        (BertConfig, TFBertModel),
        (OpenAIGPTConfig, TFOpenAIGPTModel),
        (GPT2Config, TFGPT2Model),
        (TransfoXLConfig, TFTransfoXLModel),
        (XLNetConfig, TFXLNetModel),
        (XLMConfig, TFXLMModel),
        (CTRLConfig, TFCTRLModel),
    ]
)

TF_MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForPreTraining),
        (RobertaConfig, TFRobertaForMaskedLM),
        (BertConfig, TFBertForPreTraining),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
    ]
)

TF_MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (BertConfig, TFBertForMaskedLM),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
    ]
)

TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForSequenceClassification),
        (AlbertConfig, TFAlbertForSequenceClassification),
        (RobertaConfig, TFRobertaForSequenceClassification),
        (BertConfig, TFBertForSequenceClassification),
        (XLNetConfig, TFXLNetForSequenceClassification),
        (XLMConfig, TFXLMForSequenceClassification),
    ]
)

TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [(BertConfig, TFBertForMultipleChoice), (AlbertConfig, TFAlbertForMultipleChoice)]
)

TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForQuestionAnswering),
        (AlbertConfig, TFAlbertForQuestionAnswering),
        (RobertaConfig, TFRobertaForQuestionAnswering),
        (BertConfig, TFBertForQuestionAnswering),
        (XLNetConfig, TFXLNetForQuestionAnsweringSimple),
        (XLMConfig, TFXLMForQuestionAnsweringSimple),
    ]
)

TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForTokenClassification),
        (RobertaConfig, TFRobertaForTokenClassification),
        (BertConfig, TFBertForTokenClassification),
        (XLNetConfig, TFXLNetForTokenClassification),
    ]
)


class TFAutoModel(object):
    r"""
        :class:`~transformers.TFAutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: TFT5Model (T5 model)
            - `distilbert`: TFDistilBertModel (DistilBERT model)
            - `roberta`: TFRobertaModel (RoBERTa model)
            - `bert`: TFBertModel (Bert model)
            - `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
            - `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
            - `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
            - `xlnet`: TFXLNetModel (XLNet model)
            - `xlm`: TFXLMModel (XLM model)
            - `ctrl`: TFCTRLModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModel is designed to be instantiated "
            "using the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModel.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: TFDistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: TFRobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: TFBertModel (Bert model)
                    - isInstance of `openai-gpt` configuration class: TFOpenAIGPTModel (OpenAI GPT model)
                    - isInstance of `gpt2` configuration class: TFGPT2Model (OpenAI GPT-2 model)
                    - isInstance of `ctrl` configuration class: TFCTRLModel (Salesforce CTRL  model)
                    - isInstance of `transfo-xl` configuration class: TFTransfoXLModel (Transformer-XL model)
                    - isInstance of `xlnet` configuration class: TFXLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: TFXLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModel.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: TFT5Model (T5 model)
            - `distilbert`: TFDistilBertModel (DistilBERT model)
            - `roberta`: TFRobertaModel (RoBERTa model)
            - `bert`: TFTFBertModel (Bert model)
            - `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
            - `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
            - `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
            - `xlnet`: TFXLNetModel (XLNet model)
            - `ctrl`: TFCTRLModel (CTRL model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

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

            model = TFAutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModel.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_MAPPING.keys())
            )
        )


class TFAutoModelForPreTraining(object):
    r"""
        :class:`~transformers.TFAutoModelForPreTraining` is a generic model class
        that will be instantiated as one of the model classes of the library -with the architecture used for pretraining this model– when created with the `TFAutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForPreTraining is designed to be instantiated "
            "using the `TFAutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForPreTraining.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers.TFDistilBertModelForMaskedLM` (DistilBERT model)
                - isInstance of `roberta` configuration class: :class:`~transformers.TFRobertaModelForMaskedLM` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformers.TFBertForPreTraining` (Bert model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformers.TFOpenAIGPTLMHeadModel` (OpenAI GPT model)
                - isInstance of `gpt2` configuration class: :class:`~transformers.TFGPT2ModelLMHeadModel` (OpenAI GPT-2 model)
                - isInstance of `ctrl` configuration class: :class:`~transformers.TFCTRLModelLMHeadModel` (Salesforce CTRL  model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformers.TFTransfoXLLMHeadModel` (Transformer-XL model)
                - isInstance of `xlnet` configuration class: :class:`~transformers.TFXLNetLMHeadModel` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformers.TFXLMWithLMHeadModel` (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForPreTraining.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the model classes of the library -with the architecture used for pretraining this model– from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: :class:`~transformers.TFT5ModelWithLMHead` (T5 model)
            - `distilbert`: :class:`~transformers.TFDistilBertForMaskedLM` (DistilBERT model)
            - `albert`: :class:`~transformers.TFAlbertForPreTraining` (ALBERT model)
            - `roberta`: :class:`~transformers.TFRobertaForMaskedLM` (RoBERTa model)
            - `bert`: :class:`~transformers.TFBertForPreTraining` (Bert model)
            - `openai-gpt`: :class:`~transformers.TFOpenAIGPTLMHeadModel` (OpenAI GPT model)
            - `gpt2`: :class:`~transformers.TFGPT2LMHeadModel` (OpenAI GPT-2 model)
            - `transfo-xl`: :class:`~transformers.TFTransfoXLLMHeadModel` (Transformer-XL model)
            - `xlnet`: :class:`~transformers.TFXLNetLMHeadModel` (XLNet model)
            - `xlm`: :class:`~transformers.TFXLMWithLMHeadModel` (XLM model)
            - `ctrl`: :class:`~transformers.TFCTRLLMHeadModel` (Salesforce CTRL model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
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
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model.
                (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or
                automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                  underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                  already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                  initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                  with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                  attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForPreTraining.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForPreTraining.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForPreTraining.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForPreTraining.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )


class TFAutoModelWithLMHead(object):
    r"""
        :class:`~transformers.TFAutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: TFT5ForConditionalGeneration (T5 model)
            - `distilbert`: TFDistilBertForMaskedLM (DistilBERT model)
            - `roberta`: TFRobertaForMaskedLM (RoBERTa model)
            - `bert`: TFBertForMaskedLM (Bert model)
            - `openai-gpt`: TFOpenAIGPTLMHeadModel (OpenAI GPT model)
            - `gpt2`: TFGPT2LMHeadModel (OpenAI GPT-2 model)
            - `transfo-xl`: TFTransfoXLLMHeadModel (Transformer-XL model)
            - `xlnet`: TFXLNetLMHeadModel (XLNet model)
            - `xlm`: TFXLMWithLMHeadModel (XLM model)
            - `ctrl`: TFCTRLLMHeadModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelWithLMHead is designed to be instantiated "
            "using the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelWithLMHead.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `openai-gpt` configuration class: OpenAIGPTModel (OpenAI GPT model)
                    - isInstance of `gpt2` configuration class: GPT2Model (OpenAI GPT-2 model)
                    - isInstance of `ctrl` configuration class: CTRLModel (Salesforce CTRL  model)
                    - isInstance of `transfo-xl` configuration class: TransfoXLModel (Transformer-XL model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelWithLMHead.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `t5`: TFT5ForConditionalGeneration (T5 model)
            - `distilbert`: TFDistilBertForMaskedLM (DistilBERT model)
            - `roberta`: TFRobertaForMaskedLM (RoBERTa model)
            - `bert`: TFBertForMaskedLM (Bert model)
            - `openai-gpt`: TFOpenAIGPTLMHeadModel (OpenAI GPT model)
            - `gpt2`: TFGPT2LMHeadModel (OpenAI GPT-2 model)
            - `transfo-xl`: TFTransfoXLLMHeadModel (Transformer-XL model)
            - `xlnet`: TFXLNetLMHeadModel (XLNet model)
            - `xlm`: TFXLMWithLMHeadModel (XLM model)
            - `ctrl`: TFCTRLLMHeadModel (CTRL model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

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

            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelWithLMHead.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )


class TFAutoModelForMultipleChoice:
    r"""
        :class:`~transformers.TFAutoModelForMultipleChoice` is a generic model class
        that will be instantiated as one of the multiple choice model classes of the library
        when created with the `TFAutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `albert`: TFAlbertForMultipleChoice (Albert model)
            - `bert`: TFBertForMultipleChoice (Bert model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForMultipleChoice is designed to be instantiated "
            "using the `TFAutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForMultipleChoice.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `albert` configuration class: AlbertModel (Albert model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForMulitpleChoice.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the multiple choice model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `albert`: TFRobertaForMultiple (Albert model)
            - `bert`: TFBertForMultipleChoice (Bert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

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

            model = TFAutoModelFormultipleChoice.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelFormultipleChoice.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelFormultipleChoice.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelFormultipleChoice.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING.keys()),
            )
        )


class TFAutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers.TFAutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - `bert`: TFBertForSequenceClassification (Bert model)
            - `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - `xlm`: TFXLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForSequenceClassification is designed to be instantiated "
            "using the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - `bert`: TFBertForSequenceClassification (Bert model)
            - `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - `xlm`: TFXLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

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

            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForSequenceClassification.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )


class TFAutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers.TFAutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: TFDistilBertForQuestionAnswering (DistilBERT model)
            - `albert`: TFAlbertForQuestionAnswering (ALBERT model)
            - `roberta`: TFRobertaForQuestionAnswering (RoBERTa model)
            - `bert`: TFBertForQuestionAnswering (Bert model)
            - `xlnet`: TFXLNetForQuestionAnswering (XLNet model)
            - `xlm`: TFXLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForQuestionAnswering is designed to be instantiated "
            "using the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForQuestionAnswering.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `albert` configuration class: AlbertModel (ALBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForQuestionAnswering.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `distilbert`: TFDistilBertForQuestionAnswering (DistilBERT model)
            - `albert`: TFAlbertForQuestionAnswering (ALBERT model)
            - `roberta`: TFRobertaForQuestionAnswering (RoBERTa model)
            - `bert`: TFBertForQuestionAnswering (Bert model)
            - `xlnet`: TFXLNetForQuestionAnswering (XLNet model)
            - `xlm`: TFXLMForQuestionAnswering (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

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

            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForQuestionAnswering.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )


class TFAutoModelForTokenClassification:
    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForTokenClassification is designed to be instantiated "
            "using the `TFAutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBert model)
                    - isInstance of `roberta` configuration class: RobteraModel (Roberta model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForTokenClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:
            - `bert`: BertForTokenClassification (Bert model)
            - `xlnet`: XLNetForTokenClassification (XLNet model)
            - `distilbert`: DistilBertForTokenClassification (DistilBert model)
            - `roberta`: RobertaForTokenClassification (Roberta model)

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

            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )
