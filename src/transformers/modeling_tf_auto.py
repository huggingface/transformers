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


import warnings
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    ElectraConfig,
    FlaubertConfig,
    FunnelConfig,
    GPT2Config,
    LongformerConfig,
    MobileBertConfig,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)
from .configuration_utils import PretrainedConfig
from .modeling_tf_albert import (
    TFAlbertForMaskedLM,
    TFAlbertForMultipleChoice,
    TFAlbertForPreTraining,
    TFAlbertForQuestionAnswering,
    TFAlbertForSequenceClassification,
    TFAlbertForTokenClassification,
    TFAlbertModel,
)
from .modeling_tf_bert import (
    TFBertForMaskedLM,
    TFBertForMultipleChoice,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFBertForTokenClassification,
    TFBertLMHeadModel,
    TFBertModel,
)
from .modeling_tf_camembert import (
    TFCamembertForMaskedLM,
    TFCamembertForMultipleChoice,
    TFCamembertForQuestionAnswering,
    TFCamembertForSequenceClassification,
    TFCamembertForTokenClassification,
    TFCamembertModel,
)
from .modeling_tf_ctrl import TFCTRLLMHeadModel, TFCTRLModel
from .modeling_tf_distilbert import (
    TFDistilBertForMaskedLM,
    TFDistilBertForMultipleChoice,
    TFDistilBertForQuestionAnswering,
    TFDistilBertForSequenceClassification,
    TFDistilBertForTokenClassification,
    TFDistilBertModel,
)
from .modeling_tf_electra import (
    TFElectraForMaskedLM,
    TFElectraForMultipleChoice,
    TFElectraForPreTraining,
    TFElectraForQuestionAnswering,
    TFElectraForSequenceClassification,
    TFElectraForTokenClassification,
    TFElectraModel,
)
from .modeling_tf_flaubert import (
    TFFlaubertForMultipleChoice,
    TFFlaubertForQuestionAnsweringSimple,
    TFFlaubertForSequenceClassification,
    TFFlaubertForTokenClassification,
    TFFlaubertModel,
    TFFlaubertWithLMHeadModel,
)
from .modeling_tf_funnel import (
    TFFunnelForMaskedLM,
    TFFunnelForMultipleChoice,
    TFFunnelForPreTraining,
    TFFunnelForQuestionAnswering,
    TFFunnelForSequenceClassification,
    TFFunnelForTokenClassification,
    TFFunnelModel,
)
from .modeling_tf_gpt2 import TFGPT2LMHeadModel, TFGPT2Model
from .modeling_tf_longformer import TFLongformerForMaskedLM, TFLongformerForQuestionAnswering, TFLongformerModel
from .modeling_tf_mobilebert import (
    TFMobileBertForMaskedLM,
    TFMobileBertForMultipleChoice,
    TFMobileBertForPreTraining,
    TFMobileBertForQuestionAnswering,
    TFMobileBertForSequenceClassification,
    TFMobileBertForTokenClassification,
    TFMobileBertModel,
)
from .modeling_tf_openai import TFOpenAIGPTLMHeadModel, TFOpenAIGPTModel
from .modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForMultipleChoice,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from .modeling_tf_t5 import TFT5ForConditionalGeneration, TFT5Model
from .modeling_tf_transfo_xl import TFTransfoXLLMHeadModel, TFTransfoXLModel
from .modeling_tf_xlm import (
    TFXLMForMultipleChoice,
    TFXLMForQuestionAnsweringSimple,
    TFXLMForSequenceClassification,
    TFXLMForTokenClassification,
    TFXLMModel,
    TFXLMWithLMHeadModel,
)
from .modeling_tf_xlm_roberta import (
    TFXLMRobertaForMaskedLM,
    TFXLMRobertaForMultipleChoice,
    TFXLMRobertaForQuestionAnswering,
    TFXLMRobertaForSequenceClassification,
    TFXLMRobertaForTokenClassification,
    TFXLMRobertaModel,
)
from .modeling_tf_xlnet import (
    TFXLNetForMultipleChoice,
    TFXLNetForQuestionAnsweringSimple,
    TFXLNetForSequenceClassification,
    TFXLNetForTokenClassification,
    TFXLNetLMHeadModel,
    TFXLNetModel,
)
from .utils import logging


logger = logging.get_logger(__name__)


TF_MODEL_MAPPING = OrderedDict(
    [
        (T5Config, TFT5Model),
        (DistilBertConfig, TFDistilBertModel),
        (AlbertConfig, TFAlbertModel),
        (CamembertConfig, TFCamembertModel),
        (XLMRobertaConfig, TFXLMRobertaModel),
        (LongformerConfig, TFLongformerModel),
        (RobertaConfig, TFRobertaModel),
        (BertConfig, TFBertModel),
        (OpenAIGPTConfig, TFOpenAIGPTModel),
        (GPT2Config, TFGPT2Model),
        (MobileBertConfig, TFMobileBertModel),
        (TransfoXLConfig, TFTransfoXLModel),
        (XLNetConfig, TFXLNetModel),
        (FlaubertConfig, TFFlaubertModel),
        (XLMConfig, TFXLMModel),
        (CTRLConfig, TFCTRLModel),
        (ElectraConfig, TFElectraModel),
        (FunnelConfig, TFFunnelModel),
    ]
)

TF_MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForPreTraining),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (BertConfig, TFBertForPreTraining),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (MobileBertConfig, TFMobileBertForPreTraining),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
        (ElectraConfig, TFElectraForPreTraining),
        (FunnelConfig, TFFunnelForPreTraining),
    ]
)

TF_MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        (T5Config, TFT5ForConditionalGeneration),
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForMaskedLM),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (LongformerConfig, TFLongformerForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (BertConfig, TFBertForMaskedLM),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (MobileBertConfig, TFMobileBertForMaskedLM),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (CTRLConfig, TFCTRLLMHeadModel),
        (ElectraConfig, TFElectraForMaskedLM),
        (FunnelConfig, TFFunnelForMaskedLM),
    ]
)

TF_MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        (BertConfig, TFBertLMHeadModel),
        (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel),
        (GPT2Config, TFGPT2LMHeadModel),
        (TransfoXLConfig, TFTransfoXLLMHeadModel),
        (XLNetConfig, TFXLNetLMHeadModel),
        (
            XLMConfig,
            TFXLMWithLMHeadModel,
        ),  # XLM can be MLM and CLM => model should be split similar to BERT; leave here for now
        (CTRLConfig, TFCTRLLMHeadModel),
    ]
)

TF_MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForMaskedLM),
        (AlbertConfig, TFAlbertForMaskedLM),
        (CamembertConfig, TFCamembertForMaskedLM),
        (XLMRobertaConfig, TFXLMRobertaForMaskedLM),
        (LongformerConfig, TFLongformerForMaskedLM),
        (RobertaConfig, TFRobertaForMaskedLM),
        (BertConfig, TFBertForMaskedLM),
        (MobileBertConfig, TFMobileBertForMaskedLM),
        (FlaubertConfig, TFFlaubertWithLMHeadModel),
        (XLMConfig, TFXLMWithLMHeadModel),
        (ElectraConfig, TFElectraForMaskedLM),
        (FunnelConfig, TFFunnelForMaskedLM),
    ]
)

TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict([(T5Config, TFT5ForConditionalGeneration)])

TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForSequenceClassification),
        (AlbertConfig, TFAlbertForSequenceClassification),
        (CamembertConfig, TFCamembertForSequenceClassification),
        (XLMRobertaConfig, TFXLMRobertaForSequenceClassification),
        (RobertaConfig, TFRobertaForSequenceClassification),
        (BertConfig, TFBertForSequenceClassification),
        (XLNetConfig, TFXLNetForSequenceClassification),
        (MobileBertConfig, TFMobileBertForSequenceClassification),
        (FlaubertConfig, TFFlaubertForSequenceClassification),
        (XLMConfig, TFXLMForSequenceClassification),
        (ElectraConfig, TFElectraForSequenceClassification),
        (FunnelConfig, TFFunnelForSequenceClassification),
    ]
)

TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForQuestionAnswering),
        (AlbertConfig, TFAlbertForQuestionAnswering),
        (CamembertConfig, TFCamembertForQuestionAnswering),
        (XLMRobertaConfig, TFXLMRobertaForQuestionAnswering),
        (LongformerConfig, TFLongformerForQuestionAnswering),
        (RobertaConfig, TFRobertaForQuestionAnswering),
        (BertConfig, TFBertForQuestionAnswering),
        (XLNetConfig, TFXLNetForQuestionAnsweringSimple),
        (MobileBertConfig, TFMobileBertForQuestionAnswering),
        (FlaubertConfig, TFFlaubertForQuestionAnsweringSimple),
        (XLMConfig, TFXLMForQuestionAnsweringSimple),
        (ElectraConfig, TFElectraForQuestionAnswering),
        (FunnelConfig, TFFunnelForQuestionAnswering),
    ]
)

TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, TFDistilBertForTokenClassification),
        (AlbertConfig, TFAlbertForTokenClassification),
        (CamembertConfig, TFCamembertForTokenClassification),
        (FlaubertConfig, TFFlaubertForTokenClassification),
        (XLMConfig, TFXLMForTokenClassification),
        (XLMRobertaConfig, TFXLMRobertaForTokenClassification),
        (RobertaConfig, TFRobertaForTokenClassification),
        (BertConfig, TFBertForTokenClassification),
        (MobileBertConfig, TFMobileBertForTokenClassification),
        (XLNetConfig, TFXLNetForTokenClassification),
        (ElectraConfig, TFElectraForTokenClassification),
        (FunnelConfig, TFFunnelForTokenClassification),
    ]
)

TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        (CamembertConfig, TFCamembertForMultipleChoice),
        (XLMConfig, TFXLMForMultipleChoice),
        (XLMRobertaConfig, TFXLMRobertaForMultipleChoice),
        (RobertaConfig, TFRobertaForMultipleChoice),
        (BertConfig, TFBertForMultipleChoice),
        (DistilBertConfig, TFDistilBertForMultipleChoice),
        (MobileBertConfig, TFMobileBertForMultipleChoice),
        (XLNetConfig, TFXLNetForMultipleChoice),
        (FlaubertConfig, TFFlaubertForMultipleChoice),
        (AlbertConfig, TFAlbertForMultipleChoice),
        (ElectraConfig, TFElectraForMultipleChoice),
        (FunnelConfig, TFFunnelForMultipleChoice),
    ]
)


class TFAutoModel(object):
    r"""
    :class:`~transformers.TFAutoModel` is a generic model class
    that will be instantiated as one of the base model classes of the library
    when created with the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)`
    class method.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModel is designed to be instantiated "
            "using the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModel.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

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
    @replace_list_option_in_docstrings(TF_MODEL_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModel.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModel.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

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
    @replace_list_option_in_docstrings(TF_MODEL_FOR_PRETRAINING_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.TFPretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForPreTraining.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_PRETRAINING_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the model classes of the library -with the architecture used for pretraining this model– from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.
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
                (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or
                automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                  underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                  already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                  initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                  with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                  attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForPreTraining.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForPreTraining.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForPreTraining.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForPreTraining.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in TF_MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
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

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelWithLMHead is designed to be instantiated "
            "using the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelWithLMHead.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_WITH_LM_HEAD_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelWithLMHead.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
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
    @replace_list_option_in_docstrings(TF_MODEL_WITH_LM_HEAD_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelWithLMHead.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        config = kwargs.pop("config", None)

        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in TF_MODEL_WITH_LM_HEAD_MAPPING.items():
            # Not using isinstance() here to do not take into account inheritance
            if config_class == type(config):
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

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForMultipleChoice is designed to be instantiated "
            "using the `TFAutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForMultipleChoice.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForMulitpleChoice.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
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
    @replace_list_option_in_docstrings(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the multiple choice model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelFormultipleChoice.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelFormultipleChoice.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelFormultipleChoice.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelFormultipleChoice.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

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


class TFAutoModelForCausalLM:
    r"""
    :class:`~transformers.TFAutoModelForCausalLM` is a generic model class
    that will be instantiated as one of the language modeling model classes of the library
    when created with the `TFAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)`
    class method.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForCausalLM is designed to be instantiated "
            "using the `TFAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForCausalLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.TFPretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = GPT2Config.from_pretrained('gpt2')    # Download configuration from S3 and cache.
            model = TFAutoModelForCausalLM.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_CAUSAL_LM_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.
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
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = TFAutoModelForCausalLM.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForCausalLM.from_pretrained('./test/gpt2_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForCausalLM.from_pretrained('gpt2', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/gpt2_tf_model_config.json')
            model =  TFAutoModelForCausalLM.from_pretrained('./tf_model/gpt2_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in TF_MODEL_FOR_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )


class TFAutoModelForMaskedLM:
    r"""
    :class:`~transformers.TFAutoModelForMaskedLM` is a generic model class
    that will be instantiated as one of the language modeling model classes of the library
    when created with the `TFAutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)`
    class method.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForMaskedLM is designed to be instantiated "
            "using the `TFAutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForMaskedLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_MASKED_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.TFPretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForMaskedLM.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_MASKED_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_MASKED_LM_MAPPING.keys())
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_MASKED_LM_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.
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
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = TFAutoModelForMaskedLM.from_pretrained(('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForMaskedLM.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForMaskedLM.from_pretrained(('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model =  TFAutoModelForMaskedLM.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in TF_MODEL_FOR_MASKED_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in TF_MODEL_FOR_MASKED_LM_MAPPING.keys())
            )
        )


class TFAutoModelForSeq2SeqLM:
    r"""
    :class:`~transformers.TFAutoModelForSeq2SeqLM` is a generic model class
    that will be instantiated as one of the language modeling model classes of the library
    when created with the `TFAutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)`
    class method.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForSeq2SeqLM is designed to be instantiated "
            "using the `TFAutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForSeq2SeqLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config (:class:`~transformers.TFPretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = T5Config.from_pretrained('t5')
            model = TFAutoModelForSeq2SeqLM.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()),
            )
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.
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
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = TFAutoModelForSeq2SeqLM.from_pretrained('t5-base')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForSeq2SeqLM.from_pretrained('./test/t5_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForSeq2SeqLM.from_pretrained('t5-base', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/t5_tf_model_config.json')
            model =  TFAutoModelForSeq2SeqLM.from_pretrained('./tf_model/t5_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        for config_class, model_class in TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys()),
            )
        )


class TFAutoModelForSequenceClassification(object):
    r"""
    :class:`~transformers.TFAutoModelForSequenceClassification` is a generic model class
    that will be instantiated as one of the sequence classification model classes of the library
    when created with the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
    class method.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForSequenceClassification is designed to be instantiated "
            "using the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
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
    @replace_list_option_in_docstrings(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForSequenceClassification.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

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

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForQuestionAnswering is designed to be instantiated "
            "using the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForQuestionAnswering.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

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
    @replace_list_option_in_docstrings(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForQuestionAnswering.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

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
            "`TFAutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                List options

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
    @replace_list_option_in_docstrings(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        List options

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

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
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

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
