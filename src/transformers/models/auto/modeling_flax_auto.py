# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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


from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..bert.modeling_flax_bert import FlaxBertModel
from ..roberta.modeling_flax_roberta import FlaxRobertaModel
from .configuration_auto import AutoConfig, BertConfig, RobertaConfig


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING = OrderedDict(
    [
        (RobertaConfig, FlaxRobertaModel),
        (BertConfig, FlaxBertModel),
    ]
)


class FlaxAutoModel(object):
    r"""
    :class:`~transformers.FlaxAutoModel` is a generic model class that will be instantiated as one of the base model
    classes of the library when created with the `FlaxAutoModel.from_pretrained(pretrained_model_name_or_path)` or the
    `FlaxAutoModel.from_config(config)` class methods.

    This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "FlaxAutoModel is designed to be instantiated "
            "using the `FlaxAutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`FlaxAutoModel.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiates one of the base model classes of the library from a configuration.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `roberta` configuration class: :class:`~transformers.FlaxRobertaModel` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformers.FlaxBertModel` (Bert model

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')
            # Download configuration from huggingface.co and cache.
            model = FlaxAutoModel.from_config(config)
            # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in FLAX_MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} "
            f"for this kind of FlaxAutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in FLAX_MODEL_MAPPING.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates one of the base model classes of the library from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance based on the
        `model_type` property of the config object, or when it's missing, falling back to using pattern matching on the
        `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching in the
        `pretrained_model_name_or_path` string (in the following order):

            - contains `roberta`: :class:`~transformers.FlaxRobertaModel` (RoBERTa model)
            - contains `bert`: :class:`~transformers.FlaxBertModel` (Bert model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated) To
            train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co. Valid
                  model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under a user or
                  organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using
                  :func:`~transformers.FlaxPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `pytorch index checkpoint file` (e.g. `./pt_model/pytorch_model.bin`). In this
                  case, ``from_pt`` should be set to True and a configuration object should be provided as ``config``
                  argument.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a
                  pretrained model), or
                - the model was saved using :func:`~transformers.FlaxPreTrainedModel.save_pretrained` and is reloaded
                  by supplying the save directory.
                - the model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                  configuration JSON file named `config.json` is found in the directory.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model configuration should be cached if the
                standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if
                they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}. The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionary containing missing keys, unexpected keys and error
                messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = FlaxAutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from huggingface.co and cache.
            model = FlaxAutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in FLAX_MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} "
            f"for this kind of FlaxAutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in FLAX_MODEL_MAPPING.keys())}"
        )
