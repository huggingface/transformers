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

import os
from abc import ABC, abstractmethod
from functools import partial
from pickle import UnpicklingError
from typing import Dict, Set, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

from .configuration_utils import PretrainedConfig
from .file_utils import FLAX_WEIGHTS_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from .utils import logging


logger = logging.get_logger(__name__)


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class FlaxPreTrainedModel(ABC):
    r"""
    Base class for all models.

    :class:`~transformers.FlaxPreTrainedModel` takes care of storing the configuration of the models and handles
    methods for loading, downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
          :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
          derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    base_model_prefix = ""

    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
    ):
        if config is None:
            raise ValueError("config cannot be None")

        if module is None:
            raise ValueError("module cannot be None")

        # Those are private to be exposed as typed property on derived classes.
        self._config = config
        self._module = module

        # Those are public as their type is generic to every derived classes.
        self.key = PRNGKey(seed)
        self.dtype = dtype

        # randomely initialized parameters
        random_params = self.init(self.key, input_shape)

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(random_params)).keys())
        self.params = random_params

    def init(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> Dict:
        raise NotImplementedError(f"init method has to be implemented for {self}")

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def params(self) -> Union[Dict, FrozenDict]:
        return self._params

    @property
    def required_params(self) -> Set:
        return self._required_params

    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        if isinstance(params, FrozenDict):
            params = unfreeze(params)
        param_keys = set(flatten_dict(params).keys())
        if len(self.required_params - param_keys) > 0:
            raise ValueError(
                "Some parameters are missing. Make sure that `params` include the following "
                f"parameters {self.required_params - param_keys}"
            )
        self._params = freeze(params)

    @staticmethod
    @abstractmethod
    def convert_from_pytorch(pt_state: Dict, config: PretrainedConfig) -> Dict:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs
    ):

        r"""
        Instantiate a pretrained flax model from a pre-trained model configuration.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.FlaxPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `pt index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In this
                      case, ``from_pt`` should be set to :obj:`True`.
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
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

            >>> from transformers import BertConfig, FlaxBertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = FlaxBertModel.from_pretrained('bert-base-cased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = FlaxBertModel.from_pretrained('./test/saved_model/')
            >>> # Loading from a PyTorch checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./pt_model/config.json')
            >>> model = FlaxBertModel.from_pretrained('./pt_model/pytorch_model.bin', from_pt=True, config=config)
        """
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Add the dtype to model_kwargs
        model_kwargs["dtype"] = dtype

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    # Load from a Flax checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_pt` set to False".format(
                            [FLAX_WEIGHTS_NAME, WEIGHTS_NAME],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME,
                    revision=revision,
                )

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # Instantiate model.
        with open(resolved_archive_file, "rb") as state_f:
            try:
                if from_pt:
                    import torch

                    state = torch.load(state_f)

                    state = convert_state_dict_from_pt(cls, state, config)
                else:
                    state = from_bytes(cls, state_f.read())
            except UnpicklingError:
                raise EnvironmentError(
                    f"Unable to convert pytorch model {archive_file} to Flax deserializable object. "
                )

        # init random models
        model = cls(config, *model_args, **model_kwargs)

        # if model is base model only use model_prefix key
        if cls.base_model_prefix not in dict(model.params) and cls.base_model_prefix in state:
            state = state[cls.base_model_prefix]

        # flatten dicts
        state = flatten_dict(state)
        random_state = flatten_dict(unfreeze(model.params))

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        # add missing keys as random parameters
        for missing_key in missing_keys:
            state[missing_key] = random_state[missing_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )

        # set correct parameters
        model.params = unflatten_dict(state)
        return model

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.FlaxPreTrainedModel.from_pretrained`` class method

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # get abs dir
        save_directory = os.path.abspath(save_directory)
        # save config as well
        self.config.save_pretrained(save_directory)

        # save model
        with open(os.path.join(save_directory, FLAX_WEIGHTS_NAME), "wb") as f:
            model_bytes = to_bytes(self.params)
            f.write(model_bytes)


def convert_state_dict_from_pt(model_class: ABC, state: Dict, config: PretrainedConfig):
    """
    Converts a PyTorch parameter state dict to an equivalent Flax parameter state dict
    """
    state = {k: v.numpy() for k, v in state.items()}
    state = model_class.convert_from_pytorch(state, config)
    state = unflatten_dict({tuple(k.split(".")): v for k, v in state.items()})
    return state
