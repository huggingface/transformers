# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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
from functools import partial
from pickle import UnpicklingError
from typing import Dict, Set, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

from .configuration_utils import PretrainedConfig
from .file_utils import (
    FLAX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    cached_path,
    copy_func,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    replace_return_docstrings,
)
from .generation_flax_utils import FlaxGenerationMixin
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import logging


logger = logging.get_logger(__name__)


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}


class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
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

        # randomly initialized parameters
        random_params = self.init_weights(self.key, input_shape)

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(random_params)).keys())
        self.params = random_params

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> Dict:
        raise NotImplementedError(f"init method has to be implemented for {self}")

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

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
        self._params = params

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
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
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
            ignore_mismatched_sizes (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
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
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "model", "framework": "flax", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

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
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
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
                        f"Error no file named {[FLAX_WEIGHTS_NAME, WEIGHTS_NAME]} found in directory "
                        f"{pretrained_model_name_or_path} or `from_pt` set to False"
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
                    user_agent=user_agent,
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

        # init random models
        model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            state = load_pytorch_checkpoint_in_flax_state_dict(model, resolved_archive_file)
        else:
            with open(resolved_archive_file, "rb") as state_f:
                try:
                    state = from_bytes(cls, state_f.read())
                except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                    try:
                        with open(resolved_archive_file) as f:
                            if f.read().startswith("version"):
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please install "
                                    "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                                    "you cloned."
                                )
                            else:
                                raise ValueError from e
                    except (UnicodeDecodeError, ValueError):
                        raise EnvironmentError(f"Unable to convert {archive_file} to Flax deserializable object. ")
            # make sure all arrays are stored as jnp.arrays
            # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
            # https://github.com/google/flax/issues/1261
            state = jax.tree_util.tree_map(jnp.array, state)

        # if model is base model only use model_prefix key
        if cls.base_model_prefix not in dict(model.params) and cls.base_model_prefix in state:
            state = state[cls.base_model_prefix]

        # if model is head model and we are loading weights from base model
        # we initialize new params dict with base_model_prefix
        if cls.base_model_prefix in dict(model.params) and cls.base_model_prefix not in state:
            state = {cls.base_model_prefix: state}

        # flatten dicts
        state = flatten_dict(state)

        random_state = flatten_dict(unfreeze(model.params))

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        for key in state.keys():
            if key in random_state and state[key].shape != random_state[key].shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append((key, state[key].shape, random_state[key].shape))
                    state[key] = random_state[key]
                else:
                    raise ValueError(
                        f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                        f"{state[key].shape} which is incompatible with the model shape {random_state[key].shape}. "
                        "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
                        "model."
                    )

        # add missing keys as random parameters
        for missing_key in missing_keys:
            state[missing_key] = random_state[missing_key]

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

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
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized because the shapes did not match:\n{mismatched_warning}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )

        # set correct parameters
        model.params = unflatten_dict(state)

        return model

    def save_pretrained(self, save_directory: Union[str, os.PathLike], params=None, push_to_hub=False, **kwargs):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.FlaxPreTrainedModel.from_pretrained`` class method

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # get abs dir
        save_directory = os.path.abspath(save_directory)
        # save config as well
        self.config.architectures = [self.__class__.__name__[4:]]
        self.config.save_pretrained(save_directory)

        # save model
        output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)
        with open(output_model_file, "wb") as f:
            params = params if params is not None else self.params
            model_bytes = to_bytes(params)
            f.write(model_bytes)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
FlaxPreTrainedModel.push_to_hub = copy_func(FlaxPreTrainedModel.push_to_hub)
FlaxPreTrainedModel.push_to_hub.__doc__ = FlaxPreTrainedModel.push_to_hub.__doc__.format(
    object="model", object_class="FlaxAutoModel", object_files="model checkpoint"
)


def overwrite_call_docstring(model_class, docstring):
    # copy __call__ function to be sure docstring is changed only for this function
    model_class.__call__ = copy_func(model_class.__call__)
    # delete existing docstring
    model_class.__call__.__doc__ = None
    # set correct docstring
    model_class.__call__ = add_start_docstrings_to_model_forward(docstring)(model_class.__call__)


def append_call_sample_docstring(model_class, tokenizer_class, checkpoint, output_type, config_class, mask=None):
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = add_code_sample_docstrings(
        tokenizer_class=tokenizer_class,
        checkpoint=checkpoint,
        output_type=output_type,
        config_class=config_class,
        model_cls=model_class.__name__,
    )(model_class.__call__)


def append_replace_return_docstrings(model_class, output_type, config_class):
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = replace_return_docstrings(
        output_type=output_type,
        config_class=config_class,
    )(model_class.__call__)
