# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Factory function to build auto-model classes."""

import types

from ...configuration_utils import PretrainedConfig
from ...file_utils import copy_func
from ...integrations import deepspeed_config, is_deepspeed_zero3_enabled
from ...utils import logging
from .configuration_auto import AutoConfig, replace_list_option_in_docstrings


logger = logging.get_logger(__name__)


CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the :meth:`~transformers.BaseAutoModelClass.from_pretrained` class method or the
    :meth:`~transformers.BaseAutoModelClass.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
"""

FROM_CONFIG_DOCSTRING = """
        Instantiates one of the model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use :meth:`~transformers.BaseAutoModelClass.from_pretrained` to load the model
            weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, BaseAutoModelClass
            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoConfig.from_pretrained('checkpoint_placeholder')
            >>> model = BaseAutoModelClass.from_config(config)
"""

FROM_PRETRAINED_TORCH_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the :obj:`model_type` property of the config object (either
        passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        The model is set in evaluation mode by default using ``model.eval()`` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with ``model.train()``

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (additional positional arguments, `optional`):
                Will be passed along to the underlying model ``__init__()`` method.
            config (:class:`~transformers.PretrainedConfig`, `optional`):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :meth:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
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
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            kwargs (additional keyword arguments, `optional`):
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

            >>> from transformers import AutoConfig, BaseAutoModelClass

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder')

            >>> # Update configuration during loading
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_pretrained('./tf_model/shortcut_placeholder_tf_model_config.json')
            >>> model = BaseAutoModelClass.from_pretrained('./tf_model/shortcut_placeholder_tf_checkpoint.ckpt.index', from_tf=True, config=config)
"""

FROM_PRETRAINED_TF_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the :obj:`model_type` property of the config object (either
        passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch state_dict save file` (e.g, ``./pt_model/pytorch_model.bin``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
            model_args (additional positional arguments, `optional`):
                Will be passed along to the underlying model ``__init__()`` method.
            config (:class:`~transformers.PretrainedConfig`, `optional`):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :meth:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
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
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            kwargs (additional keyword arguments, `optional`):
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

            >>> from transformers import AutoConfig, BaseAutoModelClass

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder')

            >>> # Update configuration during loading
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)
            >>> config = AutoConfig.from_pretrained('./pt_model/shortcut_placeholder_pt_model_config.json')
            >>> model = BaseAutoModelClass.from_pretrained('./pt_model/shortcut_placeholder_pytorch_model.bin', from_pt=True, config=config)
"""

FROM_PRETRAINED_FLAX_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the :obj:`model_type` property of the config object (either
        passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch state_dict save file` (e.g, ``./pt_model/pytorch_model.bin``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
            model_args (additional positional arguments, `optional`):
                Will be passed along to the underlying model ``__init__()`` method.
            config (:class:`~transformers.PretrainedConfig`, `optional`):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :meth:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
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
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            kwargs (additional keyword arguments, `optional`):
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

            >>> from transformers import AutoConfig, BaseAutoModelClass

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder')

            >>> # Update configuration during loading
            >>> model = BaseAutoModelClass.from_pretrained('checkpoint_placeholder', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)
            >>> config = AutoConfig.from_pretrained('./pt_model/shortcut_placeholder_pt_model_config.json')
            >>> model = BaseAutoModelClass.from_pretrained('./pt_model/shortcut_placeholder_pytorch_model.bin', from_pt=True, config=config)
"""


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    def from_config(cls, config, **kwargs):
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            if is_deepspeed_zero3_enabled():
                import deepspeed

                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                # this immediately partitions the model across all gpus, to avoid the overhead in time
                # and memory copying it on CPU or each GPU first
                with deepspeed.zero.Init(config=deepspeed_config()):
                    return model_class(config, **kwargs)
            else:
                return model_class(config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )


def insert_head_doc(docstring, head_doc=""):
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )


def auto_class_factory(name, model_mapping, checkpoint_for_example="bert-base-cased", head_doc=""):
    # Create a new class with the right name from the base class
    new_class = types.new_class(name, (_BaseAutoModelClass,))
    new_class._model_mapping = model_mapping
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    new_class.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Now we need to copy and re-register `from_config` and `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(model_mapping, use_model_types=False)(from_config)
    new_class.from_config = classmethod(from_config)

    if name.startswith("TF"):
        from_pretrained_docstring = FROM_PRETRAINED_TF_DOCSTRING
    elif name.startswith("Flax"):
        from_pretrained_docstring = FROM_PRETRAINED_FLAX_DOCSTRING
    else:
        from_pretrained_docstring = FROM_PRETRAINED_TORCH_DOCSTRING
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace("BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace("shortcut_placeholder", shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(model_mapping)(from_pretrained)
    new_class.from_pretrained = classmethod(from_pretrained)
    return new_class


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result
