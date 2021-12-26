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
import importlib
from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ...file_utils import copy_func
from ...utils import logging
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
from .dynamic import get_class_from_dynamic_module


logger = logging.get_logger(__name__)


CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the [`~BaseAutoModelClass.from_pretrained`] class method or the
    [`~BaseAutoModelClass.from_config`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
"""

FROM_CONFIG_DOCSTRING = """
        Instantiates one of the model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use [`~BaseAutoModelClass.from_pretrained`] to load the model
            weights.

        Args:
            config ([`PretrainedConfig`]):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass
        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained('checkpoint_placeholder')
        >>> model = BaseAutoModelClass.from_config(config)
        ```
"""

FROM_PRETRAINED_TORCH_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are
        deactivated). To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under
                      a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided
                      as `config` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (*Dict[str, torch.Tensor]*, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it
                will execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of
                      `kwargs` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied `kwargs` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
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
        ```
"""

FROM_PRETRAINED_TF_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under
                      a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In
                      this case, `from_pt` should be set to `True` and a configuration object should be provided
                      as `config` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it
                will execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of
                      `kwargs` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied `kwargs` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
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
        ```
"""

FROM_PRETRAINED_FLAX_DOCSTRING = """
        Instantiate one of the model classes of the library from a pretrained model.

        The model class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing,
        by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under
                      a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In
                      this case, `from_pt` should be set to `True` and a configuration object should be provided
                      as `config` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
            model_args (additional positional arguments, *optional*):
                Will be passed along to the underlying model `__init__()` method.
            config ([`PretrainedConfig`], *optional*):
                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
            revision(`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it
                will execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of
                      `kwargs` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied `kwargs` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's `__init__` function.

        Examples:

        ```python
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
        ```
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

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    "Loading this model requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warn(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(config.name_or_path, module_file + ".py", class_name, **kwargs)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **kwargs
            )
        if hasattr(config, "auto_map") and cls.__name__ in config.auto_map:
            if not trust_remote_code:
                raise ValueError(
                    f"Loading {pretrained_model_name_or_path} requires you to execute the modeling file in that repo "
                    "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                    "the option `trust_remote_code=True` to remove this error."
                )
            if kwargs.get("revision", None) is None:
                logger.warn(
                    "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                    "no malicious code has been contributed in a newer revision."
                )
            class_ref = config.auto_map[cls.__name__]
            module_file, class_name = class_ref.split(".")
            model_class = get_class_from_dynamic_module(
                pretrained_model_name_or_path, module_file + ".py", class_name, **kwargs
            )
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, model_class):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class)


def insert_head_doc(docstring, head_doc=""):
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )


def auto_class_update(cls, checkpoint_for_example="bert-base-cased", head_doc=""):
    # Create a new class with the right name from the base class
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Now we need to copy and re-register `from_config` and `from_pretrained` as class methods otherwise we can't
    # have a specific docstrings for them.
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(model_mapping._model_mapping, use_model_types=False)(from_config)
    cls.from_config = classmethod(from_config)

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
    from_pretrained = replace_list_option_in_docstrings(model_mapping._model_mapping)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    transformers_module = importlib.import_module("transformers")
    return getattribute_from_module(transformers_module, attr)


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:

        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type not in self._model_mapping:
            raise KeyError(key)
        model_name = self._model_mapping[model_type]
        return self._load_attr_from_module(model_type, model_name)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value
