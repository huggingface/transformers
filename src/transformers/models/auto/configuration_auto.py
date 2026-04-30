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
"""Auto Config class."""

import importlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterator, KeysView, ValuesView
from typing import Any, TypeVar

from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
from .auto_mappings import CONFIG_MAPPING_NAMES, SPECIAL_MODEL_TYPE_TO_MODULE_NAME


logger = logging.get_logger(__name__)

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


# Add non-standard models that can't be inferred from parsing the code
# New models should follow consistent naming instead of being added here!
CONFIG_MAPPING_NAMES.update(
    {
        "EvollaModel": "EvollaConfig",
        "mlcd": "MLCDVisionConfig",
        "vibevoice_acoustic_tokenizer_decoder": "VibeVoiceAcousticTokenizerDecoderConfig",
        "vibevoice_acoustic_tokenizer_encoder": "VibeVoiceAcousticTokenizerEncoderConfig",
    }
)

# TODO: depecate and remove `gpt-sw3`, old model. And prohibit mapping the same config to different model types
# Auto-classes rely a lot on these, and it is much easier when we have 1-1 mapping
CONFIG_MAPPING_NAMES = OrderedDict(**{"gpt-sw3": "GPT2Config"}, **CONFIG_MAPPING_NAMES)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME.update(
    {
        "EvollaModel": "evolla",
        "vibevoice_acoustic_tokenizer_encoder": "vibevoice_acoustic_tokenizer",
        "vibevoice_acoustic_tokenizer_decoder": "vibevoice_acoustic_tokenizer",
    }
)

# This is tied to the processing `-` -> `_` in `model_type_to_module_name`. For example, instead of putting
# `transfo-xl` (as in `CONFIG_MAPPING_NAMES`), we should use `transfo_xl`.
DEPRECATED_MODELS = []


def model_type_to_module_name(key) -> str:
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


def config_class_to_model_type(config) -> str | None:
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict[str, type[PreTrainedConfig]]):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> type[PreTrainedConfig]:
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self) -> list[str]:
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self) -> list[type[PreTrainedConfig]]:
        return [self[k] for k in self._mapping] + list(self._extra_content.values())

    def items(self) -> list[tuple[str, type[PreTrainedConfig]]]:
        return [(k, self[k]) for k in self._mapping] + list(self._extra_content.items())

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item: object) -> bool:
        return item in self._mapping or item in self._extra_content

    def register(self, key: str, value: type[PreTrainedConfig], exist_ok=False) -> None:
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict[str, str]):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        if self._initialized:
            return

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self) -> KeysView[str]:
        self._initialize()
        return self._data.keys()

    def values(self) -> ValuesView[str]:
        self._initialize()
        return self._data.values()

    def items(self) -> KeysView[str]:
        self._initialize()
        return self._data.keys()

    def __iter__(self) -> Iterator[str]:
        self._initialize()
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        self._initialize()
        return item in self._data


def _get_class_name(model_class: str | list[str]):
    if isinstance(model_class, list | tuple):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in CONFIG_MAPPING_NAMES
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({CONFIG_MAPPING_NAMES[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: CONFIG_MAPPING_NAMES[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(
    config_to_class=None, use_model_types: bool = True
) -> Callable[[_CallableT], _CallableT]:
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            # Example: -OO
            return fn
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self) -> None:
        raise OSError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs) -> PreTrainedConfig:
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PreTrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - a path to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```
        """
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        explicit_local_code = has_local_code and not CONFIG_MAPPING[config_dict["model_type"]].__module__.startswith(
            "transformers."
        )
        if has_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code and not explicit_local_code:
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            # Apply heuristic: if model_type is mistral but layer_types is present, treat as ministral
            if config_dict["model_type"] == "mistral" and "layer_types" in config_dict:
                logger.info(
                    "Detected mistral model with layer_types, treating as ministral for alternating attention compatibility. "
                )
                config_dict["model_type"] = "ministral"

            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date.\n\n"
                    "You can update Transformers with the command `pip install --upgrade transformers`. If this "
                    "does not work, and the checkpoint is very new, then there may not be a release version "
                    "that supports this model yet. In this case, you can get the most up-to-date code by installing "
                    "Transformers from source with the command "
                    "`pip install git+https://github.com/huggingface/transformers.git`"
                )
            return config_class.from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}."
        )

    @staticmethod
    def register(model_type, config, exist_ok=False) -> None:
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PreTrainedConfig`]): The config to register.
        """
        if issubclass(config, PreTrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)


__all__ = ["CONFIG_MAPPING", "AutoConfig"]
