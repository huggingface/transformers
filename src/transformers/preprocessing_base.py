# Copyright 2020 The HuggingFace Inc. team.
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
"""
Base mixin for image processors and feature extractors, providing shared
save/load/serialization logic.
"""

import copy
import json
import os
from collections import UserDict
from copy import deepcopy
from typing import Any, TypeVar

import numpy as np
from huggingface_hub import create_repo, is_offline_mode
from huggingface_hub.dataclasses import validate_typed_dict

from .dynamic_module_utils import custom_object_save
from .utils import (
    PROCESSOR_NAME,
    PushToHubMixin,
    TensorType,
    _is_tensor_or_array_like,
    is_numpy_array,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
    safe_load_json_file,
)
from .utils.hub import cached_file


logger = logging.get_logger(__name__)

PreprocessingMixinType = TypeVar("PreprocessingMixinType", bound="PreprocessingMixin")


class BatchFeature(UserDict):
    r"""
    Holds the output of the [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
            initialization.
        skip_tensor_conversion (`list[str]` or `set[str]`, *optional*):
            List or set of keys that should NOT be converted to tensors, even when `tensor_type` is specified.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        tensor_type: None | str | TensorType = None,
        skip_tensor_conversion: list[str] | set[str] | None = None,
    ):
        super().__init__(data)
        self.skip_tensor_conversion = skip_tensor_conversion
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Any:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def _get_is_as_tensor_fns(self, tensor_type: str | TensorType | None = None):
        if tensor_type is None:
            return None, None

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            def as_tensor(value):
                if torch.is_tensor(value):
                    return value

                # stack list of tensors if tensor_type is PyTorch (# torch.tensor() does not support list of tensors)
                if isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                    return torch.stack(value)

                # convert list of numpy arrays to numpy array (stack) if tensor_type is Numpy
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        value = np.array(value)
                    elif (
                        isinstance(value[0], (list, tuple))
                        and len(value[0]) > 0
                        and isinstance(value[0][0], np.ndarray)
                    ):
                        value = np.array(value)
                if isinstance(value, np.ndarray):
                    return torch.from_numpy(value)
                else:
                    return torch.tensor(value)

            is_tensor = torch.is_tensor
        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array
        return is_tensor, as_tensor

    def convert_to_tensors(
        self,
        tensor_type: str | TensorType | None = None,
        skip_tensor_conversion: list[str] | set[str] | None = None,
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            skip_tensor_conversion (`list[str]` or `set[str]`, *optional*):
                List or set of keys that should NOT be converted to tensors, even when `tensor_type` is specified.

        Note:
            Values that don't have an array-like structure (e.g., strings, dicts, lists of strings) are
            automatically skipped and won't be converted to tensors. Ragged arrays (lists of arrays with
            different lengths) are still attempted, though they may raise errors during conversion.
        """
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)
        skip_tensor_conversion = (
            skip_tensor_conversion if skip_tensor_conversion is not None else self.skip_tensor_conversion
        )

        # Do the tensor conversion in batch
        for key, value in self.items():
            # Skip keys explicitly marked for no conversion
            if skip_tensor_conversion and key in skip_tensor_conversion:
                continue

            # Skip values that are not array-like
            if not _is_tensor_or_array_like(value):
                continue

            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_values":
                    raise ValueError(
                        f"Unable to create tensor for '{key}' with overflowing values of different lengths. "
                        f"Original error: {str(e)}"
                    ) from e
                raise ValueError(
                    f"Unable to convert output '{key}' (type: {type(value).__name__}) to tensor: {str(e)}\n"
                    f"You can try:\n"
                    f"  1. Use padding=True to ensure all outputs have the same shape\n"
                    f"  2. Set return_tensors=None to return Python objects instead of tensors"
                ) from e

        return self

    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.
                To enable asynchronous data transfer, set the `non_blocking` flag in `kwargs` (defaults to `False`).

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch

        device = kwargs.get("device")
        non_blocking = kwargs.get("non_blocking", False)
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        def maybe_to(v):
            # check if v is a floating point tensor
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif isinstance(v, torch.Tensor) and device is not None:
                return v.to(device=device, non_blocking=non_blocking)
            # recursively handle lists and tuples
            elif isinstance(v, (list, tuple)):
                return type(v)(maybe_to(item) for item in v)
            else:
                return v

        self.data = {k: maybe_to(v) for k, v in self.items()}
        return self


class PreprocessingMixin(PushToHubMixin):
    """
    Base mixin providing saving/loading functionality shared by
    ImageProcessingMixin, AudioProcessingMixin and FeatureExtractionMixin.

    Subclasses must set the following class attributes:
        _config_name: str            — config file name (e.g. IMAGE_PROCESSOR_NAME)
        _type_key: str               — key added in to_dict() (e.g. "image_processor_type")
        _nested_config_keys: list    — keys to check in processor_config.json
        _auto_class_default: str     — default auto class for register_for_auto_class
        _file_type_label: str        — label for user-agent / error messages
    Optional:
        _excluded_dict_keys: set     — keys to drop from to_dict() output
        _extra_init_pops: list       — extra keys to pop in __init__
        _config_filename_kwarg: str  — kwarg name that can override the config filename
        _subfolder_default: str      — default for the subfolder kwarg
    """

    _auto_class = None

    # --- Must be overridden by subclasses ---
    _config_name: str
    _type_key: str
    _nested_config_keys: list[str] = []
    _auto_class_default: str
    _file_type_label: str

    # --- Optional overrides ---
    _excluded_dict_keys: set[str] = set()
    # Drop None-valued attrs whose class default is None from to_dict(). On by default for the
    # modern processor classes; legacy FeatureExtractionMixin opts out to keep full serialization.
    _filter_none_class_defaults: bool = True
    _extra_init_pops: list[str] = []
    _config_filename_kwarg: str | None = None
    _subfolder_default: str | None = ""

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        for key in self._extra_init_pops:
            kwargs.pop(key, None)
        # Pop "processor_class", should not be saved in config
        kwargs.pop("processor_class", None)

        if hasattr(self, "valid_kwargs") and hasattr(self.valid_kwargs, "__annotations__"):
            self._init_kwargs_from_valid_kwargs(kwargs)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _init_kwargs_from_valid_kwargs(self, kwargs: dict):
        """
        Initialize instance attributes from `valid_kwargs` annotations.

        For each key in `self.valid_kwargs.__annotations__`, pops it from `kwargs`
        and sets it on the instance (or deep-copies the class default).
        Also sets `self._valid_kwargs_names`.
        """
        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                setattr(self, key, kwarg)
            else:
                setattr(self, key, deepcopy(getattr(self, key, None)))
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    def _set_attributes(self, **kwargs):
        """Standardize instance attributes for all valid kwargs (e.g. coerce dicts to their canonical form)."""
        attributes = {key: getattr(self, key) for key in self._valid_kwargs_names}
        attributes = self._standardize_kwargs(**attributes)
        for key, value in attributes.items():
            setattr(self, key, value)

    def _standardize_kwargs(self, **kwargs) -> dict:
        """
        Hook: standardize kwargs to canonical format before validation (e.g. coerce dicts to
        `SizeDict`/`SpectrogramConfig`). Overridden by modality base classes; default is a no-op.
        """
        return kwargs

    def _validate_preprocess_kwargs(self, **kwargs):
        """
        Hook: validate the kwargs for the preprocess method. Overridden by modality base classes;
        default is a no-op.
        """

    def preprocess(self, inputs, *args, **kwargs):
        """
        Common preprocess entrypoint: validate received kwargs against `valid_kwargs`, fill in
        defaults from `self`, standardize and validate them, then dispatch to the modality-specific
        `_preprocess_*_like_inputs` implementation via `_preprocess_like_inputs`.
        """
        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Update kwargs that need further processing before being validated
        kwargs = self._standardize_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        return self._preprocess_like_inputs(inputs, *args, **kwargs)

    def _preprocess_like_inputs(self, inputs, *args, **kwargs):
        """
        Dispatch to the modality-specific `_preprocess_*_like_inputs` method
        (e.g. `_preprocess_image_like_inputs`, `_preprocess_audio_like_inputs`).
        Implemented by modality base classes.
        """
        raise NotImplementedError

    def filter_out_unused_kwargs(self, kwargs: dict) -> dict:
        """
        Filter out the unused kwargs from the kwargs dictionary.
        """
        if self.unused_kwargs is None:
            return kwargs

        for kwarg_name in self.unused_kwargs:
            if kwarg_name in kwargs:
                logger.warning_once(f"This processor does not use the `{kwarg_name}` parameter. It will be ignored.")
                kwargs.pop(kwarg_name)
        return kwargs

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        """
        Instantiates a processor from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            A processor of type [`~PreprocessingMixin`].
        """
        config_dict = config_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Use valid_kwargs pattern when available (image/audio processors)
        if hasattr(cls, "valid_kwargs") and hasattr(cls.valid_kwargs, "__annotations__"):
            config_dict.update({k: v for k, v in kwargs.items() if k in cls.valid_kwargs.__annotations__})
            processor = cls(**config_dict)

            # Apply extra kwargs to instance (BC for remote code)
            extra_keys = []
            for key in reversed(list(kwargs.keys())):
                if hasattr(processor, key) and key not in cls.valid_kwargs.__annotations__:
                    setattr(processor, key, kwargs.pop(key, None))
                    extra_keys.append(key)
            if extra_keys:
                logger.warning_once(
                    f"Processor {cls.__name__}: kwargs {extra_keys} were applied for backward compatibility. "
                    f"To avoid this warning, add them to valid_kwargs."
                )
        else:
            processor = cls(**config_dict)

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor

    @classmethod
    def from_pretrained(
        cls: type[PreprocessingMixinType],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> PreprocessingMixinType:
        r"""
        Instantiate a processor from a pretrained model name or path.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a processor file saved using the
                  [`~PreprocessingMixin.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the processor files and override the cached versions if
                they exist.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final processor object. If `True`, then this
                functions returns a `Tuple(processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not processor attributes.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are processor attributes will be used to override the
                loaded values.

        Returns:
            A processor of type [`~PreprocessingMixin`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """
        Save a processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PreprocessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_file = os.path.join(save_directory, self._config_name)

        self.to_json_file(output_file)
        logger.info(f"{self._file_type_label} saved in {output_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_file]

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", cls._subfolder_default)

        # Allow overriding the config filename via a kwarg (e.g. image_processor_filename)
        if cls._config_filename_kwarg is not None:
            config_filename = kwargs.pop(cls._config_filename_kwarg, cls._config_name)
        else:
            config_filename = cls._config_name

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": cls._file_type_label, "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, config_filename)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_config_file = pretrained_model_name_or_path
            resolved_processor_file = None
            is_local = True
        else:
            config_file = config_filename
            try:
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    filename=PROCESSOR_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    filename=config_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load {cls._file_type_label} for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {config_filename} file"
                )

        # Load config dict. Priority goes as (nested config if found -> standalone config)
        # We are downloading both configs because almost all models have a `processor_config.json` but
        # not all of these are nested. We need to check if it was saved recebtly as nested or if it is legacy style
        config_dict = None
        if resolved_processor_file is not None:
            processor_dict = safe_load_json_file(resolved_processor_file)
            for nested_key in cls._nested_config_keys:
                if nested_key in processor_dict:
                    config_dict = processor_dict[nested_key]
                    break

        if resolved_config_file is not None and config_dict is None:
            config_dict = safe_load_json_file(resolved_config_file)

        if config_dict is None:
            raise OSError(
                f"Can't load {cls._file_type_label} for '{pretrained_model_name_or_path}'. If you were trying to load"
                " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                f" directory containing a {config_filename} file"
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(
                f"loading configuration file {config_file} from cache at {resolved_config_file}"
            )

        return config_dict, kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this instance.
        """
        output = copy.deepcopy(self.__dict__)
        output[self._type_key] = self.__class__.__name__
        output.pop("_valid_kwargs_names", None)
        for key in self._excluded_dict_keys:
            if key in output:
                del output[key]
        output = {key: self._serialize_value(key, value) for key, value in output.items()}
        if self._filter_none_class_defaults:
            output = {key: value for key, value in output.items() if self._keep_in_dict(key, value)}
        return output

    def _serialize_value(self, key, value):
        """Hook: coerce a modality-specific attribute to a JSON-friendly form for `to_dict`
        (e.g. `SizeDict`/`SpectrogramConfig` → plain dict). Default: identity."""
        return value

    def _keep_in_dict(self, key, value) -> bool:
        """Keep non-None values; keep an explicit None only when the class default is itself
        non-None (i.e. the user deliberately overrode a real default with None)."""
        if value is not None:
            return True
        class_default = getattr(type(self), key, "NOT_FOUND")
        return class_default != "NOT_FOUND" and class_default is not None

    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike):
        """
        Instantiates a processor from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A processor of type [`~PreprocessingMixin`]: The processor object instantiated from that JSON file.
        """
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls(**config_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class=None):
        """
        Register this class with a given auto class.

        Args:
            auto_class (`str` or `type`, *optional*):
                The auto class to register this new processor with. Defaults to the subclass's `_auto_class_default`.
        """
        if auto_class is None:
            auto_class = cls._auto_class_default

        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class
