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
"""
Feature extraction saving/loading class for common feature extractors.
"""

import os
from collections import UserDict
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np

from .processing_base import ProcessingMixin
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    TensorType,
    _is_tensor_or_array_like,
    copy_func,
    is_numpy_array,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
)


if TYPE_CHECKING:
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor


logger = logging.get_logger(__name__)

PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]

# type hinting: specifying the type of feature extractor class that inherits from FeatureExtractionMixin
SpecificFeatureExtractorType = TypeVar("SpecificFeatureExtractorType", bound="FeatureExtractionMixin")


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


class FeatureExtractionMixin(ProcessingMixin):
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and audio feature
    extractors.
    """

    _config_name = FEATURE_EXTRACTOR_NAME
    _type_key = "feature_extractor_type"
    _nested_config_keys = ["feature_extractor", "audio_processor"]
    _auto_class_default = "AutoFeatureExtractor"
    _file_type_label = "feature extractor"
    _excluded_dict_keys = {"mel_filters", "window"}
    _extra_init_pops = []
    _config_filename_kwarg = None
    _subfolder_default = None

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        return cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: dict[str, Any], **kwargs
    ) -> Union["FeatureExtractionMixin", tuple["FeatureExtractionMixin", dict[str, Any]]]:
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in feature_extractor_dict:
                feature_extractor_dict[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        feature_extractor = cls(**feature_extractor_dict)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor


FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)
if FeatureExtractionMixin.push_to_hub.__doc__ is not None:
    FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(
        object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file"
    )
