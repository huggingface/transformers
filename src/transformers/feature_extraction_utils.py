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
"""
 Feature extraction common class for python feature extractors.
"""
import copy
import json
import os
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .file_utils import (
    FEATURE_EXTRACTOR_NAME,
    PaddingStrategy,
    TensorType,
    _is_jax,
    _is_numpy,
    _is_tensorflow,
    _is_torch,
    _is_torch_device,
    cached_path,
    hf_bucket_url,
    is_flax_available,
    is_remote_url,
    is_tf_available,
    is_torch_available,
    to_py_obj,
    torch_required,
)
from .utils import logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    if is_torch_available():
        import torch


class BatchFeature(UserDict):
    r"""
    Holds the output of the :meth:`~transformers.PreTrainedFeatureExtractor.pad` and feature extractor specific
    ``__call__`` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (:obj:`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (:obj:`Union[None, str, TensorType]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to :obj:`key` ('input_values',
        'attention_mask', etc.).
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

    # Copied from transformers.tokenization_utils_base.BatchEncoding.keys
    def keys(self):
        return self.data.keys()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.values
    def values(self):
        return self.data.values()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.items
    def items(self):
        return self.data.items()

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                The type of tensors to use. If :obj:`str`, should be one of the values of the enum
                :class:`~transformers.file_utils.TensorType`. If :obj:`None`, no modification is done.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
            is_tensor = _is_jax
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        return self

    @torch_required
    # Copied from transformers.tokenization_utils_base.BatchEncoding.to with BatchEncoding->BatchFeature
    def to(self, device: Union[str, "torch.device"]) -> "BatchFeature":
        """
        Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`~transformers.BatchFeature`: The same instance of :class:`~transformers.BatchFeature` after
            modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchFeature to type {str(device)}. This is not supported.")
        return self


class PreTrainedFeatureExtractor:
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (:obj:`int`):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (:obj:`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PreTrainedFeatureExtractor":
        r"""
        Instantiate a :class:`~transformers.PreTrainedFeatureExtractor` (or a derived class) from a pretrained feature
        extractor.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :func:`~transformers.PreTrainedFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/feature_extraction_config.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final feature extractor object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(feature_extractor, unused_kwargs)` where
                `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not feature extractor
                attributes: i.e., the part of ``kwargs`` which has not been used to update ``feature_extractor`` and is
                otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`~transformers.PreTrainedFeatureExtractor`: The feature extractor object instantiated from this
            pretrained model.

        Examples::

            # We can't instantiate directly the base class `PreTrainedFeatureExtractor` so let's show the examples on a
            # derived class: Wav2Vec2FeatureExtractor
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')    # Download feature_extraction_config from huggingface.co and cache.
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/')  # E.g. feature_extractor (or model) was saved using `save_pretrained('./test/saved_model/')`
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/preprocessor_config.json')
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False, foo=False)
            assert feature_extractor.return_attention_mask is False
            feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False,
                                                               foo=False, return_unused_kwargs=True)
            assert feature_extractor.return_attention_mask is False
            assert unused_kwargs == {'foo': False}

        """
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a feature_extractor object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PreTrainedFeatureExtractor.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Configuration saved in {output_feature_extractor_file}")

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PreTrainedFeatureExtractor` using ``from_dict``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor
            object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path
        else:
            feature_extractor_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=FEATURE_EXTRACTOR_NAME, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_feature_extractor_file = cached_path(
                feature_extractor_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load feature extractor for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {FEATURE_EXTRACTOR_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                f"Couldn't reach server at '{feature_extractor_file}' to download feature extractor configuration file or "
                "feature extractor configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_feature_extractor_file}."
            )
            raise EnvironmentError(msg)

        if resolved_feature_extractor_file == feature_extractor_file:
            logger.info(f"loading feature extractor configuration file {feature_extractor_file}")
        else:
            logger.info(
                f"loading feature extractor configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a :class:`~transformers.PreTrainedFeatureExtractor` from a Python dictionary of parameters.

        Args:
            feature_extractor_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PreTrainedFeatureExtractor.to_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            :class:`~transformers.PreTrainedFeatureExtractor`: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        output = copy.deepcopy(self.__dict__)

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a :class:`~transformers.PreTrainedFeatureExtractor` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`~transformers.PreTrainedFeatureExtractor`: The feature_extractor object instantiated from that JSON
            file.

        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this feature_extractor instance in JSON
            format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with
        ``self.padding_side``, ``self.padding_value``)

        .. note::

            If the ``processed_features`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors,
            the result will use the same type unless you provide a different tensor type with ``return_tensors``. In
            the case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            processed_features (:class:`~transformers.BatchFeature`, list of :class:`~transformers.BatchFeature`, :obj:`Dict[str, List[float]]`, :obj:`Dict[str, List[List[float]]` or :obj:`List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input (:class:`~transformers.BatchFeature` or :obj:`Dict[str,
                List[float]]`) or a batch of input values / vectors (list of :class:`~transformers.BatchFeature`,
                `Dict[str, List[List[float]]]` or `List[Dict[str, List[float]]]`) so you can use this method during
                preprocessing as well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow
                tensors), see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of :class:`~transformers.BatchFeature` or list of :class:`~transformers.BatchFeature` to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        if not required_input:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (float, int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in processed_features.items():
                processed_features[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, max_length, _ = self._get_padding_strategies(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            processed_features = self._pad(
                processed_features,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchFeature(processed_features, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in processed_features.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in processed_features.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Union[Dict[str, List[float]], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features: Dictionary of input values (`List[float]`) / input vectors (`List[List[float]]`) or batch of inputs values (`List[List[int]]`) / input vectors (`List[List[List[int]]]`)
            max_length: maximum length of the returned list and optionally padding length (see below)
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_vector = self.feature_size * [self.padding_value] if self.feature_size > 1 else self.padding_value
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = [1] * len(required_input) + [0] * difference
                processed_features[self.model_input_names[0]] = required_input + [
                    padding_vector for _ in range(difference)
                ]
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = [0] * difference + [1] * len(required_input)
                processed_features[self.model_input_names[0]] = [
                    padding_vector for _ in range(difference)
                ] + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = [1] * len(required_input)

        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None, pad_to_multiple_of=None, **kwargs):
        """
        Find the correct padding strategy
        """

        # Get padding strategy
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that" f" max_length is defined"
                )

        # Test if we have a padding value
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. "
                "Please select a value to use as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy, max_length, kwargs
