# coding=utf-8
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
 Feature extraction classes for python tokenizers.
"""
import json
import os
import copy
from enum import Enum
from collections import UserDict
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from .file_utils import (
    FEATURE_EXTRACTOR_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
    is_tf_available,
    is_torch_available,
    is_flax_available,
    torch_required,
)
from .utils import logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    if is_torch_available():
        import torch


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, BatchFeature)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif is_tf_available() and _is_tensorflow(obj):
        return obj.numpy().tolist()
    elif is_torch_available() and _is_torch(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class BatchFeature(UserDict):
    """
    Holds the output of the :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus` and
    :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase.batch_encode` methods (tokens,
    attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (:obj:`dict`):
            Dictionary of lists/arrays/tensors returned by the encode/batch_encode methods ('input_ids',
            'attention_mask', etc.).
        encoding (:obj:`tokenizers.Encoding` or :obj:`Sequence[tokenizers.Encoding]`, `optional`):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the :obj:`tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (:obj:`Union[None, str, TensorType]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to add a batch axis when converting to tensors (see :obj:`tensor_type` above).
        n_sequences (:obj:`Optional[int]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
    ):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def __getitem__(self, item: Union[int, str]) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to :obj:`key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the :obj:`tokenizers.Encoding` for batch item with index :obj:`key`.
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

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

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                The type of tensors to use. If :obj:`str`, should be one of the values of the enum
                :class:`~transformers.tokenization_utils_base.TensorType`. If :obj:`None`, no modification is done.
            prepend_batch_axis (:obj:`int`, `optional`, defaults to :obj:`False`):
                Whether or not to add the batch dimension during the conversion.
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
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self

    @torch_required
    def to(self, device: Union[str, "torch.device"]) -> "BatchFeature":
        """
        Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`~transformers.BatchEncoding`: The same instance of :class:`~transformers.BatchEncoding` after
            modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(
                f"Attempting to cast a BatchEncoding to another type, {str(device)}. This is not supported."
            )
        return self


class PreTrainedFeatureExtractor:
    """
    This is a general feature extraction class for speech recognition
    """

    def __init__(
        self, feature_dim: int, padding_value: Optional[int] = None, sampling_rate: Optional[int] = None, **kwargs
    ):
        self.feature_dim = feature_dim
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate

        if self.sampling_rate is None:
            logger.warning(
                f"It is strongly recommended to instantiate {self.__class__} with a set sampling_rate."
                f"Failing to do so can result in silent errors that might be hard to debug."
            )

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PreTrainedFeatureExtractor":
        r"""
        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pretrained model
        feature_extractoruration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model feature_extractoruration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature_extractoruration file saved using the
                  :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved feature_extractoruration JSON `file`, e.g.,
                  ``./my_model_directory/feature_extractoruration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model feature_extractoruration should be cached if
                the standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the feature_extractoruration files and override the cached
                versions if they exist.
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
                If :obj:`False`, then this function returns just the final feature_extractoruration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(feature_extractor, unused_kwargs)` where
                `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not
                feature_extractoruration attributes: i.e., the part of ``kwargs`` which has not been used to update
                ``feature_extractor`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are feature_extractoruration attributes will be used to override
                the loaded values. Behavior concerning key/value pairs whose keys are *not* feature_extractoruration
                attributes is controlled by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The feature_extractoruration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            feature_extractor = BertConfig.from_pretrained('bert-base-uncased')    # Download feature_extractoruration from huggingface.co and cache.
            feature_extractor = BertConfig.from_pretrained('./test/saved_model/')  # E.g. feature_extractor (or model) was saved using `save_pretrained('./test/saved_model/')`
            feature_extractor = BertConfig.from_pretrained('./test/saved_model/my_feature_extractoruration.json')
            feature_extractor = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert feature_extractor.output_attentions == True
            feature_extractor, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert feature_extractor.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a feature_extractoruration object to the directory ``save_directory``, so that it can be re-loaded using
        the :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature_extractoruration JSON file will be saved (will be created if it does not
                exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
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
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature_extractoruration
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
                f"Can't load feature_extractor for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {FEATURE_EXTRACTOR_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                "Couldn't reach server at '{}' to download feature_extractor configuration file or "
                "feature_extractor configuration file is not a valid JSON file. "
                "Please check network or file content here: {}.".format(
                    feature_extractor_file, resolved_feature_extractor_file
                )
            )
            raise EnvironmentError(msg)

        if resolved_feature_extractor_file == feature_extractor_file:
            logger.info("loading feature_extractor configuration file {}".format(feature_extractor_file))
        else:
            logger.info(
                "loading feature_extractor configuration file {} from cache at {}".format(
                    feature_extractor_file, resolved_feature_extractor_file
                )
            )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            feature_extractor_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature_extractoruration object. Such a dictionary can
                be retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_feature_extractor_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the feature_extractoruration object.

        Returns:
            :class:`PretrainedConfig`: The feature_extractoruration object instantiated from those parameters.
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
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PreTrainedFeatureExtractor":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The feature_extractoruration object instantiated from that JSON file.

        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this feature_extractoruration instance in
            JSON format.
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
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

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
        verbose: bool = True,
    ) -> BatchFeature:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)

        .. note::

            If the ``processed_features`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors,
            the result will use the same type unless you provide a different tensor type with ``return_tensors``. In
            the case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            processed_features (:class:`~transformers.BatchFeature`, list of :class:`~transformers.BatchFeature`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchFeature` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchFeature`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
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
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        import ipdb; ipdb.set_trace()
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]

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
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

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
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in processed_features:
                    processed_features["token_type_ids"] = (
                        processed_features["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in processed_features:
                    processed_features["special_tokens_mask"] = (
                        processed_features["special_tokens_mask"] + [1] * difference
                    )
                processed_features[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in processed_features:
                    processed_features["token_type_ids"] = [self.pad_token_type_id] * difference + processed_features[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in processed_features:
                    processed_features["special_tokens_mask"] = [1] * difference + processed_features[
                        "special_tokens_mask"
                    ]
                processed_features[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = [1] * len(required_input)

        return processed_features

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
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

        # Get truncation strategy
        if truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError("...")
            elif truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                raise ValueError("...")
#                if self.model_max_length > LARGE_INTEGER:
#                    if verbose:
#                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
#                            logger.warning(
#                                "Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. "
#                                "Default to no padding."
#                            )
#                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
#                    padding_strategy = PaddingStrategy.DO_NOT_PAD
#                else:
#                    max_length = self.model_max_length

#            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
#                if self.model_max_length > LARGE_INTEGER:
#                    if verbose:
#                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
#                            logger.warning(
#                                "Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. "
#                                "Default to no truncation."
#                            )
#                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
#                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
#                else:
#                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and not self.padding_value:
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs
