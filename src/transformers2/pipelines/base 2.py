# coding=utf-8
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
import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import ModelOutput, add_end_docstrings, infer_framework, is_tf_available, is_torch_available, logging


GenericTensor = Union[List["GenericTensor"], "torch.Tensor", "tf.Tensor"]

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TFAutoModel

if is_torch_available():
    import torch
    from torch.utils.data import DataLoader, Dataset

    from ..models.auto.modeling_auto import AutoModel

    # Re-export for backward compatibility
    from .pt_utils import KeyDataset
else:
    Dataset = None
    KeyDataset = None

if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side):
    batch_size = len(items)
    if isinstance(items[0][key], torch.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if key in ["pixel_values", "image"]:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 4 and key == "input_features":
            # this is probably a mel spectrogram batched
            return torch.cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            if max_length == min_length:
                # Bypass for `ImageGPT` which doesn't provide a padding value, yet
                # we can consistently pad since the size should be matching
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value
        elif dim == 4:
            tensor = torch.zeros((batch_size, max_length, shape[-2], shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
            elif dim == 4:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :, :] = item[key][0].clone()

        return tensor
    else:
        return [item[key] for item in items]


def pad_collate_fn(tokenizer, feature_extractor):
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            if key in {"input_ids"}:
                # ImageGPT uses a feature extractor
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                _padding_value = 0
            else:
                # This is likely another random key maybe even user provided
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded

    return inner


def infer_framework_load_model(
    model,
    config: AutoConfig,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs,
):
    """
    Select framework (TensorFlow or PyTorch) to use from the `model` passed. Returns a tuple (framework, model).

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for `model`, PyTorch is selected.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        config ([`AutoConfig`]):
            The config associated with the model to help using the correct class
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        task (`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        `Tuple`: A tuple framework, model.
    """
    if not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    if isinstance(model, str):
        model_kwargs["_from_pipeline"] = task
        class_tuple = ()
        look_pt = is_torch_available() and framework in {"pt", None}
        look_tf = is_tf_available() and framework in {"tf", None}
        if model_classes:
            if look_pt:
                class_tuple = class_tuple + model_classes.get("pt", (AutoModel,))
            if look_tf:
                class_tuple = class_tuple + model_classes.get("tf", (TFAutoModel,))
        if config.architectures:
            classes = []
            for architecture in config.architectures:
                transformers_module = importlib.import_module("transformers")
                if look_pt:
                    _class = getattr(transformers_module, architecture, None)
                    if _class is not None:
                        classes.append(_class)
                if look_tf:
                    _class = getattr(transformers_module, f"TF{architecture}", None)
                    if _class is not None:
                        classes.append(_class)
            class_tuple = class_tuple + tuple(classes)

        if len(class_tuple) == 0:
            raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")

        all_traceback = {}
        for model_class in class_tuple:
            kwargs = model_kwargs.copy()
            if framework == "pt" and model.endswith(".h5"):
                kwargs["from_tf"] = True
                logger.warning(
                    "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                    "Trying to load the model with PyTorch."
                )
            elif framework == "tf" and model.endswith(".bin"):
                kwargs["from_pt"] = True
                logger.warning(
                    "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                    "Trying to load the model with Tensorflow."
                )

            try:
                model = model_class.from_pretrained(model, **kwargs)
                if hasattr(model, "eval"):
                    model = model.eval()
                # Stop loading on the first successful load.
                break
            except (OSError, ValueError):
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue

        if isinstance(model, str):
            error = ""
            for class_name, trace in all_traceback.items():
                error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
            raise ValueError(
                f"Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n"
            )

    if framework is None:
        framework = infer_framework(model.__class__)
    return framework, model


def infer_framework_from_model(
    model,
    model_classes: Optional[Dict[str, Tuple[type]]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    **model_kwargs,
):
    """
    Select framework (TensorFlow or PyTorch) to use from the `model` passed. Returns a tuple (framework, model).

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for `model`, PyTorch is selected.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        task (`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        `Tuple`: A tuple framework, model.
    """
    if isinstance(model, str):
        config = AutoConfig.from_pretrained(model, _from_pipeline=task, **model_kwargs)
    else:
        config = model.config
    return infer_framework_load_model(
        model, config, model_classes=model_classes, _from_pipeline=task, task=task, framework=framework, **model_kwargs
    )


def get_framework(model, revision: Optional[str] = None):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
    warnings.warn(
        "`get_framework` is deprecated and will be removed in v5, use `infer_framework_from_model` instead.",
        FutureWarning,
    )
    if not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    if isinstance(model, str):
        if is_torch_available() and not is_tf_available():
            model = AutoModel.from_pretrained(model, revision=revision)
        elif is_tf_available() and not is_torch_available():
            model = TFAutoModel.from_pretrained(model, revision=revision)
        else:
            try:
                model = AutoModel.from_pretrained(model, revision=revision)
            except OSError:
                model = TFAutoModel.from_pretrained(model, revision=revision)

    framework = infer_framework(model.__class__)
    return framework


def get_default_model_and_revision(
    targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]
) -> Union[str, Tuple[str, str]]:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        `str` The model string representing the default model for this pipeline
    """
    if is_torch_available() and not is_tf_available():
        framework = "pt"
    elif is_tf_available() and not is_torch_available():
        framework = "tf"

    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError(f"The task does not provide any default models for options {task_options}")
        default_models = defaults[task_options]["model"]
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # XXX This error message needs to be updated to be more generic if more tasks are going to become
        # parametrized
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    if framework is None:
        framework = "pt"

    return default_models[framework]


class PipelineException(Exception):
    """
    Raised by a [`Pipeline`] when handling __call__.

    Args:
        task (`str`): The task of the pipeline.
        model (`str`): The model used by the pipeline.
        reason (`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing. Supported data formats
    currently includes:

    - JSON
    - CSV
    - stdin/stdout (pipe)

    `PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
    pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError(f"{self.output_path} already exists on disk")

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError(f"{self.input_path} doesnt exist on disk")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`dict` or list of `dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (`dict` or list of `dict`): The data to store.

        Returns:
            `str`: Path where the data has been saved.
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ) -> "PipelineDataFormat":
        """
        Creates an instance of the right subclass of [`~pipelines.PipelineDataFormat`] depending on `format`.

        Args:
            format (`str`):
                The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
            output_path (`str`, *optional*):
                Where to save the outgoing data.
            input_path (`str`, *optional*):
                Where to look for the input data.
            column (`str`, *optional*):
                The column to read.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether or not to overwrite the `output_path`.

        Returns:
            [`~pipelines.PipelineDataFormat`]: The proper data format.
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError(f"Unknown reader {format} (Available reader are json/csv/pipe)")


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current [`~pipelines.PipelineDataFormat`].

        Args:
            data (`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (`str`): Where to save the outgoing data.
        input_path (`str`): Where to look for the input data.
        column (`str`): The column to read.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the `output_path`.
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:
                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (`dict`): The data to store.
        """
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


PIPELINE_INIT_ARGS = r"""
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too.
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""

if is_torch_available():
    from transformers.pipelines.pt_utils import (
        PipelineChunkIterator,
        PipelineDataset,
        PipelineIterator,
        PipelinePackIterator,
    )


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance [`FeatureExtractionPipeline`] (`'feature-extraction'`) output large tensor object
    as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
    constructor argument. If set to `True`, the output will be stored in the pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        image_processor: Optional[BaseImageProcessor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: Union[int, "torch.device"] = None,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        binary_output: bool = False,
        **kwargs,
    ):
        if framework is None:
            framework, model = infer_framework_load_model(model, config=model.config)

        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.modelcard = modelcard
        self.framework = framework

        # `accelerate` device map
        hf_device_map = getattr(self.model, "hf_device_map", None)

        if hf_device_map is not None and device is not None:
            raise ValueError(
                "The model has been loaded with `accelerate` and therefore cannot be moved to a specific device. Please "
                "discard the `device` argument when creating your pipeline object."
            )

        # We shouldn't call `model.to()` for models loaded with accelerate
        if self.framework == "pt" and device is not None and not (isinstance(device, int) and device < 0):
            self.model.to(device)

        if device is None:
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = -1

        if is_torch_available() and self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device if device is not None else -1
        self.torch_dtype = torch_dtype
        self.binary_output = binary_output

        # Update config and generation_config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))
            if self.model.can_generate():
                self.model.generation_config.update(**task_specific_params.get(task))

        self.call_count = 0
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = kwargs.pop("num_workers", None)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)

        if self.image_processor is None and self.feature_extractor is not None:
            if isinstance(self.feature_extractor, BaseImageProcessor):
                # Backward compatible change, if users called
                # ImageSegmentationPipeline(.., feature_extractor=MyFeatureExtractor())
                # then we should keep working
                self.image_processor = self.feature_extractor

    def save_pretrained(self, save_directory: str, safe_serialization: bool = False):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            safe_serialization (`str`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch or Tensorflow
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        if hasattr(self, "_registered_impl"):
            # Add info to the config
            pipeline_info = self._registered_impl.copy()
            custom_pipelines = {}
            for task, info in pipeline_info.items():
                if info["impl"] != self.__class__:
                    continue

                info = info.copy()
                module_name = info["impl"].__module__
                last_module = module_name.split(".")[-1]
                # Change classes into their names/full names
                info["impl"] = f"{last_module}.{info['impl'].__name__}"
                info["pt"] = tuple(c.__name__ for c in info["pt"])
                info["tf"] = tuple(c.__name__ for c in info["tf"])

                custom_pipelines[task] = info
            self.model.config.custom_pipelines = custom_pipelines
            # Save the pipeline custom code
            custom_object_save(self, save_directory)

        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)

        if self.image_processor is not None:
            self.image_processor.save_pretrained(save_directory)

        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples:

        ```python
        # Explicitly ask for tensor allocation on CUDA device :0
        pipe = pipeline(..., device=0)
        with pipe.device_placement():
            # Every framework specific tensor allocation will be done on the request device
            output = pipe(...)
        ```"""
        if self.framework == "tf":
            with tf.device("/CPU:0" if self.device == -1 else f"/device:GPU:{self.device}"):
                yield
        else:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    yield
            else:
                yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        elif isinstance(inputs, dict):
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        elif isinstance(inputs, UserDict):
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models_names = []
            for _, model_name in supported_models.items():
                # Mapping can now contain tuples of models for the same configuration.
                if isinstance(model_name, tuple):
                    supported_models_names.extend(list(model_name))
                else:
                    supported_models_names.append(model_name)
            if hasattr(supported_models, "_model_mapping"):
                for _, model in supported_models._model_mapping._extra_content.items():
                    if isinstance(model_name, tuple):
                        supported_models_names.extend([m.__name__ for m in model])
                    else:
                        supported_models_names.append(model.__name__)
            supported_models = supported_models_names
        if self.model.__class__.__name__ not in supported_models:
            logger.error(
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are"
                f" {supported_models}."
            )

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    def get_inference_context(self):
        return torch.no_grad

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == "tf":
                model_inputs["training"] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            logger.warning(f"Ignoring args : {args}")

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
                UserWarning,
            )

        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        is_iterable = is_dataset or is_generator or is_list

        # TODO make the get_iterator work also for `tf` (and `flax`).
        can_use_iterator = self.framework == "pt" and (is_dataset or is_generator or is_list)

        if is_list:
            if can_use_iterator:
                final_iterator = self.get_iterator(
                    inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                )
                outputs = list(final_iterator)
                return outputs
            else:
                return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        elif can_use_iterator:
            return self.get_iterator(
                inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
            )
        elif is_iterable:
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        elif self.framework == "pt" and isinstance(self, ChunkPipeline):
            return next(
                iter(
                    self.get_iterator(
                        [inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params
                    )
                )
            )
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # This function should become `get_iterator` again, this is a temporary
        # easy solution.
        for input_ in inputs:
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)


class ChunkPipeline(Pipeline):
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        all_outputs = []
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            model_outputs = self.forward(model_inputs, **forward_params)
            all_outputs.append(model_outputs)
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if num_workers > 1:
            logger.warning(
                "For ChunkPipeline using num_workers>0 is likely to result in errors since everything is iterable,"
                " setting `num_workers=1` to guarantee correctness."
            )
            num_workers = 1
        dataset = PipelineChunkIterator(inputs, self.preprocess, preprocess_params)

        # TODO hack by collating feature_extractor and image_processor
        feature_extractor = self.feature_extractor if self.feature_extractor is not None else self.image_processor
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelinePackIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator


class PipelineRegistry:
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        if task in self.task_aliases:
            task = self.task_aliases[task]
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return task, targeted_task, None

        if task.startswith("translation"):
            tokens = task.split("_")
            if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
                targeted_task = self.supported_tasks["translation"]
                task = "translation"
                return task, targeted_task, (tokens[1], tokens[3])
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}"
        )

    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        pt_model: Optional[Union[type, Tuple[type]]] = None,
        tf_model: Optional[Union[type, Tuple[type]]] = None,
        default: Optional[Dict] = None,
        type: Optional[str] = None,
    ) -> None:
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}...")

        if pt_model is None:
            pt_model = ()
        elif not isinstance(pt_model, tuple):
            pt_model = (pt_model,)

        if tf_model is None:
            tf_model = ()
        elif not isinstance(tf_model, tuple):
            tf_model = (tf_model,)

        task_impl = {"impl": pipeline_class, "pt": pt_model, "tf": tf_model}

        if default is not None:
            if "model" not in default and ("pt" in default or "tf" in default):
                default = {"model": default}
            task_impl["default"] = default

        if type is not None:
            task_impl["type"] = type

        self.supported_tasks[task] = task_impl
        pipeline_class._registered_impl = {task: task_impl}

    def to_dict(self):
        return self.supported_tasks
