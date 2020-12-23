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
import csv
import json
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from transformers.file_utils import add_end_docstrings, is_tf_available, is_torch_available
from transformers.modelcard import ModelCard
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


if is_tf_available():
    import tensorflow as tf

    from transformers.models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
    )

if is_torch_available():
    import torch

    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoModel

if TYPE_CHECKING:
    from transformers.modeling_tf_utils import TFPreTrainedModel
    from transformers.modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


def get_framework(model, revision: Optional[str] = None):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
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

    framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"
    return framework


def get_default_model(targeted_task: Dict, framework: Optional[str], task_options: Optional[Any]) -> str:
    """
    Select a default model to use for a given task. Defaults to pytorch if ambiguous.

    Args:
        targeted_task (:obj:`Dict` ):
           Dictionary representing the given task, that should contain default models

        framework (:obj:`str`, None)
           "pt", "tf" or None, representing a specific framework if it was specified, or None if we don't know yet.

        task_options (:obj:`Any`, None)
           Any further value required by the task to get fully specified, for instance (SRC, TGT) languages for
           translation task.

    Returns

        :obj:`str` The model string representing the default model for this pipeline
    """
    if is_torch_available() and not is_tf_available():
        framework = "pt"
    elif is_tf_available() and not is_torch_available():
        framework = "tf"

    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError("The task does not provide any default models for options {}".format(task_options))
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
    Raised by a :class:`~transformers.Pipeline` when handling __call__.

    Args:
        task (:obj:`str`): The task of the pipeline.
        model (:obj:`str`): The model used by the pipeline.
        reason (:obj:`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each :class:`~transformers.pipelines.Pipeline`.
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

    :obj:`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets
    columns to pipelines keyword arguments through the :obj:`dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
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
                raise OSError("{} already exists on disk".format(self.output_path))

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError("{} doesnt exist on disk".format(self.input_path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.

        Returns:
            :obj:`str`: Path where the data has been saved.
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
        Creates an instance of the right subclass of :class:`~transformers.pipelines.PipelineDataFormat` depending on
        :obj:`format`.

        Args:
            format: (:obj:`str`):
                The format of the desired pipeline. Acceptable values are :obj:`"json"`, :obj:`"csv"` or :obj:`"pipe"`.
            output_path (:obj:`str`, `optional`):
                Where to save the outgoing data.
            input_path (:obj:`str`, `optional`):
                Where to look for the input data.
            column (:obj:`str`, `optional`):
                The column to read.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to overwrite the :obj:`output_path`.

        Returns:
            :class:`~transformers.pipelines.PipelineDataFormat`: The proper data format.
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError("Unknown reader {} (Available reader are json/csv/pipe)".format(format))


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
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
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`List[dict]`): The data to store.
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
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
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
            data (:obj:`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process. For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
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
            data (:obj:`dict`): The data to store.
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
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):

        if framework is None:
            framework = get_framework(model)

        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.device = device if framework == "tf" else torch.device("cpu" if device < 0 else "cuda:{}".format(device))
        self.binary_output = binary_output

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
        if self.framework == "tf":
            with tf.device("/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)):
                yield
        else:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (:obj:`List[str]` or :obj:`dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models = [item[1].__name__ for item in supported_models.items()]
        if self.model.__class__.__name__ not in supported_models:
            raise PipelineException(
                self.task,
                self.model.base_model_prefix,
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}",
            )

    def _parse_and_tokenize(self, inputs, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching

        Args:
            inputs: dict holding all the keyword arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array

        Returns:
            Numpy array
        """
        # Encode for forward
        with self.device_placement():
            if self.framework == "tf":
                # TODO trace model
                predictions = self.model(inputs.data, training=False)[0]
            else:
                with torch.no_grad():
                    inputs = self.ensure_tensor_on_device(**inputs)
                    predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()


@add_end_docstrings(PIPELINE_INIT_ARGS)
class SummarizationPipeline(Pipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'. See the up-to-date
    list of available models on `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__.

    Usage::

        # use bart in pytorch
        summarizer = pipeline("summarization")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)

        # use t5 in tf
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)
    """

    def __init__(self, *args, **kwargs):
        kwargs.update(task="summarization")
        super().__init__(*args, **kwargs)

        self.check_model_type(
            TF_MODEL_WITH_LM_HEAD_MAPPING if self.framework == "tf" else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        )

    def __call__(
        self, *documents, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        r"""
        Summarize the text(s) given as inputs.

        Args:
            documents (`str` or :obj:`List[str]`):
                One or several articles (or one list of articles) to summarize.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **summary_text** (:obj:`str`, present when ``return_text=True``) -- The summary of the corresponding
              input.
            - **summary_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``) --
              The token ids of the summary.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"
        assert len(documents) > 0, "Please provide a document to summarize"

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""

        if isinstance(documents[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"

            documents = ([prefix + document for document in documents[0]],)
            padding = True

        elif isinstance(documents[0], str):
            documents = (prefix + documents[0],)
            padding = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    documents[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*documents, padding=padding)

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]
            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            min_length = generate_kwargs.get("min_length", self.model.config.min_length)
            if input_length < min_length // 2:
                logger.warning(
                    "Your min_length is set to {}, but you input_length is only {}. You might consider decreasing min_length manually, e.g. summarizer('...', min_length=10)".format(
                        min_length, input_length
                    )
                )

            max_length = generate_kwargs.get("max_length", self.model.config.max_length)
            if input_length < max_length:
                logger.warning(
                    "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
                        max_length, input_length
                    )
                )

            summaries = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )

            results = []
            for summary in summaries:
                record = {}
                if return_tensors:
                    record["summary_token_ids"] = summary
                if return_text:
                    record["summary_text"] = self.tokenizer.decode(
                        summary,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


@add_end_docstrings(PIPELINE_INIT_ARGS)
class TranslationPipeline(Pipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=translation>`__.

    Usage::
        en_fr_translator = pipeline("translation_en_to_fr")
        en_fr_translator("How old are you?")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(
            TF_MODEL_WITH_LM_HEAD_MAPPING if self.framework == "tf" else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        )

    def __call__(
        self, *args, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        r"""
        Translate the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Texts to be translated.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **translation_text** (:obj:`str`, present when ``return_text=True``) -- The translation.
            - **translation_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the translation.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""

        if isinstance(args[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            args = ([prefix + text for text in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    args[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, padding=padding)

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]

            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            max_length = generate_kwargs.get("max_length", self.model.config.max_length)
            if input_length > 0.9 * max_length:
                logger.warning(
                    "Your input_length: {} is bigger than 0.9 * max_length: {}. You might consider increasing your max_length manually, e.g. translator('...', max_length=400)".format(
                        input_length, max_length
                    )
                )

            translations = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )
            results = []
            for translation in translations:
                record = {}
                if return_tensors:
                    record["translation_token_ids"] = translation
                if return_text:
                    record["translation_text"] = self.tokenizer.decode(
                        translation,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    This Text2TextGenerationPipeline pipeline can currently be loaded from :func:`~transformers.pipeline` using the
    following task identifier: :obj:`"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models <https://huggingface.co/models?filter=seq2seq>`__.

    Usage::

        text2text_generator = pipeline("text2text-generation")
        text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        )

    def __call__(
        self, *args, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        r"""
        Generate the output text(s) using text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Input text for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        if isinstance(args[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            padding = True

        elif isinstance(args[0], str):
            padding = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    args[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, padding=padding)

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)

            generations = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )
            results = []
            for generation in generations:
                record = {}
                if return_tensors:
                    record["generated_token_ids"] = generation
                if return_text:
                    record["generated_text"] = self.tokenizer.decode(
                        generation,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the
    addition of new user input and generated model responses. A conversation needs to contain an unprocessed user input
    before being passed to the :class:`~transformers.ConversationalPipeline`. This user input is either created when
    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response("input")` after a
    conversation turn.

    Arguments:
        text (:obj:`str`, `optional`):
            The initial user input to start the conversation. If not provided, a user input needs to be provided
            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can
            begin.
        conversation_id (:obj:`uuid.UUID`, `optional`):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.

    Usage::

        conversation = Conversation("Going to the movies tonight - any suggestions?")

        # Steps usually performed by the model when generating a response:
        # 1. Mark the user input as processed (moved to the history)
        conversation.mark_processed()
        # 2. Append a mode response
        conversation.append_response("The Big lebowski.")

        conversation.add_user_input("Is it good?")
    """

    def __init__(self, text: str = None, conversation_id: UUID = None):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        self.uuid: UUID = conversation_id
        self.past_user_inputs: List[str] = []
        self.generated_responses: List[str] = []
        self.history: List[int] = []
        self.new_user_input: Optional[str] = text

    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
        field.

        Args:
            text (:obj:`str`): The user input for the next conversation round.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not existing and unprocessed user input should be overwritten when this function is called.
        """
        if self.new_user_input:
            if overwrite:
                logger.warning(
                    'User input added while unprocessed input was existing: "{}" was overwritten with: "{}".'.format(
                        self.new_user_input, text
                    )
                )
                self.new_user_input = text
            else:
                logger.warning(
                    'User input added while unprocessed input was existing: "{}" new input ignored: "{}". '
                    "Set `overwrite` to True to overwrite unprocessed user input".format(self.new_user_input, text)
                )
        else:
            self.new_user_input = text

    def mark_processed(self):
        """
        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
        empties the :obj:`new_user_input` field.
        """
        if self.new_user_input:
            self.past_user_inputs.append(self.new_user_input)
        self.new_user_input = None

    def append_response(self, response: str):
        """
        Append a response to the list of generated responses.

        Args:
            response (:obj:`str`): The model generated response.
        """
        self.generated_responses.append(response)

    def set_history(self, history: List[int]):
        """
        Updates the value of the history of the conversation. The history is represented by a list of :obj:`token_ids`.
        The history is used by the model to generate responses based on the previous conversation turns.

        Args:
            history (:obj:`List[int]`): History of tokens provided and generated for this conversation.
        """
        self.history = history

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Return:
            :obj:`str`:

            Example: Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user >> Going to the movies tonight - any
            suggestions? bot >> The Big Lebowski
        """
        output = "Conversation id: {} \n".format(self.uuid)
        for user_input, generated_response in zip(self.past_user_inputs, self.generated_responses):
            output += "user >> {} \n".format(user_input)
            output += "bot >> {} \n".format(generated_response)
        if self.new_user_input is not None:
            output += "user >> {} \n".format(self.new_user_input)
        return output


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        min_length_for_response (:obj:`int`, `optional`, defaults to 32):
            The minimum length (in number of tokens) for a response.
    """,
)
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    This conversational pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"conversational"`.

    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,
    currently: `'microsoft/DialoGPT-small'`, `'microsoft/DialoGPT-medium'`, `'microsoft/DialoGPT-large'`. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=conversational>`__.

    Usage::

        conversational_pipeline = pipeline("conversational")

        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")

        conversational_pipeline([conversation_1, conversation_2])

        conversation_1.add_user_input("Is it an action movie?")
        conversation_2.add_user_input("What is the genre of this book?")

        conversational_pipeline([conversation_1, conversation_2])
    """

    def __init__(self, min_length_for_response=32, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We need at least an eos_token
        assert self.tokenizer.eos_token_id is not None, "DialoguePipeline tokenizer should have an EOS token set"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.min_length_for_response = min_length_for_response

    def __call__(
        self,
        conversations: Union[Conversation, List[Conversation]],
        clean_up_tokenization_spaces=True,
        **generate_kwargs
    ):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
        """

        if isinstance(conversations, Conversation):
            conversations = [conversations]
        # Input validation
        if isinstance(conversations, list):
            for conversation in conversations:
                assert isinstance(
                    conversation, Conversation
                ), "DialoguePipeline expects a Conversation or list of Conversations as an input"
                if conversation.new_user_input is None:
                    raise ValueError(
                        "Conversation with UUID {} does not contain new user input to process. "
                        "Add user inputs with the conversation's `add_user_input` method".format(
                            type(conversation.uuid)
                        )
                    )
            assert (
                self.tokenizer.pad_token_id is not None or self.tokenizer.eos_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id or eos_token_id when using a batch input"
        else:
            raise ValueError("DialoguePipeline expects a Conversation or list of Conversations as an input")

        with self.device_placement():

            inputs = self._parse_and_tokenize([conversation.new_user_input for conversation in conversations])
            histories = [conversation.history for conversation in conversations]
            max_length = generate_kwargs.get("max_length", self.model.config.max_length)
            inputs = self._concat_inputs_history(inputs, histories, max_length)

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]

            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            if input_length > 0.9 * max_length:
                logger.warning(
                    "Longest conversation length: {} is bigger than 0.9 * max_length: {}. "
                    "You might consider trimming the early phase of the conversation".format(input_length, max_length)
                )
            generated_responses = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )

            if self.model.config.is_encoder_decoder:
                if self.framework == "pt":
                    history = torch.cat((inputs["input_ids"], generated_responses[:, 1:]), 1)
                elif self.framework == "tf":
                    history = tf.concat([inputs["input_ids"], generated_responses[:, 1:]], 1)
            else:
                history = generated_responses

            history = self._clean_padding_history(history)
            if self.model.config.is_encoder_decoder:
                start_position = 1
            else:
                start_position = input_length

            output = []
            for conversation_index, conversation in enumerate(conversations):
                conversation.mark_processed()
                conversation.generated_responses.append(
                    self.tokenizer.decode(
                        generated_responses[conversation_index][start_position:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                )
                conversation.set_history(history[conversation_index])
                output.append(conversation)
            if len(output) == 1:
                return output[0]
            else:
                return output

    def _parse_and_tokenize(self, inputs, **kwargs):
        """
        Parse arguments and tokenize, adding an EOS token at the end of the user input
        """
        # Parse arguments
        inputs = self.tokenizer(inputs, add_special_tokens=False, padding=False).get("input_ids", [])
        for input in inputs:
            input.append(self.tokenizer.eos_token_id)
        return inputs

    def _clean_padding_history(self, generated_tensor) -> List[List[int]]:
        """
        Cleans the padding history. Padding may be generated in two places when multiple conversations are provided as
        an input:

            - at the end of the concatenated history and new user input, so that all input to the model have the same
              length
            - at the end of the generated response, as some responses will be longer than others
        This method cleans up these padding token so that the history for each conversation is not impacted by the
        batching process.
        """
        outputs = []
        for sequence in generated_tensor:
            sequence_tokens = []
            is_previous_pad = False
            for token in sequence:
                if token == self.tokenizer.pad_token_id:
                    if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                        continue
                    if is_previous_pad:
                        continue
                    else:
                        is_previous_pad = True
                else:
                    is_previous_pad = False
                if self.framework == "pt":
                    sequence_tokens.append(token.item())
                else:
                    sequence_tokens.append(int(token.numpy()))

            outputs.append(sequence_tokens)
        return outputs

    def _concat_inputs_history(self, inputs: List[List[int]], histories: List[Optional[List[int]]], max_length: int):
        """
        Builds an input prepended by the history for this conversation, allowing multi-turn conversation with context
        """
        outputs = []
        for new_input, history in zip(inputs, histories):
            if history is not None:
                new_input = history + new_input
            if len(new_input) > max_length - self.min_length_for_response:
                cutoff_eos_index = 0
                while len(new_input) - cutoff_eos_index > max_length - self.min_length_for_response:
                    if cutoff_eos_index >= len(new_input):
                        break
                    cutoff_eos_index = new_input[cutoff_eos_index:].index(self.tokenizer.eos_token_id)
                    if cutoff_eos_index == 0 or cutoff_eos_index == len(new_input) - 1:
                        break
                    else:
                        new_input = new_input[cutoff_eos_index + 1 :]
            outputs.append(new_input)
        padded_outputs = self.tokenizer.pad(
            {"input_ids": outputs}, padding="longest", return_attention_mask=True, return_tensors=self.framework
        )
        return padded_outputs
