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
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .configuration_auto import AutoConfig
from .configuration_utils import PretrainedConfig
from .data import SquadExample, squad_convert_examples_to_features
from .file_utils import is_tf_available, is_torch_available
from .modelcard import ModelCard
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BasicTokenizer
from .tokenization_utils import PreTrainedTokenizer


if is_tf_available():
    import tensorflow as tf
    from .modeling_tf_auto import (
        TFAutoModel,
        TFAutoModelForSequenceClassification,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
    )

if is_torch_available():
    import torch
    from .modeling_auto import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
    )

if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .modeling_tf_utils import TFPreTrainedModel


logger = logging.getLogger(__name__)


def get_framework(model=None):
    """ Select framework (TensorFlow/PyTorch) to use.
        If both frameworks are installed and no specific model is provided, defaults to using PyTorch.
    """
    if is_tf_available() and is_torch_available() and model is not None and not isinstance(model, str):
        # Both framework are available but the user supplied a model class instance.
        # Try to guess which framework to use from the model classname
        framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"
    elif not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    else:
        # framework = 'tf' if is_tf_available() else 'pt'
        framework = "pt" if is_torch_available() else "tf"
    return framework


class ArgumentHandler(ABC):
    """
    Base interface for handling varargs for each Pipeline
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default varargs argument parser handling parameters for each Pipeline
    """

    @staticmethod
    def handle_kwargs(kwargs: Dict) -> List:
        if len(kwargs) == 1:
            output = list(kwargs.values())
        else:
            output = list(chain(kwargs.values()))

        return DefaultArgumentHandler.handle_args(output)

    @staticmethod
    def handle_args(args: Sequence[Any]) -> List[str]:

        # Only one argument, let's do case by case
        if len(args) == 1:
            if isinstance(args[0], str):
                return [args[0]]
            elif not isinstance(args[0], list):
                return list(args)
            else:
                return args[0]

        # Multiple arguments (x1, x2, ...)
        elif len(args) > 1:
            if all([isinstance(arg, str) for arg in args]):
                return list(args)

            # If not instance of list, then it should instance of iterable
            elif isinstance(args, Iterable):
                return list(chain.from_iterable(chain(args)))
            else:
                raise ValueError(
                    "Invalid input type {}. Pipeline supports Union[str, Iterable[str]]".format(type(args))
                )
        else:
            return []

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0 and len(args) > 0:
            raise ValueError("Pipeline cannot handle mixed args and kwargs")

        if len(kwargs) > 0:
            return DefaultArgumentHandler.handle_kwargs(kwargs)
        else:
            return DefaultArgumentHandler.handle_args(args)


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing.
    Supported data formats currently includes:
     - JSON
     - CSV
     - stdin/stdout (pipe)

    PipelineDataFormat also includes some utilities to work with multi-columns like mapping from datasets columns
    to pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
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
    def save(self, data: dict):
        """
        Save the provided data object with the representation for the current `DataFormat`.
        :param data: data to store
        :return:
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.
        :param data: data to store
        :return: (str) Path where the data has been saved
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
    ):
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError("Unknown reader {} (Available reader are json/csv/pipe)".format(format))


class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
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
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False,
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
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process.
    For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}
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


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:
        Input -> Tokenization -> Model Inference -> Post-Processing (Task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument. Users can specify
    device argument as an integer, -1 meaning "CPU", >= 0 referring the CUDA device ordinal.

    Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') outputs large
    tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the binary_output constructor argument. If set to True, the output will be stored in the
    pickle format.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e. pickle) or as raw text.

    Return:
        :obj:`List` or :obj:`Dict`:
        Pipeline returns list or dictionary depending on:

         - Whether the user supplied multiple samples
         - Whether the pipeline exposes multiple fields in the output object
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
            framework = get_framework()

        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.device = device if framework == "tf" else torch.device("cpu" if device < 0 else "cuda:{}".format(device))
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory):
        """
        Save the pipeline's model and tokenizer to the specified save_directory
        """
        if not os.path.isdir(save_directory):
            logger.error("Provided path ({}) should be a directory".format(save_directory))
            return

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
        example:
            # Explicitly ask for tensor allocation on CUDA device :0
            nlp = pipeline(..., device=0)
            with nlp.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = nlp(...)
        Returns:
            Context manager
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
        :param inputs:
        :return:
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def _parse_and_tokenize(self, *args, pad_to_max_length=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            pad_to_max_length=pad_to_max_length,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array.
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


class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using Model head. This pipeline extracts the hidden states from the base transformer,
    which can be used as features in downstream tasks.

    This feature extraction pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "feature-extraction", for extracting features of a sequence.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    `huggingface.co/models <https://huggingface.co/models>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).tolist()


class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any ModelWithLMHead head. This pipeline predicts the words that will follow a specified text prompt.

    This language generation pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "text-generation", for generating text from a specified prompt.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling objective,
    which includes the uni-directional models in the library (e.g. gpt2).
    See the list of available community models on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=lm-head>`__.
    """

    # Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    ALLOWED_MODELS = [
        "XLNetLMHeadModel",
        "TransfoXLLMHeadModel",
        "ReformerModelWithLMHead",
        "GPT2LMHeadModel",
        "OpenAIGPTLMHeadModel",
        "CTRLLMHeadModel",
        "TFXLNetLMHeadModel",
        "TFTransfoXLLMHeadModel",
        "TFGPT2LMHeadModel",
        "TFOpenAIGPTLMHeadModel",
        "TFCTRLLMHeadModel",
    ]

    def __call__(
        self, *args, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        if self.model.__class__.__name__ not in self.ALLOWED_MODELS:
            raise NotImplementedError(
                "Generation is currently not supported for {}. Please select a model from {} for generation.".format(
                    self.model.__class__.__name__, self.ALLOWED_MODELS
                )
            )

        text_inputs = self._args_parser(*args)

        results = []
        for prompt_text in text_inputs:
            # Manage correct placement of the tensors
            with self.device_placement():
                if self.model.__class__.__name__ in ["XLNetLMHeadModel", "TransfoXLLMHeadModel"]:
                    inputs = self._parse_and_tokenize(
                        self.PADDING_TEXT + prompt_text, pad_to_max_length=False, add_special_tokens=False
                    )
                else:
                    inputs = self._parse_and_tokenize(prompt_text, pad_to_max_length=False, add_special_tokens=False)

                # set input_ids to None to allow empty prompt
                if inputs["input_ids"].shape[-1] == 0:
                    inputs["input_ids"] = None
                    inputs["attention_mask"] = None

                if self.framework == "pt" and inputs["input_ids"] is not None:
                    inputs = self.ensure_tensor_on_device(**inputs)

                input_ids = inputs["input_ids"]

                # Ensure that batch size = 1 (batch generation not allowed for now)
                assert (
                    input_ids is None or input_ids.shape[0] == 1
                ), "Batch generation is currently not supported. See https://github.com/huggingface/transformers/issues/3021 for more information."

                output_sequences = self.model.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL

            result = []
            for generated_sequence in output_sequences:
                generated_sequence = generated_sequence.numpy().tolist()
                record = {}
                if return_tensors:
                    record["generated_token_ids"] = generated_sequence
                if return_text:
                    # Decode text
                    text = self.tokenizer.decode(
                        generated_sequence,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )

                    # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                    if input_ids is None:
                        prompt_length = 0
                    else:
                        prompt_length = len(
                            self.tokenizer.decode(
                                input_ids[0],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                            )
                        )

                    record["generated_text"] = prompt_text + text[prompt_length:]

                result.append(record)
            results += [result]

        if len(results) == 1:
            return results[0]

        return results


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForSequenceClassification head. See the
    `sequence classification usage <../usage.html#sequence-classification>`__ examples for more information.

    This text classification pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "sentiment-analysis", for classifying sequences according to positive or negative sentiments.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=text-classification>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]


class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using ModelWithLMHead head. See the
    `masked language modeling usage <../usage.html#masked-language-modeling>`__ examples for more information.

    This mask filling pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "fill-mask", for predicting masked tokens in a sequence.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=lm-head>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        topk=5,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

        self.topk = topk

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        outputs = self._forward(inputs, return_tensors=True)

        results = []
        batch_size = outputs.shape[0] if self.framework == "tf" else outputs.size(0)

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            if self.framework == "tf":
                masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy().item()
                logits = outputs[i, masked_index, :]
                probs = tf.nn.softmax(logits)
                topk = tf.math.top_k(probs, k=self.topk)
                values, predictions = topk.values.numpy(), topk.indices.numpy()
            else:
                masked_index = (input_ids == self.tokenizer.mask_token_id).nonzero().item()
                logits = outputs[i, masked_index, :]
                probs = logits.softmax(dim=0)
                values, predictions = probs.topk(self.topk)

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens),
                        "score": v,
                        "token": p,
                        "token_str": self.tokenizer.convert_ids_to_tokens(p),
                    }
                )

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results


class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head. See the
    `named entity recognition usage <../usage.html#named-entity-recognition>`__ examples for more information.

    This token recognition pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "ner", for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous.

    The models that this pipeline can use are models that have been fine-tuned on a token classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=token-classification>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities

    def __call__(self, *args, **kwargs):
        inputs = self._args_parser(*args, **kwargs)
        answers = []
        for sentence in inputs:

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer.encode_plus(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    max_length=self.tokenizer.max_len,
                )

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens.data)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            entity_groups = []
            entity_group_disagg = []
            # Filter to labels not in `self.ignore_labels`
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if self.model.config.id2label[label_idx] not in self.ignore_labels
            ]

            for idx, label_idx in filtered_labels_idx:

                entity = {
                    "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                    "score": score[idx][label_idx].item(),
                    "entity": self.model.config.id2label[label_idx],
                    "index": idx,
                }
                last_idx, _ = filtered_labels_idx[-1]
                if self.grouped_entities:
                    if not entity_group_disagg:
                        entity_group_disagg += [entity]
                        if idx == last_idx:
                            entity_groups += [self.group_entities(entity_group_disagg)]
                        continue

                    # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
                    if (
                        entity["entity"] == entity_group_disagg[-1]["entity"]
                        and entity["index"] == entity_group_disagg[-1]["index"] + 1
                    ):
                        entity_group_disagg += [entity]
                        # Group the entities at the last entity
                        if idx == last_idx:
                            entity_groups += [self.group_entities(entity_group_disagg)]
                    # If the current entity is different from the previous entity, aggregate the disaggregated entity group
                    else:
                        entity_groups += [self.group_entities(entity_group_disagg)]
                        entity_group_disagg = [entity]

                entities += [entity]

            # Append
            if self.grouped_entities:
                answers += [entity_groups]
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_entities(self, entities):
        """
        Returns grouped entities
        """
        # Get the last entity in the entity group
        entity = entities[-1]["entity"]
        scores = np.mean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
        }
        return entity_group


NerPipeline = TokenClassificationPipeline


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped
    to internal SquadExample / SquadFeature structures.

    QuestionAnsweringArgumentHandler manages all the possible to create SquadExample from the command-line supplied
    arguments.
    """

    def __call__(self, *args, **kwargs):
        # Position args, handling is sensibly the same as X and data, so forwarding to avoid duplicating
        if args is not None and len(args) > 0:
            if len(args) == 1:
                kwargs["X"] = args[0]
            else:
                kwargs["X"] = list(args)

        # Generic compatibility with sklearn and Keras
        # Batched data
        if "X" in kwargs or "data" in kwargs:
            inputs = kwargs["X"] if "X" in kwargs else kwargs["data"]

            if isinstance(inputs, dict):
                inputs = [inputs]
            else:
                # Copy to avoid overriding arguments
                inputs = [i for i in inputs]

            for i, item in enumerate(inputs):
                if isinstance(item, dict):
                    if any(k not in item for k in ["question", "context"]):
                        raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")

                    inputs[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif not isinstance(item, SquadExample):
                    raise ValueError(
                        "{} argument needs to be of type (list[SquadExample | dict], SquadExample, dict)".format(
                            "X" if "X" in kwargs else "data"
                        )
                    )

            # Tabular input
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], str):
                kwargs["question"] = [kwargs["question"]]

            if isinstance(kwargs["context"], str):
                kwargs["context"] = [kwargs["context"]]

            inputs = [
                QuestionAnsweringPipeline.create_sample(q, c) for q, c in zip(kwargs["question"], kwargs["context"])
            ]
        else:
            raise ValueError("Unknown arguments {}".format(kwargs))

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using ModelForQuestionAnswering head. See the
    `question answering usage <../usage.html#question-answering>`__ examples for more information.

    This question answering can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "question-answering", for answering questions given a context.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=question-answering>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=device,
            task=task,
            **kwargs,
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the SquadExample/SquadFeatures internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample(s).
        We currently support extractive question answering.
        Arguments:
             question: (str, List[str]) The question to be ask for the associated context
             context: (str, List[str]) The context in which we will look for the answer.

        Returns:
            SquadExample initialized with the corresponding question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *args, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list = [
            squad_convert_examples_to_features(
                [example],
                self.tokenizer,
                kwargs["max_seq_len"],
                kwargs["doc_stride"],
                kwargs["max_question_len"],
                False,
                tqdm_enabled=False,
            )
            for example in examples
        ]
        all_answers = []
        for features, example in zip(features_list, examples):
            model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            # Manage tensor allocation on correct device
            with self.device_placement():
                if self.framework == "tf":
                    fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
                    start, end = self.model(fw_args)
                    start, end = start.numpy(), end.numpy()
                else:
                    with torch.no_grad():
                        # Retrieve the score for the context tokens only (removing question tokens)
                        fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                        start, end = self.model(**fw_args)
                        start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_) / np.sum(np.exp(start_))
                end_ = np.exp(end_) / np.sum(np.exp(end_))

                # Mask padding and question
                start_, end_ = (
                    start_ * np.abs(np.array(feature.p_mask) - 1),
                    end_ * np.abs(np.array(feature.p_mask) - 1),
                )

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                start_[0] = end_[0] = 0

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                        "answer": " ".join(
                            example.doc_tokens[feature.token_to_orig_map[s] : feature.token_to_orig_map[e] + 1]
                        ),
                    }
                    for s, e, score in zip(starts, ends, scores)
                ]

            if kwargs["handle_impossible_answer"]:
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
        """
        Take the output of any QuestionAnswering head and will generate probalities for each span to be
        the actual answer.
        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start: numpy array, holding individual start probabilities for each token
            end: numpy array, holding individual end probabilities for each token
            topk: int, indicates how many possible answer span(s) to extract from the model's output
            max_answer_len: int, maximum size of the answer to extract from the model's output
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def span_to_answer(self, text: str, start: int, end: int):
        """
        When decoding from token probalities, this method maps token indexes to actual word in
        the initial context.

        Args:
            text: str, the actual context to extract the answer from
            start: int, starting answer token index
            end: int, ending answer token index

        Returns:
            dict: {'answer': str, 'start': int, 'end': int}
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }


class SummarizationPipeline(Pipeline):
    """
    Summarize news articles and other documents

    Usage::

        # use bart in pytorch
        summarizer = pipeline("summarization")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)

        # use t5 in tf
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)

    The models that this pipeline can use are models that have been fine-tuned on a summarization task,
    which is currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__.

    Arguments:
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`, a string
            checkpoint identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.

            If :obj:`None`, the default of the pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If :obj:`None`, the default of the pipeline will be loaded.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __call__(
        self, *documents, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        r"""
        Args:
            *documents: (list of strings) articles to be summarized
            return_text: (bool, default=True) whether to add a decoded "summary_text" to each result
            return_tensors: (bool, default=False) whether to return the raw "summary_token_ids" to each result

            clean_up_tokenization_spaces: (`optional`) bool whether to include extra spaces in the output
            **generate_kwargs: extra kwargs passed to `self.model.generate`_

        Returns:
            list of dicts with 'summary_text' and/or 'summary_token_ids' for each document_to_summarize

        .. _`self.model.generate`:
            https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration.generate

        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"
        assert len(documents) > 0, "Please provide a document to summarize"

        if self.framework == "tf" and "BartForConditionalGeneration" in self.model.__class__.__name__:
            raise NotImplementedError(
                "Tensorflow is not yet supported for Bart. Please consider using T5, e.g. `t5-base`"
            )

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""

        if isinstance(documents[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"

            documents = ([prefix + document for document in documents[0]],)
            pad_to_max_length = True

        elif isinstance(documents[0], str):
            documents = (prefix + documents[0],)
            pad_to_max_length = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    documents[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*documents, pad_to_max_length=pad_to_max_length)

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
                inputs["input_ids"], attention_mask=inputs["attention_mask"], **generate_kwargs,
            )

            results = []
            for summary in summaries:
                record = {}
                if return_tensors:
                    record["summary_token_ids"] = summary
                if return_text:
                    record["summary_text"] = self.tokenizer.decode(
                        summary, skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


class TranslationPipeline(Pipeline):
    """
    Translates from one language to another.

    Usage::
        en_fr_translator = pipeline("translation_en_to_fr")
        en_fr_translator("How old are you?")

    The models that this pipeline can use are models that have been fine-tuned on a translation task,
    currently: "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=translation>`__.

    Arguments:
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`, a string
            checkpoint identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
            If :obj:`None`, the default of the pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.
            If :obj:`None`, the default of the pipeline will be loaded.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __call__(
        self, *args, return_tensors=False, return_text=True, clean_up_tokenization_spaces=False, **generate_kwargs
    ):
        r"""
        Args:
            *args: (list of strings) texts to be translated
            return_text: (bool, default=True) whether to add a decoded "translation_text" to each result
            return_tensors: (bool, default=False) whether to return the raw "translation_token_ids" to each result

            **generate_kwargs: extra kwargs passed to `self.model.generate`_

        Returns:
            list of dicts with 'translation_text' and/or 'translation_token_ids' for each text_to_translate
        .. _`self.model.generate`:
            https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""

        if isinstance(args[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            args = ([prefix + text for text in args[0]],)
            pad_to_max_length = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            pad_to_max_length = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    args[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, pad_to_max_length=pad_to_max_length)

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
                inputs["input_ids"], attention_mask=inputs["attention_mask"], **generate_kwargs,
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


# Register all the supported tasks here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel if is_tf_available() else None,
        "pt": AutoModel if is_torch_available() else None,
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
    },
    "ner": {
        "impl": TokenClassificationPipeline,
        "tf": TFAutoModelForTokenClassification if is_tf_available() else None,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": TFAutoModelForQuestionAnswering if is_tf_available() else None,
        "pt": AutoModelForQuestionAnswering if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad", "tf": "distilbert-base-cased-distilled-squad"},
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "distilroberta-base", "tf": "distilroberta-base"}},
    },
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "facebook/bart-large-cnn", "tf": "t5-small"}},
    },
    "translation_en_to_fr": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
    },
    "translation_en_to_de": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
    },
    "translation_en_to_ro": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {"model": {"pt": "gpt2", "tf": "gpt2"}},
    },
}


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.

    Pipeline are made of:

        - A Tokenizer instance in charge of mapping raw textual input to token
        - A Model instance
        - Some (optional) post processing for enhancing model's output


    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - "feature-extraction": will return a :class:`~transformers.FeatureExtractionPipeline`
            - "sentiment-analysis": will return a :class:`~transformers.TextClassificationPipeline`
            - "ner": will return a :class:`~transformers.TokenClassificationPipeline`
            - "question-answering": will return a :class:`~transformers.QuestionAnsweringPipeline`
            - "fill-mask": will return a :class:`~transformers.FillMaskPipeline`
            - "summarization": will return a :class:`~transformers.SummarizationPipeline`
            - "translation_xx_to_yy": will return a :class:`~transformers.TranslationPipeline`
            - "text-generation": will return a :class:`~transformers.TextGenerationPipeline`
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`,
            a model identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.

            If :obj:`None`, the default for this pipeline will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`, defaults to :obj:`None`):
            The configuration that will be used by the pipeline to instantiate the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If :obj:`None`, the default for this pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If :obj:`None`, the default for this pipeline will be loaded.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.

    Returns:
        :class:`~transformers.Pipeline`: Class inheriting from :class:`~transformers.Pipeline`, according to
        the task.

    Examples::

        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')

        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        # Named entity recognition pipeline, passing in a specific model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pipeline('ner', model=model, tokenizer=tokenizer)
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    framework = framework or get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    task_class, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"][framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    if isinstance(model, str):
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        elif framework == "tf" and model.endswith(".bin"):
            model_kwargs["from_pt"] = True
            logger.warning(
                "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                "Trying to load the model with Tensorflow."
            )
        model = model_class.from_pretrained(model, config=config, **model_kwargs)

    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard, framework=framework, task=task, **kwargs)
