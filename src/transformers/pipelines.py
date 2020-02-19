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
from os.path import abspath, exists
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig
from .configuration_distilbert import DistilBertConfig
from .configuration_roberta import RobertaConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLMConfig
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

    def __call__(self, *args, **kwargs):
        if "X" in kwargs:
            return kwargs["X"]
        elif "data" in kwargs:
            return kwargs["data"]
        elif len(args) == 1:
            if isinstance(args[0], list):
                return args[0]
            else:
                return [args[0]]
        elif len(args) > 1:
            return list(args)
        raise ValueError("Unable to infer the format of the provided data (X=, data=, ...)")


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

    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False):
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
        format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False
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
    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False):
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
    def __init__(self, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False):
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
        **model**: ``(str, PretrainedModel, TFPretrainedModel)``:
            Reference to the model to use through this pipeline.

        **tokenizer**: ``(str, PreTrainedTokenizer)``:
            Reference to the tokenizer to use through this pipeline.

        **args_parser**: ``ArgumentHandler``:
            Reference to the object in charge of parsing supplied pipeline parameters.

        **device**: ``int``:
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.

        **binary_output** ``bool`` (default: False):
            Flag indicating if the output the pipeline should happen in a binary format (i.e. pickle) or as raw text.

    Return:
        Pipeline returns list or dictionary depending on:
         - Does the user provided multiple sample
         - The pipeline expose multiple fields in the output object

    Examples:
        nlp = pipeline('ner')
        nlp = pipeline('ner', model='...', config='...', tokenizer='...')
        nlp = NerPipeline(model='...', config='...', tokenizer='...')
        nlp = QuestionAnsweringPipeline(model=AutoModel.from_pretrained('...'), tokenizer='...')
    """

    default_input_names = None

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
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

    def inputs_for_model(self, features: Union[dict, List[dict]]) -> Dict:
        """
        Generates the input dictionary with model-specific parameters.

        Returns:
            dict holding all the required parameters for model's forward
        """
        args = ["input_ids", "attention_mask"]

        if not isinstance(self.model.config, (DistilBertConfig, XLMConfig, RobertaConfig)):
            args += ["token_type_ids"]

        # PR #1548 (CLI) There is an issue with attention_mask
        # if 'xlnet' in model_type or 'xlm' in model_type:
        #     args += ['cls_index', 'p_mask']

        if isinstance(features, dict):
            return {k: features[k] for k in args}
        else:
            return {k: [feature[k] for feature in features] for k in args}

    def _parse_and_tokenize(self, *texts, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*texts, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            inputs, add_special_tokens=True, return_tensors=self.framework, max_length=self.tokenizer.max_len
        )

        # Filter out features not available on specific models
        inputs = self.inputs_for_model(inputs)

        return inputs

    def __call__(self, *texts, **kwargs):
        inputs = self._parse_and_tokenize(*texts, **kwargs)
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
                predictions = self.model(inputs, training=False)[0]
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
    Feature extraction pipeline using Model head.
    """

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).tolist()


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForTextClassification head.
    """

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1)
        return [{"label": self.model.config.id2label[item.argmax()], "score": item.max()} for item in scores]


class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using ModelWithLMHead head.
    """

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        topk=5,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
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
                result.append({"sequence": self.tokenizer.decode(tokens), "score": v, "token": p})

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results


class NerPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels

    def __call__(self, *texts, **kwargs):
        inputs = self._args_parser(*texts, **kwargs)
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
                    entities = self.model(tokens)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            answer = []
            for idx, label_idx in enumerate(labels_idx):
                if self.model.config.id2label[label_idx] not in self.ignore_labels:
                    answer += [
                        {
                            "word": self.tokenizer.decode([int(input_ids[idx])]),
                            "score": score[idx][label_idx].item(),
                            "entity": self.model.config.id2label[label_idx],
                        }
                    ]

            # Append
            answers += [answer]
        if len(answers) == 1:
            return answers[0]
        return answers


TokenClassificationPipeline = NerPipeline


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
    Question Answering pipeline using ModelForQuestionAnswering head.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model,
        tokenizer: Optional[PreTrainedTokenizer],
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=device,
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

    def __call__(self, *texts, **kwargs):
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

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*texts, **kwargs)
        features_list = [
            squad_convert_examples_to_features(
                [example],
                self.tokenizer,
                kwargs["max_seq_len"],
                kwargs["doc_stride"],
                kwargs["max_question_len"],
                False,
            )
            for example in examples
        ]
        all_answers = []
        for features, example in zip(features_list, examples):
            fw_args = self.inputs_for_model([f.__dict__ for f in features])

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

                # TODO : What happens if not possible
                # Mask CLS
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
        return {"answer": " ".join(words), "start": max(0, char_start_idx), "end": min(len(text), char_end_idx)}


# Register all the supported task here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel if is_tf_available() else None,
        "pt": AutoModel if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"},
            "config": None,
            "tokenizer": "distilbert-base-cased",
        },
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
            "config": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer": "distilbert-base-uncased",
        },
    },
    "ner": {
        "impl": NerPipeline,
        "tf": TFAutoModelForTokenClassification if is_tf_available() else None,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
            "config": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "tokenizer": "bert-large-cased",
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": TFAutoModelForQuestionAnswering if is_tf_available() else None,
        "pt": AutoModelForQuestionAnswering if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad", "tf": "distilbert-base-cased-distilled-squad"},
            "config": None,
            "tokenizer": ("distilbert-base-cased", {"use_fast": False}),
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilroberta-base", "tf": "distilroberta-base"},
            "config": None,
            "tokenizer": ("distilroberta-base", {"use_fast": False}),
        },
    },
}


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    modelcard: Optional[Union[str, ModelCard]] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.
    Pipeline are made of:
        A Tokenizer instance in charge of mapping raw textual input to token
        A Model instance
        Some (optional) post processing for enhancing model's output

    Examples:
        pipeline('sentiment-analysis')
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
        pipeline('ner', model=AutoModel.from_pretrained(...), tokenizer=AutoTokenizer.from_pretrained(...)
        pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', tokenizer='bert-base-cased')
        pipeline('ner', model='https://...pytorch-model.bin', config='https://...config.json', tokenizer='bert-base-cased')
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    framework = framework or get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    task, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        models, config, tokenizer = tuple(targeted_task["default"].values())
        model = models[framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str) and model in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            tokenizer = model
        elif isinstance(config, str) and config in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/url/shortcut name to a pretrained tokenizer."
            )

    # Try to infer modelcard from model or config name (if provided as str)
    if modelcard is None:
        # Try to fallback on one of the provided string for model or config (will replace the suffix)
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

    return task(model=model, tokenizer=tokenizer, modelcard=modelcard, framework=framework, **kwargs)
