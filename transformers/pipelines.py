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
from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import json
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import groupby
from os.path import abspath, exists
from typing import Union, Optional, Tuple, List, Dict

import numpy as np

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PretrainedConfig, \
    SquadExample, squad_convert_examples_to_features, is_tf_available, is_torch_available, logger

if is_tf_available():
    from transformers import TFAutoModel, TFAutoModelForSequenceClassification, \
        TFAutoModelForQuestionAnswering, TFAutoModelForTokenClassification

if is_torch_available():
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, \
        AutoModelForQuestionAnswering, AutoModelForTokenClassification


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
        if 'X' in kwargs:
            return kwargs['X']
        elif 'data' in kwargs:
            return kwargs['data']
        elif len(args) == 1:
            if isinstance(args[0], list):
                return args[0]
            else:
                return [args[0]]
        elif len(args) > 1:
            return list(args)
        raise ValueError('Unable to infer the format of the provided data (X=, data=, ...)')


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing.
    Supported data formats currently includes:
     - JSON
     - CSV

    PipelineDataFormat also includes some utilities to work with multi-columns like mapping from datasets columns
    to pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.
    """
    SUPPORTED_FORMATS = ['json', 'csv', 'pipe']

    def __init__(self, output: Optional[str], path: Optional[str], column: Optional[str]):
        self.output = output
        self.path = path
        self.column = column.split(',') if column else ['']
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split('=')) if '=' in c else (c, c) for c in self.column]

        if output is not None:
            if exists(abspath(self.output)):
                raise OSError('{} already exists on disk'.format(self.output))

            if not exists(abspath(self.path)):
                raise OSError('{} doesnt exist on disk'.format(self.path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: dict):
        raise NotImplementedError()

    @staticmethod
    def from_str(name: str, output: Optional[str], path: Optional[str], column: Optional[str]):
        if name == 'json':
            return JsonPipelineDataFormat(output, path, column)
        elif name == 'csv':
            return CsvPipelineDataFormat(output, path, column)
        elif name == 'pipe':
            return PipedPipelineDataFormat(output, path, column)
        else:
            raise KeyError('Unknown reader {} (Available reader are json/csv/pipe)'.format(name))


class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(self, output: Optional[str], path: Optional[str], column: Optional[str]):
        super().__init__(output, path, column)

    def __iter__(self):
        with open(self.path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        with open(self.output, 'w') as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(self, output: Optional[str], path: Optional[str], column: Optional[str]):
        super().__init__(output, path, column)

        with open(path, 'r') as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        with open(self.output, 'w') as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process.
    For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}
    """
    def __iter__(self):
        import sys
        for line in sys.stdin:

            # Split for multi-columns
            if '\t' in line:

                line = line.split('\t')
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                print(line)
                yield line

    def save(self, data: dict):
        print(data)


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
    """
    def __init__(self, model, tokenizer: PreTrainedTokenizer = None,
                 args_parser: ArgumentHandler = None, device: int = -1, **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._args_parser = args_parser or DefaultArgumentHandler()

        # Special handling
        if self.device >= 0 and not is_tf_available():
            self.model = self.model.to('cuda:{}'.format(self.device))

    def save_pretrained(self, save_directory):
        """
        Save the pipeline's model and tokenizer to the specified save_directory
        """
        if not os.path.isdir(save_directory):
            logger.error("Provided path ({}) should be a directory".format(save_directory))
            return

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        Se
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
        if is_tf_available():
            import tensorflow as tf
            with tf.device('/CPU:0' if self.device == -1 else '/device:GPU:{}'.format(self.device)):
                yield
        else:
            import torch
            if self.device >= 0:
                torch.cuda.set_device(self.device)

            yield

    def inputs_for_model(self, features: Union[dict, List[dict]]) -> Dict:
        """
        Generates the input dictionary with model-specific parameters.

        Returns:
            dict holding all the required parameters for model's forward
        """
        args = ['input_ids', 'attention_mask']
        model_type = type(self.model).__name__.lower()

        if 'distilbert' not in model_type and 'xlm' not in model_type:
            args += ['token_type_ids']

        if 'xlnet' in model_type or 'xlm' in model_type:
            args += ['cls_index', 'p_mask']

        if isinstance(features, dict):
            return {k: features[k] for k in args}
        else:
            return {k: [feature[k] for feature in features] for k in args}

    def __call__(self, *texts, **kwargs):
        # Parse arguments
        inputs = self._args_parser(*texts, **kwargs)

        # Encode for forward
        with self.device_placement():
            inputs = self.tokenizer.batch_encode_plus(
                inputs, add_special_tokens=True,
                return_tensors='tf' if is_tf_available() else 'pt',
                # max_length=self.model.config.max_position_embedding
                max_length=511
            )

            # Filter out features not available on specific models
            inputs = self.inputs_for_model(inputs)
            return self._forward(inputs)

    def _forward(self, inputs):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
        Returns:
            Numpy array
        """
        if is_tf_available():
            # TODO trace model
            predictions = self.model(inputs)[0]
        else:
            import torch
            with torch.no_grad():
                predictions = self.model(**inputs)[0]

        return predictions.numpy()


class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using Model head.
    """
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).tolist()


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForTextClassification head.
    """

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1)
        return [{'label': self.model.config.id2label[item.argmax()], 'score': item.max()} for item in scores]


class NerPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head.
    """

    def __call__(self, *texts, **kwargs):
        inputs, answers = self._args_parser(*texts, **kwargs), []
        for sentence in inputs:

            # Ugly token to word idx mapping (for now)
            token_to_word, words = [], sentence.split(' ')
            for i, w in enumerate(words):
                tokens = self.tokenizer.tokenize(w)
                token_to_word += [i] * len(tokens)

            # Manage correct placement of the tensors
            with self.device_placement():
                tokens = self.tokenizer.encode_plus(
                    sentence, return_attention_mask=False,
                    return_tensors='tf' if is_tf_available() else 'pt',
                    max_length=512
                )

                # Forward
                if is_torch_available():
                    with torch.no_grad():
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                else:
                    entities = self.model(tokens)[0][0].numpy()

            # Normalize scores
            answer, token_start = [], 1
            for idx, word in groupby(token_to_word):

                # Sum log prob over token, then normalize across labels
                score = np.exp(entities[token_start]) / np.exp(entities[token_start]).sum(-1, keepdims=True)
                label_idx = score.argmax()

                answer += [{
                    'word': words[idx],
                    'score': score[label_idx].item(),
                    'entity': self.model.config.id2label[label_idx]
                }]

                # Update token start
                token_start += len(list(word))

            # Append
            answers += [answer]
        return answers


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
                kwargs['X'] = args[0]
            else:
                kwargs['X'] = list(args)

            # Generic compatibility with sklearn and Keras
            # Batched data
        if 'X' in kwargs or 'data' in kwargs:
            data = kwargs['X'] if 'X' in kwargs else kwargs['data']

            if not isinstance(data, list):
                data = [data]

            for i, item in enumerate(data):
                if isinstance(item, dict):
                    if any(k not in item for k in ['question', 'context']):
                        raise KeyError('You need to provide a dictionary with keys {question:..., context:...}')
                    data[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif isinstance(item, SquadExample):
                    continue
                else:
                    raise ValueError(
                        '{} argument needs to be of type (list[SquadExample | dict], SquadExample, dict)'
                            .format('X' if 'X' in kwargs else 'data')
                    )
            inputs = data

            # Tabular input
        elif 'question' in kwargs and 'context' in kwargs:
            if isinstance(kwargs['question'], str):
                kwargs['question'] = [kwargs['question']]

            if isinstance(kwargs['context'], str):
                kwargs['context'] = [kwargs['context']]

            inputs = [QuestionAnsweringPipeline.create_sample(q, c) for q, c in zip(kwargs['question'], kwargs['context'])]
        else:
            raise ValueError('Unknown arguments {}'.format(kwargs))

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using ModelForQuestionAnswering head.
    """

    @staticmethod
    def create_sample(question: Union[str, List[str]], context: Union[str, List[str]]) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the SquadExample/SquadFeatures internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample(s).
        We currently support extractive question answering.
        Args:
             question: (str, List[str]) The question to be ask for the associated context
             context: (str, List[str]) The context in which we will look for the answer.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __init__(self, model, tokenizer: Optional[PreTrainedTokenizer], device: int = -1, **kwargs):
        super().__init__(model, tokenizer, args_parser=QuestionAnsweringArgumentHandler(),
                         device=device, **kwargs)

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
        kwargs.setdefault('topk', 1)
        kwargs.setdefault('doc_stride', 128)
        kwargs.setdefault('max_answer_len', 15)
        kwargs.setdefault('max_seq_len', 384)
        kwargs.setdefault('max_question_len', 64)

        if kwargs['topk'] < 1:
            raise ValueError('topk parameter should be >= 1 (got {})'.format(kwargs['topk']))

        if kwargs['max_answer_len'] < 1:
            raise ValueError('max_answer_len parameter should be >= 1 (got {})'.format(kwargs['max_answer_len']))

        # Convert inputs to features
        examples = self._args_parser(*texts, **kwargs)
        features = squad_convert_examples_to_features(examples, self.tokenizer, kwargs['max_seq_len'], kwargs['doc_stride'], kwargs['max_question_len'], False)
        fw_args = self.inputs_for_model([f.__dict__ for f in features])

        # Manage tensor allocation on correct device
        with self.device_placement():
            if is_tf_available():
                import tensorflow as tf
                fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
                start, end = self.model(fw_args)
                start, end = start.numpy(), end.numpy()
            else:
                import torch
                with torch.no_grad():
                    # Retrieve the score for the context tokens only (removing question tokens)
                    fw_args = {k: torch.tensor(v) for (k, v) in fw_args.items()}
                    start, end = self.model(**fw_args)
                    start, end = start.cpu().numpy(), end.cpu().numpy()

        answers = []
        for (example, feature, start_, end_) in zip(examples, features, start, end):
            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_) / np.sum(np.exp(start_))
            end_ = np.exp(end_) / np.sum(np.exp(end_))

            # Mask padding and question
            start_, end_ = start_ * np.abs(np.array(feature.p_mask) - 1), end_ * np.abs(np.array(feature.p_mask) - 1)

            # TODO : What happens if not possible
            # Mask CLS
            start_[0] = end_[0] = 0

            starts, ends, scores = self.decode(start_, end_, kwargs['topk'], kwargs['max_answer_len'])
            char_to_word = np.array(example.char_to_word_offset)

            # Convert the answer (tokens) back to the original text
            answers += [
                {
                    'score': score.item(),
                    'start': np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                    'end': np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                    'answer': ' '.join(example.doc_tokens[feature.token_to_orig_map[s]:feature.token_to_orig_map[e] + 1])
                }
                for s, e, score in zip(starts, ends, scores)
            ]

        if len(answers) == 1:
            return answers[0]
        return answers

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
        return {'answer': ' '.join(words), 'start': max(0, char_start_idx), 'end': min(len(text), char_end_idx)}


# Register all the supported task here
SUPPORTED_TASKS = {
    'feature-extraction': {
        'impl': FeatureExtractionPipeline,
        'tf': TFAutoModel if is_tf_available() else None,
        'pt': AutoModel if is_torch_available() else None,
        'default': {
            'model': 'distilbert-base-uncased',
            'config': None,
            'tokenizer': 'bert-base-uncased'
        }
    },
    'sentiment-analysis': {
        'impl': TextClassificationPipeline,
        'tf': TFAutoModelForSequenceClassification if is_tf_available() else None,
        'pt': AutoModelForSequenceClassification if is_torch_available() else None,
        'default': {
            'model': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-pytorch_model.bin',
            'config': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-config.json',
            'tokenizer': 'bert-base-uncased'
        }
    },
    'ner': {
        'impl': NerPipeline,
        'tf': TFAutoModelForTokenClassification if is_tf_available() else None,
        'pt': AutoModelForTokenClassification if is_torch_available() else None,
        'default': {
            'model': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-finetuned-conll03-english-pytorch_model.bin',
            'config': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-finetuned-conll03-english-config.json',
            'tokenizer': 'bert-base-cased'
        }
    },
    'question-answering': {
        'impl': QuestionAnsweringPipeline,
        'tf': TFAutoModelForQuestionAnswering if is_tf_available() else None,
        'pt': AutoModelForQuestionAnswering if is_torch_available() else None,
        'default': {
            'model': 'distilbert-base-uncased-distilled-squad',
            'config': None,
            'tokenizer': 'bert-base-uncased'
        }
    }
}


def pipeline(task: str, model: Optional = None,
             config: Optional[Union[str, PretrainedConfig]] = None,
             tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None, **kwargs) -> Pipeline:
    """
    Utility factory method to build a pipeline.
    Pipeline are made of:
        A Tokenizer instance in charge of mapping raw textual input to token
        A Model instance
        Some (optional) post processing for enhancing model's output

    Examples:
        pipeline('ner')
    """
    # Try to infer tokenizer from model name (if provided as str)
    if tokenizer is None:
        if model is not None and not isinstance(model, str):
            # Impossible to guest what is the right tokenizer here
            raise Exception('Tokenizer cannot be None if provided model is a PreTrainedModel instance')
        else:
            tokenizer = model

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task, allocator = targeted_task['impl'], targeted_task['tf'] if is_tf_available() else targeted_task['pt']

    # Handling for default model for the task
    if model is None:
        model, config, tokenizer = tuple(targeted_task['default'].values())

    # Allocate tokenizer
    tokenizer = tokenizer if isinstance(tokenizer, PreTrainedTokenizer) else AutoTokenizer.from_pretrained(tokenizer)

    # Special handling for model conversion
    if isinstance(model, str):
        from_tf = model.endswith('.h5') and not is_tf_available()
        from_pt = model.endswith('.bin') and not is_torch_available()

        if from_tf:
            logger.warning('Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. '
                           'Trying to load the model with PyTorch.')
        elif from_pt:
            logger.warning('Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. '
                           'Trying to load the model with Tensorflow.')
    else:
        from_tf = from_pt = False

    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    if allocator.__name__.startswith('TF'):
        model = allocator.from_pretrained(model, config=config, from_pt=from_pt)
    else:
        model = allocator.from_pretrained(model, config=config, from_tf=from_tf)
    return task(model, tokenizer, **kwargs)
