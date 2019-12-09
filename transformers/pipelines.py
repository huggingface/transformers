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

import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List, Dict

import numpy as np

from transformers import is_tf_available, is_torch_available, logger, AutoTokenizer, PreTrainedTokenizer

if is_tf_available():
    from transformers import TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering

if is_torch_available():
    from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering


class Pipeline(ABC):
    def __init__(self, model, tokenizer: PreTrainedTokenizer = None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    @abstractmethod
    def from_config(cls, model, tokenizer: PreTrainedTokenizer, **kwargs):
        raise NotImplementedError()

    def save_pretrained(self, save_directory):
        if not os.path.isdir(save_directory):
            logger.error("Provided path ({}) should be a directory".format(save_directory))
            return

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def transform(self, *texts, **kwargs):
        # Generic compatibility with sklearn and Keras
        return self(*texts, **kwargs)

    def predict(self, *texts, **kwargs):
        # Generic compatibility with sklearn and Keras
        return self(*texts, **kwargs)

    @abstractmethod
    def __call__(self, *texts, **kwargs):
        raise NotImplementedError()


class TextClassificationPipeline(Pipeline):
    def __init__(self, model, tokenizer: PreTrainedTokenizer, nb_classes: int = 2):
        super().__init__(model, tokenizer)

        if nb_classes < 2:
            raise Exception('Invalid parameter nb_classes. int >= 2 is required (got: {})'.format(nb_classes))
        self._nb_classes = nb_classes

    @classmethod
    def from_config(cls, model, tokenizer: PreTrainedTokenizer, **kwargs):
        return cls(model, tokenizer, **kwargs)

    def __call__(self, *texts, **kwargs):
        # Generic compatibility with sklearn and Keras
        if 'X' in kwargs and not texts:
            texts = kwargs.pop('X')

        inputs = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, return_tensors='tf' if is_tf_available() else 'pt'
        )

        special_tokens_mask = inputs.pop('special_tokens_mask')

        if is_tf_available():
            # TODO trace model
            predictions = self.model(**inputs)[0]
        else:
            import torch
            with torch.no_grad():
                predictions = self.model(**inputs)[0]

        return predictions.numpy().tolist()


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeling involving Tokenization and Inference.
    TODO:
     - top-k answers
     - return start/end chars
     - return score
    """

    def __init__(self, model, tokenizer: Optional[PreTrainedTokenizer]):
        super().__init__(model, tokenizer)

    @staticmethod
    def create_sample(question: Union[str, List[str]], context: Union[str, List[str]]) -> Union[dict, List[Dict]]:
        is_list = isinstance(question, list)

        if is_list:
            return [{'question': q, 'context': c} for q, c in zip(question, context)]
        else:
            return {'question': question, 'context': context}

    @classmethod
    def from_config(cls, model, tokenizer: PreTrainedTokenizer, **kwargs):
        pass

    def __call__(self, *texts, **kwargs):
        # Set defaults values
        kwargs.setdefault('max_answer_len', 15)
        kwargs.setdefault('topk', 1)

        if kwargs['topk'] < 1:
            raise ValueError('topk parameter should be >= 1 (got {})'.format(kwargs['topk']))

        if kwargs['max_answer_len'] < 1:
            raise ValueError('max_answer_len parameter should be >= 1 (got {})'.format(kwargs['max_answer_len']))

        # Tabular input
        if 'question' in kwargs and 'context' in kwargs:
            texts = QuestionAnsweringPipeline.create_sample(kwargs['question'], kwargs['context'])
        elif 'data' in kwargs:
            texts = kwargs['data']
        # Generic compatibility with sklearn and Keras
        elif 'X' in kwargs and not texts:
            texts = kwargs.pop('X')
        else:
            (texts, ) = texts

        if not isinstance(texts, (dict, list)):
            raise Exception('QuestionAnsweringPipeline requires predict argument to be a tuple (context, question) or a List of dict.')

        if not isinstance(texts, list):
            texts = [texts]

        # Map to tuple (question, context)
        texts = [(text['question'], text['context']) for text in texts]
        inputs = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=False, return_tensors='tf' if is_tf_available() else 'pt',
            return_attention_masks=True, return_input_lengths=False
        )

        token_type_ids = inputs.pop('token_type_ids')

        if is_tf_available():
            # TODO trace model
            start, end = self.model(inputs)
            start, end = start.numpy(), end.numpy()
        else:
            import torch
            with torch.no_grad():
                # Retrieve the score for the context tokens only (removing question tokens)
                start, end = self.model(**inputs)
                start, end = start.cpu().numpy(), end.cpu().numpy()

        answers = []
        for i in range(len(texts)):
            context_idx = token_type_ids[i] == 1
            start_, end_ = start[i, context_idx], end[i, context_idx]

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_) / np.sum(np.exp(start_))
            end_ = np.exp(end_) / np.sum(np.exp(end_))
            starts, ends, scores = self.decode(start_, end_, kwargs['topk'], kwargs['max_answer_len'])

            # Convert the answer (tokens) back to the original text
            answers += [[
                {**{'score': score}, **self.span_to_answer(texts[i][1], s, e)}
                for s, e, score in zip(starts, ends, scores)
            ]]

        return answers

    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
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
    'text-classification': {
        'impl': TextClassificationPipeline,
        'tf': TFAutoModelForSequenceClassification if is_tf_available() else None,
        'pt': AutoModelForSequenceClassification if is_torch_available() else None
    },
    'question-answering': {
        'impl': QuestionAnsweringPipeline,
        'tf': TFAutoModelForQuestionAnswering if is_tf_available() else None,
        'pt': AutoModelForQuestionAnswering if is_torch_available() else None
    }
}


def pipeline(task: str, model, tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None, **kwargs) -> Pipeline:
    """
    Utility factory method to build pipeline.
    """
    # Try to infer tokenizer from model name (if provided as str)
    if tokenizer is None and isinstance(model, str):
        tokenizer = model
    else:
        # Impossible to guest what is the right tokenizer here
        raise Exception('Tokenizer cannot be None if provided model is a PreTrainedModel instance')

    tokenizer = tokenizer if isinstance(tokenizer, PreTrainedTokenizer) else AutoTokenizer.from_pretrained(tokenizer)

    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task, allocator = targeted_task['impl'], targeted_task['tf'] if is_tf_available() else targeted_task['pt']

    model = allocator.from_pretrained(model)
    return task(model, tokenizer, **kwargs)
