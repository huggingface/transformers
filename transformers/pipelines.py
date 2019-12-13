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
from itertools import groupby
from typing import Union, Optional, Tuple, List, Dict

import numpy as np

from transformers import AutoTokenizer, PreTrainedTokenizer, PretrainedConfig, \
    SquadExample, squad_convert_examples_to_features, is_tf_available, is_torch_available, logger

if is_tf_available():
    from transformers import TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering, TFAutoModelForTokenClassification

if is_torch_available():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification


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


class NerPipeline(Pipeline):

    def __init__(self, model, tokenizer: PreTrainedTokenizer):
        super().__init__(model, tokenizer)

    @classmethod
    def from_config(cls, model, tokenizer: PreTrainedTokenizer, **kwargs):
        pass

    def __call__(self, *texts, **kwargs):
        (texts, ), answers = texts, []

        for sentence in texts:

            # Ugly token to word idx mapping (for now)
            token_to_word, words = [], sentence.split(' ')
            for i, w in enumerate(words):
                tokens = self.tokenizer.tokenize(w)
                token_to_word += [i] * len(tokens)
            tokens = self.tokenizer.encode_plus(sentence, return_attention_mask=False, return_tensors='tf' if is_tf_available() else 'pt')

            # Forward
            if is_torch_available():
                with torch.no_grad():
                    entities = self.model(**tokens)[0][0].cpu().numpy()
            else:
                entities = self.model(tokens)[0][0].numpy()

            # Normalize scores
            answer, token_start = [], 1
            for idx, word in groupby(token_to_word[1:-1]):

                # Sum log prob over token, then normalize across labels
                score = np.exp(entities[token_start]) / np.exp(entities[token_start]).sum(-1, keepdims=True)
                label_idx = score.argmax()

                answer += [{
                    'word': words[idx - 1], 'score': score[label_idx], 'entity': self.model.config.id2label[label_idx]
                }]

                # Update token start
                token_start += len(list(word))

            # Append
            answers += [answer]
        return answers


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline involving Tokenization and Inference.
    """

    @classmethod
    def from_config(cls, model, tokenizer: PreTrainedTokenizer, **kwargs):
        pass

    @staticmethod
    def create_sample(question: Union[str, List[str]], context: Union[str, List[str]]) -> Union[SquadExample, List[SquadExample]]:
        is_list = isinstance(question, list)

        if is_list:
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    @staticmethod
    def handle_args(*inputs, **kwargs) -> List[SquadExample]:
        # Position args, handling is sensibly the same as X and data, so forwarding to avoid duplicating
        if inputs is not None and len(inputs) > 1:
            kwargs['X'] = inputs

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

    def __init__(self, model, tokenizer: Optional[PreTrainedTokenizer]):
        super().__init__(model, tokenizer)

    def inputs_for_model(self, features: Union[SquadExample, List[SquadExample]]) -> Dict:
        args = ['input_ids', 'attention_mask']
        model_type = type(self.model).__name__.lower()

        if 'distilbert' not in model_type and 'xlm' not in model_type:
            args += ['token_type_ids']

        if 'xlnet' in model_type or 'xlm' in model_type:
            args += ['cls_index', 'p_mask']

        if isinstance(features, SquadExample):
            return {k: features.__dict__[k] for k in args}
        else:
            return {k: [feature.__dict__[k] for feature in features] for k in args}

    def __call__(self, *texts, **kwargs):
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

        examples = QuestionAnsweringPipeline.handle_args(texts, **kwargs)

        # Convert inputs to features
        features = squad_convert_examples_to_features(examples, self.tokenizer, kwargs['max_seq_len'], kwargs['doc_stride'], kwargs['max_question_len'], False)
        fw_args = self.inputs_for_model(features)

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

            # TODO : What happend if not possible
            # Mask CLS
            start_[0] = end_[0] = 0

            starts, ends, scores = self.decode(start_, end_, kwargs['topk'], kwargs['max_answer_len'])
            char_to_word = np.array(example.char_to_word_offset)

            # Convert the answer (tokens) back to the original text
            answers += [[
                {
                    'score': score,
                    'start': np.where(char_to_word == feature.token_to_orig_map[s])[0][0],
                    'end': np.where(char_to_word == feature.token_to_orig_map[e])[0][-1],
                    'answer': ' '.join(example.doc_tokens[feature.token_to_orig_map[s]: feature.token_to_orig_map[e] + 1])
                }
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
    'ner': {
      'impl': NerPipeline,
      'tf': TFAutoModelForTokenClassification if is_tf_available() else None,
      'pt': AutoModelForTokenClassification if is_torch_available() else None,
    },
    'question-answering': {
        'impl': QuestionAnsweringPipeline,
        'tf': TFAutoModelForQuestionAnswering if is_tf_available() else None,
        'pt': AutoModelForQuestionAnswering if is_torch_available() else None
    }
}


def pipeline(task: str, model, config: Optional[PretrainedConfig] = None, tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None, **kwargs) -> Pipeline:
    """
    Utility factory method to build pipeline.
    """
    # Try to infer tokenizer from model name (if provided as str)
    if not isinstance(tokenizer, PreTrainedTokenizer):
        if not isinstance(model, str):
            # Impossible to guest what is the right tokenizer here
            raise Exception('Tokenizer cannot be None if provided model is a PreTrainedModel instance')
        else:
            tokenizer = model

    tokenizer = tokenizer if isinstance(tokenizer, PreTrainedTokenizer) else AutoTokenizer.from_pretrained(tokenizer)

    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task, allocator = targeted_task['impl'], targeted_task['tf'] if is_tf_available() else targeted_task['pt']

    model = allocator.from_pretrained(model)
    return task(model, tokenizer, **kwargs)
