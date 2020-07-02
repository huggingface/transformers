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
"""Tokenization classes for DPR."""


import logging
from typing import Optional, Union

from .tokenization_bert import BertTokenizer, BertTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-question_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-reader-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}

CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-ctx_encoder-single-nq-base": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-question_encoder-single-nq-base": 512,
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/dpr-reader-single-nq-base": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-question_encoder-single-nq-base": {"do_lower_case": True},
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/dpr-reader-single-nq-base": {"do_lower_case": True},
}


class DPRContextEncoderTokenizer(BertTokenizer):
    r"""
    Constructs a  DPRContextEncoderTokenizer.

    :class:`~transformers.DPRContextEncoderTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION


class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a  "Fast" DPRContextEncoderTokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DDPRContextEncoderTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION


class DPRQuestionEncoderTokenizer(BertTokenizer):
    r"""
    Constructs a  DPRQuestionEncoderTokenizer.

    :class:`~transformers.DPRQuestionEncoderTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION


class DPRQuestionEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a  "Fast" DPRQuestionEncoderTokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRQuestionEncoderTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION

    def __call__(
        self,
        question,
        titles,
        texts,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        max_length: Optional[int] = 512,
        return_tensors=None,
        **kwargs
    ):
        """
        Return a dictionary with the token ids of the input strings and other information to give to `DPRReader.generate`.
        It converts the strings of a question and different passages (title + text) in a sequence of ids (integer), using the tokenizer and vocabulary.
        The resulting `input_ids` is a matrix of size (n_passages, sequence_length) with the format:

            [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

        Args:
            `question` (:obj:`str`):
                The question to be encoded.
            `titles` (:obj:`str`, :obj:`List[str]`):
                The passagestitles to be encoded. This can be a string, a list of strings if there are several passages.
            `texts` (:obj:`str`, :obj:`List[str]`):
                The passages texts to be encoded. This can be a string, a list of strings if there are several passages.
            `padding` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control padding. Accepts the following values:

                * `True` or `'longest'`: pad to the longest sequence in the batch (or no padding if only a single sequence if provided),
                * `'max_length'`: pad to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`)
                * `False` or `'do_not_pad'` (default): No padding (i.e. can output batch with sequences of uneven lengths)
            `truncation` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control truncation. Accepts the following values:

                * `True` or `'only_first'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`).
                * `False` or `'do_not_truncate'` (default): No truncation (i.e. can output batch with sequences length greater than the model max admissible input size)
            `max_length` (:obj:`Union[int, None]`, `optional`, defaults to :obj:`None`):
                Control the length for padding/truncation. Accepts the following values

                * `None` (default): This will use the predefined model max length if required by one of the truncation/padding parameters. If the model has no specific max input length (e.g. XLNet) truncation/padding to max length is deactivated.
                * `any integer value` (e.g. `42`): Use this specific maximum length value if required by one of the truncation/padding parameters.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[list[int]],
                    passage_offsets: list[int],
                    sequence_lenghts: list[int]
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a the `DPRReader.generate` method
            - ``passage_offsets``: list of indices of the beginning of each passage text inside the `input_ids` matrix
            - ``sequence_lenghts``: list of indices of the whole sequqnce length of row inside the `input_ids` matrix

        """
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        assert len(titles) == len(
            texts
        ), "There should be as many titles than texts but got {} titles and {} texts.".format(len(titles), len(texts))
        encoded_question_and_titles = super().__call__(
            [question] * n_passages, titles, padding=False, truncation=False
        )["input_ids"]
        passage_offsets = [
            len(encoded_question_and_title) for encoded_question_and_title in encoded_question_and_titles
        ]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_sequences = [
            (encoded_question_and_title + encoded_text)[:max_length]
            if max_length is not None and truncation
            else encoded_question_and_title + encoded_text
            for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
        ]
        input_ids = self.pad(
            {"input_ids": encoded_sequences}, padding=padding, max_length=max_length, return_tensors=return_tensors
        )["input_ids"]
        sequence_lenghts = [len(encoded_sequence) for encoded_sequence in encoded_sequences]
        return {"input_ids": input_ids, "passage_offsets": passage_offsets, "sequence_lenghts": sequence_lenghts}


class DPRReaderTokenizer(BertTokenizer):
    r"""
    Constructs a  DPRReaderTokenizer.

    :class:`~transformers.DPRReaderTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION


class DPRReaderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a  "Fast" DPRReaderTokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRReaderTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION

    def __call__(
        self,
        question,
        titles,
        texts,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        max_length: Optional[int] = 512,
        return_tensors=None,
        **kwargs
    ):
        """
        Return a dictionary with the token ids of the input strings and other information to give to `DPRReader.generate`.
        It converts the strings of a question and different passages (title + text) in a sequence of ids (integer), using the tokenizer and vocabulary.
        The resulting `input_ids` is a matrix of size (n_passages, sequence_length) with the format:

            [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

        Args:
            `question` (:obj:`str`):
                The question to be encoded.
            `titles` (:obj:`str`, :obj:`List[str]`):
                The passagestitles to be encoded. This can be a string, a list of strings if there are several passages.
            `texts` (:obj:`str`, :obj:`List[str]`):
                The passages texts to be encoded. This can be a string, a list of strings if there are several passages.
            `padding` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control padding. Accepts the following values:

                * `True` or `'longest'`: pad to the longest sequence in the batch (or no padding if only a single sequence if provided),
                * `'max_length'`: pad to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`)
                * `False` or `'do_not_pad'` (default): No padding (i.e. can output batch with sequences of uneven lengths)
            `truncation` (:obj:`Union[bool, str]`, `optional`, defaults to :obj:`True`):
                Activate and control truncation. Accepts the following values:

                * `True` or `'only_first'`: truncate to a max length specified in `max_length` or to the max acceptable input length for the model if no length is provided (`max_length=None`).
                * `False` or `'do_not_truncate'` (default): No truncation (i.e. can output batch with sequences length greater than the model max admissible input size)
            `max_length` (:obj:`Union[int, None]`, `optional`, defaults to :obj:`None`):
                Control the length for padding/truncation. Accepts the following values

                * `None` (default): This will use the predefined model max length if required by one of the truncation/padding parameters. If the model has no specific max input length (e.g. XLNet) truncation/padding to max length is deactivated.
                * `any integer value` (e.g. `42`): Use this specific maximum length value if required by one of the truncation/padding parameters.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[list[int]],
                    passage_offsets: list[int],
                    sequence_lenghts: list[int]
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a the `DPRReader.generate` method
            - ``passage_offsets``: list of indices of the beginning of each passage text inside the `input_ids` matrix
            - ``sequence_lenghts``: list of indices of the whole sequqnce length of row inside the `input_ids` matrix

        """
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        assert len(titles) == len(
            texts
        ), "There should be as many titles than texts but got {} titles and {} texts.".format(len(titles), len(texts))
        encoded_question_and_titles = super().__call__(
            [question] * n_passages, titles, padding=False, truncation=False
        )["input_ids"]
        passage_offsets = [
            len(encoded_question_and_title) for encoded_question_and_title in encoded_question_and_titles
        ]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_sequences = [
            (encoded_question_and_title + encoded_text)[:max_length]
            if max_length is not None and truncation
            else encoded_question_and_title + encoded_text
            for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
        ]
        input_ids = self.pad(
            {"input_ids": encoded_sequences}, padding=padding, max_length=max_length, return_tensors=return_tensors
        )["input_ids"]
        sequence_lenghts = [len(encoded_sequence) for encoded_sequence in encoded_sequences]
        return {"input_ids": input_ids, "passage_offsets": passage_offsets, "sequence_lenghts": sequence_lenghts}
