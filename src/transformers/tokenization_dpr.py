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

from .tokenization_bert import BertTokenizer, BertTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "dpr-ctx_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "dpr-question_encoder-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}
READER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "dpr-reader-single-nq-base": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    }
}

CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dpr-ctx_encoder-single-nq-base": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dpr-question_encoder-single-nq-base": 512,
}
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dpr-reader-single-nq-base": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "dpr-ctx_encoder-single-nq-base": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "dpr-question_encoder-single-nq-base": {"do_lower_case": True},
}
READER_PRETRAINED_INIT_CONFIGURATION = {
    "dpr-reader-single-nq-base": {"do_lower_case": True},
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
