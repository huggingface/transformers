# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RAG."""
import os

from .tokenization_auto import AutoTokenizer
from .tokenization_bart import BartTokenizer, BartTokenizerFast
from .utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


RAG_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/rag-sequence-nq": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
        "facebook/rag-token-nq": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    },
    "merges_file": {
        "facebook/rag-sequence-nq": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
        "facebook/rag-token-nq": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    },
}


class RagDefaultTokenizer(BartTokenizer):
    r"""
    Constructs a  RagDefaultTokenizer.

    :class:`~transformers.RagDefaultTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = RAG_PRETRAINED_VOCAB_FILES_MAP


class RagDefaultTokenizerFast(BartTokenizerFast):
    r"""
    Constructs a  RagDefaultTokenizerFast.

    :class:`~transformers.RagDefaultTokenizerFast` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = RAG_PRETRAINED_VOCAB_FILES_MAP


class RagTokenizer:
    def __init__(self, question_encoder_tokenizer, generator_tokenizer):
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.generator_tokenizer = generator_tokenizer

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        question_encoder_tokenizer_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_tokenizer_path = os.path.join(save_directory, "generator_tokenizer")
        self.question_encoder_tokenizer.save_pretrained(question_encoder_tokenizer_path)
        self.generator_tokenizer.save_pretrained(generator_tokenizer_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        question_encoder_tokenizer_path = os.path.join(pretrained_model_name_or_path, "question_encoder_tokenizer")
        generator_tokenizer_path = os.path.join(pretrained_model_name_or_path, "generator_tokenizer")
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(
            question_encoder_tokenizer_path, config=config.question_encoder
        )
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path, config=config.generator)
        return cls(question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer)

    def __call__(self, *args, **kwargs):
        # TODO
        pass
