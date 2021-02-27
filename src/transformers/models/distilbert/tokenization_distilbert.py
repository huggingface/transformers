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
"""Tokenization classes for DistilBERT."""

from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "distilbert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "distilbert-base-uncased-distilled-squad": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        "distilbert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        "distilbert-base-cased-distilled-squad": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/vocab.txt",
        "distilbert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "distilbert-base-uncased": 512,
    "distilbert-base-uncased-distilled-squad": 512,
    "distilbert-base-cased": 512,
    "distilbert-base-cased-distilled-squad": 512,
    "distilbert-base-german-cased": 512,
    "distilbert-base-multilingual-cased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {"do_lower_case": True},
    "distilbert-base-uncased-distilled-squad": {"do_lower_case": True},
    "distilbert-base-cased": {"do_lower_case": False},
    "distilbert-base-cased-distilled-squad": {"do_lower_case": False},
    "distilbert-base-german-cased": {"do_lower_case": False},
    "distilbert-base-multilingual-cased": {"do_lower_case": False},
}


class DistilBertTokenizer(BertTokenizer):
    r"""
    Construct a DistilBERT tokenizer.

    :class:`~transformers.DistilBertTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["input_ids", "attention_mask"]
