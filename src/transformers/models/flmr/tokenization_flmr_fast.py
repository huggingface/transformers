# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team, The Hugging Face Team.
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
"""Tokenization classes for FLMR."""


from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_flmr import FLMRContextEncoderTokenizer, FLMRQueryEncoderTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer_config.json"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/context_tokenizer/vocab.txt"
        ),
        "LinWeizheDragon/FLMR": ("https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/context_tokenizer/vocab.txt"),
    },
    "tokenizer_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/context_tokenizer/tokenizer_config.json"
        ),
        "LinWeizheDragon/FLMR": (
            "https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/context_tokenizer/tokenizer_config.json"
        ),
    },
}
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/query_tokenizer/vocab.txt"
        ),
        "LinWeizheDragon/FLMR": ("https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/query_tokenizer/vocab.txt"),
    },
    "tokenizer_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/query_tokenizer/tokenizer_config.json"
        ),
        "LinWeizheDragon/FLMR": ("https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/query_tokenizer/tokenizer_config.json"),
    },
}


CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "LinWeizheDragon/PreFLMR_ViT-L": 512,
    "LinWeizheDragon/FLMR": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "LinWeizheDragon/PreFLMR_ViT-L": 512,
    "LinWeizheDragon/FLMR": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "LinWeizheDragon/PreFLMR_ViT-L": {"do_lower_case": True},
    "LinWeizheDragon/FLMR": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "LinWeizheDragon/PreFLMR_ViT-L": {"do_lower_case": True},
    "LinWeizheDragon/FLMR": {"do_lower_case": True},
}


class FLMRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" FLMRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`FLMRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = FLMRContextEncoderTokenizer


class FLMRQueryEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" FLMRQueryEncoderTokenizer tokenizer (backed by HuggingFace's *tokenizers* library).

    [`FLMRTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = FLMRQueryEncoderTokenizer
