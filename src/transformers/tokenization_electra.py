# coding=utf-8
# Copyright 2020 The Google AI Team, Stanford University and The HuggingFace Inc. team.
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

from .tokenization_bert import BertTokenizer, BertTokenizerFast


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/electra-small": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small/vocab.txt",
        "google/electra-base": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base/vocab.txt",
        "google/electra-large": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small": 512,
    "google/electra-base": 512,
    "google/electra-large": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small": {"do_lower_case": False},
    "google/electra-base": {"do_lower_case": False},
    "google/electra-large": {"do_lower_case": False},
}


class ElectraTokenizer(BertTokenizer):
    r"""
    Constructs an Electra tokenizer.
    :class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.ElectraTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


class ElectraTokenizerFast(BertTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
