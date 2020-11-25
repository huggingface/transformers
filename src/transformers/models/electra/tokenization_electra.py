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

from ..bert.tokenization_bert import BertTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

ELECTRA_PRETRAINED_TOKENIZER_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]


class ElectraTokenizer(BertTokenizer):
    r"""
    Construct an ELECTRA tokenizer.

    :class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = 512
