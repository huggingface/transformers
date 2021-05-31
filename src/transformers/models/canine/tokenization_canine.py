# coding=utf-8
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for CANINE."""
from ...utils import logging
from ...tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/canine-s": "https://huggingface.co/google/canine-s/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/canine-s": 16384,
}


class CanineTokenizer(PreTrainedTokenizer):
    r"""
    Construct a CANINE tokenizer.

    :class:`~transformers.CanineTokenizer` inherits from :class:`~transformers.PreTrainedTokenizer`.

    Refer to superclass :class:`~transformers.PreTrainedTokenizer` for usage examples and documentation concerning
    parameters.
    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
