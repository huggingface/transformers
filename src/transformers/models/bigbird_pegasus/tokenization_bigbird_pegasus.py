# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for BigBirdPegasus."""


from ...utils import logging
from ..big_bird.tokenization_big_bird import BigBirdTokenizer
from typing import List, Optional

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-pegasus-large-arxiv": "https://huggingface.co/google/bigbird-pegasus-large-arxiv/resolve/main/spiece.model",
        "google/bigbird-pegasus-large-pubmed": "https://huggingface.co/google/bigbird-pegasus-large-pubmed/resolve/main/spiece.model",
        "google/bigbird-pegasus-large-bigpatent": "https://huggingface.co/google/bigbird-pegasus-large-bigpatent/resolve/main/spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-pegasus-large-arxiv": 4096,
    "google/bigbird-pegasus-large-pubmed": 4096,
    "google/bigbird-pegasus-large-bigpatent": 4096,
}


class BigBirdPegasusTokenizer(BigBirdTokenizer):
    r"""
    Construct a BigBirdPegasus tokenizer.

    :class:`~transformers.BigBirdPegasusTokenizer` is identical to :class:`~transformers.BigBirdTokenizer`. Refer to superclass
    :class:`~transformers.BigBirdTokenizer` for usage examples and documentation concerning the initialization
    parameters and other methods.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # TODO: confirm once
    # run_summarization is not adding CLS & SEP
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for summarization tasks by concatenating and
        adding special tokens. A Big Bird sequence has the following format:

        - single sequence: ``X``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
