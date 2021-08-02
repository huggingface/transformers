# coding=utf-8
# Copyright Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy and The HuggingFace Inc. team.
# All rights reserved.
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
"""Tokenization classes for Splinter."""
from typing import List, Optional

from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "splinter-base": "https://huggingface.co/splinter-base/resolve/main/vocab.txt",
        "splinter-large": "https://huggingface.co/splinter-large/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "splinter-base": 512,
    "splinter-large": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "splinter-base": {"do_lower_case": False},
    "splinter-large": {"do_lower_case": False},
}


class SplinterTokenizer(BertTokenizer):
    r"""
    Construct a Splinter tokenizer.

    :class:`~transformers.SplinterTokenizer` is similar to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    It adds a special question token in order to create question representations from its contextualized embedding.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        question_token="[QUESTION]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=(question_token,),
            **kwargs,
        )

        self.question_token = question_token

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a pair of sequence for question answering tasks by concatenating and adding special
        tokens. A Splinter sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences for question answering: ``[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                The question token IDs if pad_on_right, else context tokens IDs
            token_ids_1 (:obj:`List[int]`, `optional`):
                The context token IDs if pad_on_right, else question token IDs

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # Input is question-then-context
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # Input is context-then-question
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. `What are token type IDs?
        <../glossary.html#token-type-ids>`__

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (:obj:`List[int]`): The first tokenized sequence.
            token_ids_1 (:obj:`List[int]`, `optional`): The second tokenized sequence.

        Returns:
            :obj:`List[int]`: The token type ids.
        """
        if token_ids_1 is None:
            return [0] + [0] * len(token_ids_0) + [0]

        if self.padding_side == "right":
            # Input is question-then-context
            return [0] + [0] * len(token_ids_0) + [0, 0, 0] + [1] * len(token_ids_1) + [1]
        else:
            # Input is context-then-question
            return [0] + [0] * len(token_ids_0) + [0] + [1] * len(token_ids_1) + [1] + [1] + [1]

    @property
    def question_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.

        Returns :obj:`None` if the token has not been set.
        """
        return self.convert_tokens_to_ids(self.question_token)
