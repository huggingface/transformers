# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Fast Tokenization classes for MPNet."""


from typing import List, Optional

from ...tokenization_utils import AddedToken
from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_mpnet import MPNetTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "mpnet-base": "",
    },
    "tokenizer_file": {
        "mpnet-base": "",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "mpnet-base": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "mpnet-base": {"do_lower_case": True},
}


class MPNetTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" MPNet tokenizer, derived from the fast Bert tokenizer. Peculiarities:

    This tokenizer inherits from :class:`~transformers.BertTokenizerFast` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        bos_token (:obj:`string`, `optional`, defaults to "<s>"):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            .. note:: When building a sequence using special tokens, this is not the token that is used for the
            beginning of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`string`, `optional`, defaults to "</s>"):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`string`, `optional`, defaults to "</s>"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`string`, `optional`, defaults to "<s>"):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`string`, `optional`, defaults to "<unk>"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`string`, `optional`, defaults to "<mask>"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = MPNetTokenizer

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def mask_token(self) -> str:
        """
        :obj:`str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while
        not having been set.

        MPNet tokenizer has a special mask token to be usble in the fill-mask pipeline. The mask token will greedily
        comprise the space before the `<mask>`.
        """
        if self._mask_token is None and self.verbose:
            logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on MPNet.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        # So we set lstrip to True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
