# coding=utf-8
# Copyright (c) 2020, VinAI Research and the HuggingFace Inc. team.
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
""" Tokenization classes for PhoBERT """


import logging
from typing import List, Optional
from .tokenization_utils import PreTrainedTokenizer

from .tokenization_bertweet import Dictionary
import fastBPE

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "merges_file": "bpe.codes",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/phobert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/vinai/phobert-base/vocab.txt",
        "vinai/phobert-large": "https://s3.amazonaws.com/models.huggingface.co/bert/vinai/phobert-large/vocab.txt",
    },
    "merges_file": {
        "vinai/phobert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/vinai/phobert-base/bpe.codes",
        "vinai/phobert-large": "https://s3.amazonaws.com/models.huggingface.co/bert/vinai/phobert-large/bpe.codes",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vinai/phobert-base": 256,
    "vinai/phobert-large": 256,
}

class PhobertTokenizer(PreTrainedTokenizer):
    """
        Extend PretrainedTokenizer, with `fastBPE` (https://github.com/glample/fastBPE) required.
        Install: pip3 install fastBPE

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file which here is the bpe-codes file
        bos_token (:obj:`string`, `optional`, defaults to "<s>"):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning
                of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`string`, `optional`, defaults to "</s>"):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end
                of sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`string`, `optional`, defaults to "</s>"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        cls_token (:obj:`string`, `optional`, defaults to "<s>"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
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
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            max_len=256,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.vocab = Dictionary()
        self.vocab.add_from_file(vocab_file)
        self.bpe = fastBPE.fastBPE(merges_file)

        self.vocab_file = vocab_file
        self.merges_file = merges_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A PhoBERT sequence has the following format:
        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.vocab.index(self.cls_token)] + token_ids_0 + [self.vocab.index(self.sep_token)]
        cls = [self.vocab.index(self.cls_token)]
        sep = [self.vocab.index(self.sep_token)]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.vocab.index(self.sep_token), self.vocab.index(self.cls_token)] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates token_type_ids for PhoBERT, similar to RoBERTa, the values of token_type_ids will be alls 0.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of zeros.
        """
        sep = [self.vocab.index(self.sep_token)]
        cls = [self.vocab.index(self.cls_token)]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _tokenize(self, word_segmented_text):
        """Apply fastBPE on the word-segmented text"""
        return self.bpe.apply([word_segmented_text])[0].split()

    def convert_tokens_to_ids(self, token):
        """ Converts a list of tokens into a list of ids using the vocab."""
        tokens = " ".join(token)
        return self.vocab.encode_line(tokens, append_eos=False, add_if_not_exist=False).long().tolist()
