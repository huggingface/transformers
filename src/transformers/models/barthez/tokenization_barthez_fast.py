# coding=utf-8
# Copyright 2020 Ecole Polytechnique and the HuggingFace Inc. team.
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
# limitations under the License
""" Tokenization classes for the BARThez model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_barthez import BarthezTokenizer
else:
    BarthezTokenizer = None

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/tokenizer.json",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/tokenizer.json",
        "moussaKam/barthez-orangesum-title": "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}

SPIECE_UNDERLINE = "▁"


class BarthezTokenizerFast(PreTrainedTokenizerFast):
    """
    Adapted from :class:`~transformers.CamembertTokenizer` and :class:`~transformers.BartTokenizer`. Construct a "fast"
    BARThez tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.

    Attributes: sp_model (:obj:`SentencePieceProcessor`): The `SentencePiece` processor that is used for every
    conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]
    slow_tokenizer_class = BarthezTokenizer

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
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
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab_file = vocab_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BARThez sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
