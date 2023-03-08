# coding=utf-8
# Copyright 2023 HuggingFace Inc. team.
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
""" Tokenization class for model UDOP."""
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple

import sentencepiece as spm
from tokenizers import processors

from transformers import PreTrainedTokenizer
from transformers.convert_slow_tokenizer import SpmConverter
from transformers.utils import logging


# The special tokens of UDOPTokenizer is hard-coded with <extra_id_{}>
# Created another class UDOPTokenizer extending it to add special visual tokens like <loc_{}>
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class UdopTokenizer(PreTrainedTokenizer):
    """
    Construct a UDOP tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (`int`, *optional*, defaults to 100):
           Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are
            accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1. These tokens can be
            retrieved by calling get_sentinel_tokens method and token ids can be by calling get_sentinel_token_ids
            method
         additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=501,
        other_extra_ids=200,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        **kwargs,
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and "<extra_id_0>" not in additional_special_tokens:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_ids = 0

        if loc_extra_ids > 0 and "<loc_0>" not in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])

        if other_extra_ids > 0 and "<other_0>" not in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids + self._other_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4
        elif token.startswith("<extra_l_id_"):
            match = re.match(r"<extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3
        elif token.startswith("</extra_l_id_"):
            match = re.match(r"</extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2
        elif token.startswith("<extra_t_id_"):
            match = re.match(r"<extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids
        elif token.startswith("</extra_t_id_"):
            match = re.match(r"</extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids
        elif token.startswith("<loc_"):
            match = re.match(r"<loc_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids
        elif token.startswith("<other_"):
            match = re.match(r"<other_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1

        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids - 1:
                index_loc = self.vocab_size - 1 - index
                token = f"<other_{index_loc}>"
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 5 - 1:
                index_loc = self.vocab_size - self._other_extra_ids - 1 - index
                token = f"<loc_{index_loc}>"
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 4 - 1:
                token = "</extra_t_id_{}>".format(
                    self.vocab_size - self._other_extra_ids - self._loc_extra_ids - 1 - index
                )
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 3 - 1:
                token = "<extra_t_id_{}>".format(
                    self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids - 1 - index
                )
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 2 - 1:
                token = "</extra_l_id_{}>".format(
                    self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2 - 1 - index
                )
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<extra_l_id_{}>".format(
                    self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3 - 1 - index
                )
            elif index > self.sp_model.get_piece_size() - 1:
                token = "<extra_id_{}>".format(
                    self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4 - 1 - index
                )
            else:
                raise
        return token

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)


# Below are for Rust-based Fast Tokenizer
class UdopConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_l_id_{}>".format(i), 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_l_id_{}>".format(i), 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_t_id_{}>".format(i), 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_t_id_{}>".format(i), 0.0) for i in range(num_extra_ids - 1, -1, -1)]

        num_loc_extra_ids = self.original_tokenizer._loc_extra_ids
        vocab += [("<loc_{}>".format(i), 0.0) for i in range(num_loc_extra_ids - 1, -1, -1)]

        num_other_extra_ids = self.original_tokenizer._other_extra_ids
        vocab += [("<other_0{}>".format(i), 0.0) for i in range(num_other_extra_ids - 1, -1, -1)]

        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_udoptokenizer(UdopTokenizer):
    return UdopConverter(UdopTokenizer).converted()
