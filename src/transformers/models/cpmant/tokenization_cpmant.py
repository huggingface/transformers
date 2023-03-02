# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for CPMAnt."""
import collections
import os
from typing import List, Optional, Tuple

from transformers import is_torch_available
from transformers.utils import is_jieba_available


if is_torch_available():
    import torch

if is_jieba_available():
    import jieba

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cpm-ant-10b": 1024,
}

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
            else:
                sub_tokens.append(cur_substr)
                start = end

        return sub_tokens


class CPMAntTokenizer(PreTrainedTokenizer):
    """
    Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bod_token (`str`, *optional*, defaults to `<d>`):
            The beginning of document token.
        eod_token (`str`, *optional*, defaults to `</d>`):
            The end of document token.
        bos_token (`str`, *optional*, defaults to `<s>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `</s>`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `<pad>`):
            The token used for padding.
        unk_token (`str`, *optional*, defaults to `<unk>`):
            The unknown token.
        line_token (`str`, *optional*, defaults to `</n>`):
            The line token.
        space_token (`str`, *optional*, defaults to `</_>`):
            The space token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bod_token="<d>",
        eod_token="</d>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        line_token="</n>",
        space_token="</_>",
        **kwargs,
    ):
        super().__init__(
            bod_token=bod_token,
            eod_token=eod_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            line_token=line_token,
            space_token=space_token,
            padding_side="left",
            **kwargs,
        )
        self.bod_token = bod_token
        self.eod_token = eod_token
        self.encoder = load_vocab(vocab_file)
        self.encoder[" "] = self.encoder[space_token]
        self.encoder["\n"] = self.encoder[line_token]

        del self.encoder[space_token]
        del self.encoder[line_token]

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=self.unk_token)

    # always add special tokens
    def __call__(self, text, *args, **kwargs):
        if "add_special_tokens" in kwargs:
            kwargs["add_special_tokens"] = True
        ret = super().__call__(text, *args, **kwargs)
        return ret

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def bod_id(self):
        return self.encoder[self.bod_token]

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def eos_id(self):
        return self.encoder[self.eos_token]

    @property
    def bos_id(self):
        return self.encoder[self.bos_token]

    @property
    def pad_token_id(self):
        return self.encoder[self.pad_token]

    @property
    def unk_id(self):
        return self.encoder[self.unk_token]

    @property
    def newline_id(self):
        return self.encoder["\n"]

    def __len__(self):
        return len(self.encoder)

    def tokenize(self, text):
        """Tokenize a string."""
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def encode(self, text, **kwargs):
        """Encode a string into ids."""
        return [self.encoder[x] for x in self.tokenize(text)]

    def decode(self, tokens, **kwargs):
        """Decode ids into a string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        tokens = [i for i in tokens if i >= 0]
        tokens = [x for x in tokens if x != self.pad_token_id and x != self.eos_id and x != self.bos_id]
        text = "".join([self.decoder[x] for x in tokens])
        return text

    def check(self, token):
        return token in self.encoder

    def convert_tokens_to_ids(self, tokens):
        ret_ids = [self.encoder.get(x, self.encoder[self.unk_token]) for x in tokens]
        ret_ids = [j for j in ret_ids if j != self.unk_id]
        return ret_ids

    def convert_ids_to_tokens(self, ids):
        return [self.decoder[x] if x >= 0 else self.unk_token for x in ids]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in self.encoder.items():
                if index != token_index:
                    logging.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CPMAnt sequence has the following format:

        - single sequence: `[BOS] Sequence`.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence that special tokens will be added.
            token_ids_1 (`List[int]`): Dummy param in CPMAntTokenizer.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        return [self.bos_id] + token_ids_0
