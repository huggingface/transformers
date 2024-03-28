# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Tokenization classes for RWKV5."""

import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
}
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ArthurZ/rwkv-5-utf": "https://huggingface.co/ArthurZ/rwkv-5-utf/blob/main/vocab.txt",
    },
}


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text.
    The separators are kept
    """
    text = text.strip()
    if not text:
        return []
    tokens = re.split(b"(?= )", text)
    return tokens


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token):
        self.vocab = vocab
        self.unk_token = unk_token

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = bytes(chars[start:end])
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                try:
                    cur_substr = cur_substr.decode()
                except UnicodeDecodeError:
                    cur_substr = str(cur_substr)
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class Rwkv5Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {"ArthurZ/rwkv-5-utf": 2048}

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, bos_token="<s>", eos_token="<s>", unk_token="<s>", **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        with open(vocab_file, "r") as reader:
            tokens = reader.readlines()
        vocab = {}
        for index, token in enumerate(tokens):
            token = eval(token.rstrip("\n"))
            vocab[token] = index

        self.add_bos_token = True
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=str(unk_token))
        self._added_tokens_decoder = {0: AddedToken(str(bos_token))}
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = {str(self.convert_ids_to_tokens(i)): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, split_special_tokens=False):
        return self.wordpiece_tokenizer.tokenize(text.encode("utf-8"))

    def _convert_token_to_id(self, token):
        """Converts a token (byte) to an id using the vocab."""
        if token.startswith("b'\\"):
            token = eval(token)
        elif not isinstance(token, bytes):
            token = token.encode("utf-8", errors="replace")
        return self.encoder.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (byte) using the vocab."""
        token = self.decoder.get(index, self.unk_token)
        if isinstance(token, (bytes)):
            token = token.decode("utf-8", errors="replace")
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (bytes) in a single string. Additional tokens are encoded to bytes"""
        out_string = b"".join([k.encode(errors="replace") if isinstance(k, str) else k for k in tokens]).decode(
            "utf-8"
        )
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w") as writer:
            for token, token_index in sorted(self.encoder.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(str(token) + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
