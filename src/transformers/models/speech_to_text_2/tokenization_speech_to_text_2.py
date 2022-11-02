# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for Speech2Text2."""

import json
import os
from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/vocab.json"
        ),
    },
    "tokenizer_config_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/tokenizer_config.json"
        ),
    },
    "merges_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/merges.txt"
        ),
    },
}

BPE_TOKEN_MERGES = "</w>"
BPE_TOKEN_VOCAB = "@@ "


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# Speech2Text2 has no max input length
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/s2t-wav2vec2-large-en-de": 1024}


class Speech2Text2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Speech2Text2Tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        do_lower_case=False,
        merges_file=None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

        self.do_lower_case = do_lower_case

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        if merges_file is None:
            logger.info(f"No merges files provided. {self.__class__.__name__} can only be used for decoding.")

            self.bpe_ranks = None
            self.cache = None
        else:
            with open(merges_file, encoding="utf-8") as merges_handle:
                merges = merges_handle.read().split("\n")[:-1]

            merges = [tuple(merge.split()[:2]) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + BPE_TOKEN_MERGES,)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  " + BPE_TOKEN_MERGES:
            word = "\n" + BPE_TOKEN_MERGES

        if word.endswith(BPE_TOKEN_MERGES):
            word = word.replace(BPE_TOKEN_MERGES, "")

        word = word.replace(" ", BPE_TOKEN_VOCAB)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""

        if self.bpe_ranks is None:
            raise ValueError(
                "This tokenizer was instantiated without a `merges.txt` file, so"
                " that it can only be used for decoding, not for encoding."
                "Make sure to provide `merges.txt` file at instantiation to enable "
                "encoding."
            )

        if self.do_lower_case:
            text = text.lower()

        text = text.split()

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of output tokens into a single string.
        """
        # combine tokens
        string = " ".join(tokens)

        # make sure @@ tokens are concatenated
        string = "".join(string.split(BPE_TOKEN_VOCAB))

        return string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        if self.bpe_ranks is None:
            return (vocab_file,)

        with open(merges_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merges_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return (vocab_file, merges_file)
