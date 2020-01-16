# coding=utf-8
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
"""Tokenization classes for OpenAI GPT."""


import json
import logging
import os
from functools import lru_cache

import regex as re
import tokenizers as tk

from .tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
        "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
        "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json",
        "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-vocab.json",
    },
    "merges_file": {
        "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
        "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
        "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt",
        "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding and tokenize methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer's ``encode``, ``decode``, and ``tokenize`` methods will not conserve
          the spaces at the beginning of a string: `tokenizer.decode(tokenizer.encode(" Hello")) = "Hello"`
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.max_len_single_sentence = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
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
        self.cache[token] = word
        return word

    def _tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        if add_prefix_space:
            text = " " + text

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file


class GPT2TokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_to_max_length=False,
        add_prefix_space=False,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        **kwargs
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        self._tokenizer = tk.Tokenizer(tk.models.BPE.from_files(vocab_file, merges_file))
        self._update_special_tokens()
        self._tokenizer.with_pre_tokenizer(tk.pre_tokenizers.ByteLevel.new(add_prefix_space=add_prefix_space))
        self._tokenizer.with_decoder(tk.decoders.ByteLevel.new())
        if max_length:
            self._tokenizer.with_truncation(max_length, stride=stride, strategy=truncation_strategy)
        self._tokenizer.with_padding(
            max_length=max_length if pad_to_max_length else None,
            direction=self.padding_side,
            pad_id=self.pad_token_id if self.pad_token_id is not None else 0,
            pad_type_id=self.pad_token_type_id,
            pad_token=self.pad_token if self.pad_token is not None else "",
        )
        self._decoder = tk.decoders.ByteLevel.new()
