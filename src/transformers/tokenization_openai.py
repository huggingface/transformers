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
import re
from typing import List, Optional, Union

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer, Sequence, unicode_normalizer_from_str
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import BpeTrainer

from .tokenization_bert import BasicTokenizer
from .tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json"},
    "merges_file": {"openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-gpt": 512,
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer. Peculiarities:
        - lower case all inputs
        - uses SpaCy tokenizer and ftfy for pre-BPE tokenization if they are installed, fallback to BERT's BasicTokenizer if not.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        super().__init__(unk_token=unk_token, **kwargs)

        self.max_len_single_sentence = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens

        try:
            import ftfy
            from spacy.lang.en import English

            _nlp = English()
            self.nlp = _nlp.Defaults.create_tokenizer(_nlp)
            self.fix_text = ftfy.fix_text
        except ImportError:
            logger.warning("ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.")
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

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
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """ Tokenize a string. """
        split_tokens = []
        if self.fix_text is None:
            # Using BERT's BasicTokenizer
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])
        else:
            # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend([t for t in self.bpe(token.text.lower()).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an id in a token (BPE) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

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


class _OpenAIGPTCharBPETokenizer(BaseTokenizer):
    """
    OpenAI character-level BPE Tokenizer
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: Optional[str] = "<unk>",
        suffix: Optional[str] = "</w>",
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[str] = None,
    ):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE.from_files(
                    vocab_file, merges_file, dropout=dropout, unk_token=unk_token, end_of_word_suffix=suffix
                )
            )
        else:
            tokenizer = Tokenizer(BPE.empty())

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        # OpenAI normalization is the same as Bert
        normalizers += [BertNormalizer()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.decoder = BPEDecoder(suffix=suffix)

        parameters = {
            "model": "BPE",
            "unk_token": unk_token,
            "suffix": suffix,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        suffix: Optional[str] = "</w>",
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix=suffix,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)


class OpenAIGPTTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        kwargs.setdefault("unk_token", unk_token)
        super().__init__(
            _OpenAIGPTCharBPETokenizer(vocab_file=vocab_file, merges_file=merges_file, unk_token=unk_token), **kwargs
        )
