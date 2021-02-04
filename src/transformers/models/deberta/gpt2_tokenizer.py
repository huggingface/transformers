# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
""" GPT2 Tokenization class for model DeBERTa."""

import os
import unicodedata
from functools import lru_cache


try:
    import regex as re
except ImportError:
    raise ImportError("Please install regex with: pip install regex")

___all__ = ["GPT2Tokenizer"]

VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings. The reversible bpe codes work on unicode
    strings. This means you need a large # of unicode characters in your vocab if you want to avoid UNKs. When you're
    at something like a 10B token dataset you end up needing around 5K for decent coverage. This is a signficant
    percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode
    strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
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
    """
    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length
    strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip([tuple(k) for k in bpe_merges], range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

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

    def split_to_words(self, text):
        return list(re.findall(self.pat, text))

    def encode(self, text):
        bpe_tokens = []
        for token in self.split_to_words(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder(encoder, vocab):
    return Encoder(
        encoder=encoder,
        bpe_merges=vocab,
    )


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class GPT2Tokenizer(object):
    """
    A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer

    Args:
        vocab_file (:obj:`str`, optional):
            The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases
            <https://github.com/microsoft/DeBERTa/releases>`_, e.g. "bpe_encoder", default: `None`.

            If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file
            is a state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used
            in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. The difference between our wrapped GPT2
            tokenizer and RoBERTa wrapped tokenizer are,

            - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a
              sentence. We use `[CLS]` and `[SEP]` as the `start` and `end` token of input sentence which is the same
              as `BERT`.

            - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0,
              `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264

        special_tokens (:obj:`list`, optional):
            List of special tokens to be added to the end of the vocabulary.
    """

    def __init__(self, vocab_file=None, special_tokens=None, **kwargs):
        import torch

        self.pad_token = "[PAD]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"

        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_token_id = self.add_symbol(self.pad_token)
        self.cls_token_id = self.add_symbol(self.cls_token)
        self.sep_token_id = self.add_symbol(self.sep_token)
        self.unk_token_id = self.add_symbol(self.unk_token)

        self.gpt2_encoder = torch.load(vocab_file)
        self.bpe = get_encoder(self.gpt2_encoder["encoder"], self.gpt2_encoder["vocab"])
        for w, n in self.gpt2_encoder["dict_map"]:
            self.add_symbol(w, n)

        self.mask_token = "[MASK]"
        self.mask_id = self.add_symbol(self.mask_token)
        self.special_tokens = ["[MASK]", "[SEP]", "[PAD]", "[UNK]", "[CLS]"]
        if special_tokens is not None:
            for t in special_tokens:
                self.add_special_token(t)

        self.vocab = self.indices
        self.ids_to_tokens = self.symbols

    def tokenize(self, text):
        """
        Convert an input text to tokens.

        Args:
          text (:obj:`str`): input text to be tokenized.

        Returns:
          A list of byte tokens where each token represent the byte id in GPT2 byte dictionary

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
        """
        bpe = self._encode(text)

        return [t for t in bpe.split(" ") if t]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert list of tokens to ids

        Args:
          tokens (:obj:`list<str>`): list of tokens

        Returns:
          List of ids
        """

        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Convert list of ids to tokens

        Args:
          ids (:obj:`list<int>`): list of ids

        Returns:
          List of tokens
        """

        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def split_to_words(self, text):
        return self.bpe.split_to_words(text)

    def decode(self, tokens):
        """
        Decode list of tokens to text strings

        Args:
          tokens (:obj:`list<str>`): list of tokens.

        Returns:
          Text string corresponds to the input tokens.

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
          >>> tokenizer.decode(tokens)
          'Hello world!'
        """
        return self.bpe.decode([int(t) for t in tokens if t not in self.special_tokens])

    def add_special_token(self, token):
        """
        Adds a special token to the dictionary

        Args:
          token (:obj:`str`): Tthe new token/word to be added to the vocabulary.

        Returns:
          The id of new token in the vocabulary.

        """
        self.special_tokens.append(token)
        return self.add_symbol(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        s = self._decode(token)
        if len(s) == 1 and (_is_whitespace(list(s)[0]) or _is_control(list(s)[0]) or _is_punctuation(list(s)[0])):
            return False

        return not s.startswith(" ")

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym]

    def _encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def _decode(self, x: str) -> str:
        return self.bpe.decode(map(int, x.split()))

    def add_symbol(self, word, n=1):
        """
        Adds a word to the dictionary

        Args:
          word (:obj:`str`): Tthe new token/word to be added to the vocabulary.
          n (int, optional): The frequency of the word.

        Returns:
          The id of the new word.

        """
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def save_pretrained(self, path: str, filename_prefix: str = None):
        import torch

        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        torch.save(self.gpt2_encoder, full_path)
        return (full_path,)
