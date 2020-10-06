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
""" Tokenization class for model DeBERTa."""

import logging
import os
import pathlib
import random
import unicodedata
from functools import lru_cache
from zipfile import ZipFile

import tqdm

import requests

from .tokenization_utils import PreTrainedTokenizer


try:
    import regex as re
except ImportError:
    raise ImportError("Please install regex with: pip install regex")


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-base/bpe_encoder.bin",
        "microsoft/deberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-large/bpe_encoder.bin",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

__all__ = ["DebertaTokenizer"]


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
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


class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip([tuple(k) for k in bpe_merges], range(len(bpe_merges))))
        self.cache = {}
        self.random = random.Random(0)

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


def download_asset(name, tag=None, no_cache=False, cache_dir=None):
    _tag = tag
    if _tag is None:
        _tag = "latest"
    if not cache_dir:
        cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
    os.makedirs(cache_dir, exist_ok=True)
    output = os.path.join(cache_dir, name)
    if os.path.exists(output) and (not no_cache):
        return output

    repo = "https://api.github.com/repos/microsoft/DeBERTa/releases"
    releases = requests.get(repo).json()
    if tag and tag != "latest":
        release = [r for r in releases if r["name"].lower() == tag.lower()]
        if len(release) != 1:
            raise Exception(f"{tag} can't be found in the repository.")
    else:
        release = releases[0]
    asset = [s for s in release["assets"] if s["name"].lower() == name.lower()]
    if len(asset) != 1:
        raise Exception(f"{name} can't be found in the release.")
    url = asset[0]["url"]
    headers = {}
    headers["Accept"] = "application/octet-stream"
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Request for {url} return {resp.status_code}, {resp.text}")
    try:
        with open(output, "wb") as fs:
            progress = tqdm(
                total=int(resp.headers["Content-Length"]) if "Content-Length" in resp.headers else -1,
                ncols=80,
                desc=f"Downloading {name}",
            )
            for c in resp.iter_content(chunk_size=1024 * 1024):
                fs.write(c)
            progress.update(len(c))
            progress.close()
    except Exception:
        os.remove(output)
        raise

    return output


def load_vocab(name=None, tag=None, no_cache=False, cache_dir=None):
    import torch

    if name is None:
        name = "bpe_encoder"

    model_path = name
    if model_path and (not os.path.exists(model_path)) and not (("/" in model_path) or ("\\" in model_path)):
        _tag = tag
        if _tag is None:
            _tag = "latest"
        if not cache_dir:
            cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
        os.makedirs(cache_dir, exist_ok=True)
        out_dir = os.path.join(cache_dir, name)
        model_path = os.path.join(out_dir, "bpe_encoder.bin")
        if (not os.path.exists(model_path)) or no_cache:
            asset = download_asset(name + ".zip", tag=tag, no_cache=no_cache, cache_dir=cache_dir)
            with ZipFile(asset, "r") as zipf:
                for zip_info in zipf.infolist():
                    if zip_info.filename[-1] == "/":
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zipf.extract(zip_info, out_dir)
    elif not model_path:
        return None, None

    encoder_state = torch.load(model_path)
    return encoder_state


class GPT2Tokenizer(object):
    """    A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer

  Args:
    vocab_file (:obj:`str`, optional):
      The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases <https://github.com/microsoft/DeBERTa/releases>`_, \
              e.g. "bpe_encoder", default: `None`.

          If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file is a \
          state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. \
          The difference between our wrapped GPT2 tokenizer and RoBERTa wrapped tokenizer are,

          - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a sentence. We use `[CLS]` and `[SEP]` as the `start` and `end`\
              token of input sentence which is the same as `BERT`.

          - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0, `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264

    do_lower_case (:obj:`bool`, optional):
      Whether to convert inputs to lower case. **Not used in GPT2 tokenizer**.

    special_tokens (:obj:`list`, optional):
      List of special tokens to be added to the end of the vocabulary.


  """

    def __init__(self, vocab_file=None, do_lower_case=True, special_tokens=None):
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

        self.gpt2_encoder = load_vocab(vocab_file)
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
        """Convert an input text to tokens.

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
        """Convert list of tokens to ids.
        Args:
          tokens (:obj:`list<str>`): list of tokens

        Returns:
          List of ids
        """

        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        """Convert list of ids to tokens.
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
        """Decode list of tokens to text strings.
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
        """Adds a special token to the dictionary.
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
        """Adds a word to the dictionary.
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

    def save_pretrained(self, path: str):
        import torch

        torch.save(self.gpt2_encoder, path)


class DebertaTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa tokenizer, which runs end-to-end tokenization: punctuation
    splitting + wordpiece

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = XxxTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.do_lower_case = do_lower_case
        self.gpt2_tokenizer = GPT2Tokenizer(vocab_file)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self.gpt2_tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text):
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self.gpt2_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.gpt2_tokenizer.sym(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return self.gpt2_tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

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
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0,
                )
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        ~
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        self.gpt2_tokenizer.save_pretrained(vocab_file)
        return (vocab_file,)
