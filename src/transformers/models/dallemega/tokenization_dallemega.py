# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import html
import json
import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "word_frequency_file": "enwiki-words-frequency.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "dalle-mini/dalle-mega": "https://huggingface.co/dalle-mini/dalle-mega/resolve/main/vocab.json",
    },
    "merges_file": {
        "dalle-mini/dalle-mega": "https://huggingface.co/dalle-mini/dalle-mega/resolve/main/merges.txt",
    },
    "word_frequency_file": {
        "dalle-mini/dalle-mega": (
            "https://huggingface.co/dalle-mini/dalle-mini/resolve/main/enwiki-words-frequency.txt"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dalle-mini/dalle-mega": 1024,
}


class HashtagProcessor:
    # Adapted from wordninja library
    # We use our wikipedia word count + a good heuristic to make it work
    def __init__(self, word_frequency_file):
        self._word_cost = (
            line.split()[0] for line in Path(word_frequency_file).read_text(encoding="utf8").splitlines()
        )
        self._word_cost = {str(k): math.log(float(i + 1)) for i, k in enumerate(self._word_cost)}
        self._max_word = max(len(x) for x in self._word_cost.keys())
        self._SPLIT_RE = re.compile("[^a-zA-Z0-9']+")

    def __call__(self, s):
        """Uses dynamic programming to infer the location of spaces in a string without spaces."""
        l = [self._split(x) for x in self._SPLIT_RE.split(s)]
        return " ".join([item for sublist in l for item in sublist])

    def _split(self, s):
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self._max_word) : i]))
            return min((c + self._word_cost.get(s[i - k - 1 : i].lower(), 9e999), k + 1) for k, c in candidates)

        # Build the cost array
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            newToken = True
            if not s[i - k : i] == "'":  # ignore a lone apostrophe
                if len(out) > 0:
                    # re-attach split 's and split digits
                    if out[-1] == "'s" or (s[i - 1].isdigit() and out[-1][0].isdigit()):  # digit followed by digit
                        out[-1] = s[i - k : i] + out[-1]  # combine current token with previous token
                        newToken = False

            if newToken:
                out.append(s[i - k : i])

            i -= k

        return reversed(out)


class Normalizer:
    "Normalize text"

    def __init__(self, word_frequency_file):
        try:
            import emoji
            import ftfy
            from unidecode import unidecode
        except ImportError:
            raise ImportError(
                "Normalizer requires emoji, ftfy, and unidecode  but one or several of them were not found in your"
                " environment. You can install it with ```pip install -U emoji ftfy unidecode```"
            )

        self._hashtag_processor = HashtagProcessor(word_frequency_file)
        self._re_ignore_chars = r"[_#\\]"
        # based on wiki word occurrence
        self.person_token = [("a person", 282265), ("someone", 121194), ("somebody", 12219)]
        self.temp_token = "xtokx"  # avoid repeating chars

    def __call__(self, t):
        # fix some characters
        text = ftfy.fix_text(text)
        # fix html
        text = self.fix_html(text)
        # decode emojis (would be removed by unidecode)
        text = emoji.demojize(text)
        # decode and simplify text: see unidecode library
        text = unidecode(text)
        # lower case
        text = self.text.lower()
        # replace <PERSON> (for CC12M)
        text = self.replace_person_token(text)
        # remove wiki reference (for WIT)
        text = self.remove_wiki_ref(text)
        # remove html tags
        text = self.remove_html_tags(text)
        # remove urls
        text = self.remove_urls(text)
        # remove commas in numbers
        text = self.remove_comma_numbers(text)
        # handle dots in numbers and quotes - Part 1
        text = self.pre_process_dot_numbers(text)
        text = self.pre_process_quotes(text)
        text = self.pre_process_dates(text)
        # handle special characters
        text = self.handle_special_chars(text)
        # handle hashtags
        text = self.expand_hashtags(text)
        # ignore useless characters
        text = self.ignore_chars(text)
        # simplify quotes
        text = self.simplify_quotes(text)
        # all punctuation becomes commas
        text = self.replace_punctuation_with_commas(text)
        # handle dots in numbers and quotes - Part 2
        text = self.post_process_dot_numbers(text)
        text = self.post_process_quotes(text)
        text = self.post_process_dates(text)
        # handle repeating characters
        text = self.remove_repeating_chars(text)
        # merge quotes
        text = self.merge_quotes(text)
        # merge commas
        text = self.merge_commas(text)
        # remove multiple spaces
        text = self.remove_extra_spaces(text)
        # remove first and last comma
        text = self.remove_first_last_commas(text)
        # always start with a space
        return f" {text}"

    def replace_person_token(self, text):
        "Used for CC12M"
        text = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", text)
        while "<person>" in text:
            text = text.replace("<person>", f" {random.choices(*tuple(zip(*self.person_token)))[0]} ", 1)
        return text

    def fix_html(self, text):
        # from OpenAI CLIP
        return html.unescape(html.unescape(text))

    def replace_punctuation_with_commas(self, text):
        return re.sub("[()[\].,|:;?!=+~\-\/{}]", ",", text)

    def simplify_quotes(self, text):
        return re.sub("""['"`]""", ' " ', text)

    def merge_quotes(self, text):
        return re.sub('(\s*"+\s*)+', ' " ', text)

    def remove_comma_numbers(self, text):
        def _f(text):
            return re.sub("(\d),(\d{3})", r"\1\2", text)

        return _f(_f(text))

    def pre_process_dot_numbers(self, text):
        return re.sub("(\w)\.(\w)", rf"\1{self.temp_token}dot{self.temp_token}\2", text)

    def post_process_dot_numbers(self, text):
        return re.sub(f"{self.temp_token}dot{self.temp_token}", ".", text)

    def pre_process_quotes(self, text):
        # allows quotes only for 's, 't, 'd, 'm, 'll, 're, 've
        return re.subext(r"'(?=([stdm]|(ll)|(re)|(ve)|(ll))\b)", rf"{self.temp_token}quote{self.temp_token}", text)

    def post_process_quotes(self, text):
        return re.sub(f"{self.temp_token}quote{self.temp_token}", "'", text)

    def pre_process_dates(self, text):
        return re.sub("(\d)/(\d)", rf"\1{self.temp_token}slash{self.temp_token}\2", text)

    def post_process_dates(self, text):
        return re.sub(f"{self.temp_token}slash{self.temp_token}", "/", text)

    def merge_commas(self, text):
        return re.sub("(\s*,+\s*)+", ", ", text)

    def add_space_after_commas(self, text):
        return re.sub(",", ", ", text)

    def handle_special_chars(self, text):
        "Handle special characters"
        # replace "-" with a space when between words without space
        t = re.sub("(\w)-(\w)", r"\1 \2", text)
        # always add space around some characters
        return re.sub("([%&\/$*])", r" \1 ", text)

    def expand_hashtags(self, text):
        "Remove # and try to split words"
        return re.sub("#(\w+)", lambda m: self._hashtag_processor(m.group(1)), text)

    def ignore_chars(self, text):
        "Ignore useless characters"
        return re.sub(self._re_ignore_chars, " ", text)

    def remove_extra_spaces(self, text):
        "Remove extra spaces (including \t and \n)"
        return re.sub("\s+", " ", text)

    def remove_repeating_chars(self, text):
        "If the same character is present 4+ times (not 3 because of roman 'VIII'), replace with single instance"
        return re.sub(r"(\D)(\1{3,})", r"\1", text)

    def remove_urls(self, text):
        return re.sub(r"http\S+", "", text)

    def remove_html_tags(self, text):
        return re.sub("<[^<]+?>", " ", text)

    def remove_first_last_commas(self, text):
        text = text.strip()
        text = text[:-1] if text and text[-1] == "," else text
        text = text[1:] if text and text[0] == "," else text
        return text.strip()

    def remove_wiki_ref(self, text):
        text = re.sub(r"\A\s*\[\d+\]", "", text)
        return re.sub(r"\[\d+\]\s*\Z", "", text)


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class DalleMegaTokenizer(PreTrainedTokenizer):
    """
    Constructs a DalleMega tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import DalleMegaTokenizer
    >>> tokenizer = DalleMegaTokenizer.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")['input_ids']
    [0, 31414, 232, 2]
    >>> tokenizer(" Hello world")['input_ids']
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (DalleMega tokenizer detect beginning of words by the preceding space).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        word_frequency_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        normalize_text=True,
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

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
        self.add_prefix_space = add_prefix_space
        self.word_frequency_file = word_frequency_file

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.normalize_text = normalize_text
        if normalize_text:
            self.normalizer = Normalizer()       

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

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

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DalleMega sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
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
        special tokens using the tokenizer `prepare_for_model` method.

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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. DalleMega does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
