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

import os
import unicodedata

import sentencepiece as sp
import six


VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}

__all__ = ["SPMTokenizer"]


class SPMTokenizer:
    def __init__(
        self, vocab_file, do_lower_case=False, special_tokens=None, bpe_dropout=0, split_by_punct=False, **kwargs
    ):
        self.split_by_punct = split_by_punct
        spm = sp.SentencePieceProcessor()
        assert os.path.exists(vocab_file)
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        # Token map
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.id_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        _special_tokens = ["[MASK]", "[SEP]", "[PAD]", "[UNK]", "[CLS]"]
        self.special_tokens = []
        if special_tokens is not None:
            _special_tokens.extend(special_tokens)
        for t in _special_tokens:
            self.add_special_token(t)

        self.spm = spm

    def tokenize(self, text):
        pieces = self._encode_as_pieces(text)

        def _norm(x):
            if x not in self.vocab or x == "<unk>":
                return "[UNK]"
            else:
                return x

        pieces = [_norm(p) for p in pieces]
        return pieces

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] if t in self.vocab else 1 for t in tokens]

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            return self.spm.decode_pieces([t for t in tokens if t not in self.special_tokens])
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.id_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        if (
            len(token) == 1
            and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "[MASK]"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode_as_pieces(w) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode_as_pieces(text)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save_pretrained(self, path: str, filename_prefix: str = None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)


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


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
