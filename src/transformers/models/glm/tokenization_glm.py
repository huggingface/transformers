# coding=utf-8
# Copyright 2022 shunxing1234 and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GLM."""
import json
import sys
from typing import List, Optional
import os

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from functools import lru_cache
import regex as re


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "shunxing1234/GLM-base-cased": "https://huggingface.co/shunxing1234/GLM-base-cased/resolve/main/vocab.json",
    },
    "merges_file": {
        "shunxing1234/GLM-base-cased": "https://huggingface.co/shunxing1234/GLM-base-cased/resolve/main/merges.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "shunxing1234/GLM-base-cased": 1024,
}


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
    _chr = chr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [_chr(n) for n in cs]
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


class GLMTokenizer(PreTrainedTokenizer):
    """
    Construct a GLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors='replace',
            unk_token="[UNK]",
            ENC_token="[CLS]",
            eos_token="<|endoftext|>",
            sop_token='<|startofpiece|>',
            eop_token='<|endofpiece|>',
            gMASK_token='[gMASK]',
            sMASK_token='[sMASK]',
            add_prefix_space=False,
            **kwargs
    ):
        ENC_token = AddedToken(ENC_token, lstrip=False, rstrip=False) if isinstance(ENC_token, str) else ENC_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        sop_token = AddedToken(sop_token, lstrip=False, rstrip=False) if isinstance(sop_token, str) else sop_token
        eop_token = AddedToken(eop_token, lstrip=False, rstrip=False) if isinstance(eop_token, str) else eop_token
        gMASK_token = AddedToken(gMASK_token, lstrip=False, rstrip=False) if isinstance(gMASK_token, str) else gMASK_token
        sMASK_token = AddedToken(sMASK_token, lstrip=False, rstrip=False) if isinstance(sMASK_token, str) else sMASK_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(ENC_token=ENC_token, eos_token=eos_token,
                         unk_token=unk_token, sop_token=sop_token, eop_token=eop_token,
                         gMASK_token=gMASK_token, sMASK_token=sMASK_token, add_prefix_space=add_prefix_space, **kwargs)

        "Initialisation"
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

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @property
    def vocab_size(self):
        "Returns vocab size done"
        return len(self.encoder)

    def get_vocab(self):
        "Returns vocab as a dict done"
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
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
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text, process_fn = None):
        """ Tokenize a string.  done"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = []
        for token in re.findall(self.pat, processed_text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a sequence of tokens into ids using the vocab. done"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts a sequence of ids in BPE tokens using the vocab. done"""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. done"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory. done"""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["merges_file"])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

