# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HugginFace Inc. team.
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
import os
import re
import json
from tqdm import tqdm
import logging

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'openai-gpt': "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json",
}
PRETRAINED_MERGES_ARCHIVE_MAP = {
    'openai-gpt': "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'openai-gpt': 512,
}
VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt'

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
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

class OpenAIGPTTokenizer(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            merges_file = PRETRAINED_MERGES_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            merges_file = os.path.join(pretrained_model_name_or_path, MERGES_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
            resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    pretrained_model_name_or_path,
                    vocab_file, merges_file))
            return None
        if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
            logger.info("loading merges file {}".format(merges_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
            logger.info("loading merges file {} from cache at {}".format(
                merges_file, resolved_merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, resolved_merges_file, *inputs, **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, special_tokens=None, max_len=None):
        try:
            import ftfy
            import spacy
        except ImportError:
            raise ImportError("Please install ftfy and spacy to use OpenAI GPT tokenizer.")

        self.max_len = max_len if max_len is not None else int(1e12)
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.fix_text = ftfy.fix_text
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        if not special_tokens:
            self.special_tokens = {}
        else:
            self.special_tokens = dict((tok, len(self.encoder) + i) for i, tok in enumerate(special_tokens))

    def set_special_tokens(self, special_tokens):
        self.special_tokens = dict((tok, len(self.encoder) + i) for i, tok in enumerate(special_tokens))

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

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
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def tokenize(self, text):
        split_tokens = []
        text = self.nlp(text_standardize(self.fix_text(text)))
        for token in text:
            split_tokens.extend([t for t in self.bpe(token.text.lower()).split(' ')])
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.decoder[i])
        return tokens

    def decode(self, ids):
        """Converts a sequence of ids in a string."""
        tokens = self.convert_ids_to_tokens(ids)
        out_string = ''.join(tokens).replace('</w>', ' ')
        return out_string
