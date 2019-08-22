# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import re
from io import open

from .tokenization_utils import PreTrainedTokenizer
from .tokenization_bert import BasicTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-vocab.json",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-vocab.json",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-vocab.json",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-vocab.json",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-vocab.json",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-vocab.json",
    },
    'merges_file':
    {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-merges.txt",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-merges.txt",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'xlm-mlm-en-2048': 512,
    'xlm-mlm-ende-1024': 512,
    'xlm-mlm-enfr-1024': 512,
    'xlm-mlm-enro-1024': 512,
    'xlm-mlm-tlm-xnli15-1024': 512,
    'xlm-mlm-xnli15-1024': 512,
    'xlm-clm-enfr-1024': 512,
    'xlm-clm-ende-1024': 512,
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
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

class XLMTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer for XLM, adapted from OpenAI BPE tokenizer. Peculiarities:

        - lower case all inputs

        - uses `SpaCy tokenizer <https://spacy.io/api/tokenizer/>`_ and \
        `ftfy <https://ftfy.readthedocs.io/en/latest/>`_ for pre-BPE tokenization if they are installed, \
        fallback to BERT's BasicTokenizer if not.

        - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
        (ex: "__classify__") to a vocabulary.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", bos_token="<s>",
                 sep_token="</s>", pad_token="<pad>", cls_token="</s>",
                 mask_token="<special1>", additional_special_tokens=["<special0>",
                 "<special1>", "<special2>", "<special3>", "<special4>", "<special5>",
                 "<special6>", "<special7>", "<special8>", "<special9>"], **kwargs):
        super(XLMTokenizer, self).__init__(unk_token=unk_token, bos_token=bos_token,
                                           sep_token=sep_token, pad_token=pad_token,
                                           cls_token=cls_token, mask_token=mask_token,
                                           additional_special_tokens=additional_special_tokens,
                                           **kwargs)
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

        self.encoder = json.load(open(vocab_file, encoding="utf-8"))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(merges_file, encoding='utf-8').read().split('\n')[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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

    def _tokenize(self, text):
        """ Tokenize a string. """
        split_tokens = []
        if self.fix_text is None:
            # Using BERT's BasicTokenizer
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend([t for t in self.bpe(token).split(' ')])
        else:
            # Using SpaCy & ftfy (original tokenization process of OpenAI GPT)
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend([t for t in self.bpe(token.text.lower()).split(' ')])
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ''.join(tokens).replace('</w>', ' ').strip()
        return out_string

    def add_special_tokens_single_sentence(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        An XLM sequence has the following format: [CLS] X [SEP]
        """
        return [self._convert_token_to_id(self.cls_token)] + token_ids + [self._convert_token_to_id(self.sep_token)]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        An XLM sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self._convert_token_to_id(self.sep_token)]
        cls = [self._convert_token_to_id(self.cls_token)]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return vocab_file, merge_file
