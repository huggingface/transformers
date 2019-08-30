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
import sys
import unicodedata
from io import open

import sacremoses as sm

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
        'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-vocab.json",
        'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-vocab.json",
    }
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
        'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-merges.txt",
        'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-merges.txt",
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
    'xlm-mlm-17-1280': 512,
    'xlm-mlm-100-1280': 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    'xlm-mlm-en-2048': {"do_lowercase_and_remove_accent": True},
    'xlm-mlm-ende-1024': { "do_lowercase_and_remove_accent": True,
                            "id2lang": { "0": "de",
                                        "1": "en"},
                           "lang2id": { "de": 0,
                                        "en": 1 }},
    'xlm-mlm-enfr-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "fr"},
                           "lang2id": { "en": 0,
                                        "fr": 1 }},
    'xlm-mlm-enro-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "ro"},
                           "lang2id": { "en": 0,
                                        "ro": 1 }},
    'xlm-mlm-tlm-xnli15-1024': { "do_lowercase_and_remove_accent": True,
                                 "id2lang": {   "0": "ar",
                                                "1": "bg",
                                                "2": "de",
                                                "3": "el",
                                                "4": "en",
                                                "5": "es",
                                                "6": "fr",
                                                "7": "hi",
                                                "8": "ru",
                                                "9": "sw",
                                                "10": "th",
                                                "11": "tr",
                                                "12": "ur",
                                                "13": "vi",
                                                "14": "zh"},
                                 "lang2id": {   "ar": 0,
                                                "bg": 1,
                                                "de": 2,
                                                "el": 3,
                                                "en": 4,
                                                "es": 5,
                                                "fr": 6,
                                                "hi": 7,
                                                "ru": 8,
                                                "sw": 9,
                                                "th": 10,
                                                "tr": 11,
                                                "ur": 12,
                                                "vi": 13,
                                                "zh": 14 }},
    'xlm-mlm-xnli15-1024': { "do_lowercase_and_remove_accent": True,
                             "id2lang": {   "0": "ar",
                                                "1": "bg",
                                                "2": "de",
                                                "3": "el",
                                                "4": "en",
                                                "5": "es",
                                                "6": "fr",
                                                "7": "hi",
                                                "8": "ru",
                                                "9": "sw",
                                                "10": "th",
                                                "11": "tr",
                                                "12": "ur",
                                                "13": "vi",
                                                "14": "zh"},
                                 "lang2id": {   "ar": 0,
                                                "bg": 1,
                                                "de": 2,
                                                "el": 3,
                                                "en": 4,
                                                "es": 5,
                                                "fr": 6,
                                                "hi": 7,
                                                "ru": 8,
                                                "sw": 9,
                                                "th": 10,
                                                "tr": 11,
                                                "ur": 12,
                                                "vi": 13,
                                                "zh": 14 }},
    'xlm-clm-enfr-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "en",
                                        "1": "fr"},
                           "lang2id": { "en": 0,
                                        "fr": 1 }},
    'xlm-clm-ende-1024': { "do_lowercase_and_remove_accent": True,
                           "id2lang": { "0": "de",
                                        "1": "en"},
                           "lang2id": { "de": 0,
                                        "en": 1 }},
    'xlm-mlm-17-1280': {"do_lowercase_and_remove_accent": False},
    'xlm-mlm-100-1280': {"do_lowercase_and_remove_accent": False},
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


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    text = ' '.join(text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output).lower().split(' ')


def replace_unicode_punct(text):
    '''
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    '''
    text = text.replace('，', ',')
    text = re.sub(r'。\s*', '. ', text)
    text = text.replace('、', ',')
    text = text.replace('”', '"')
    text = text.replace('“', '"')
    text = text.replace('∶', ':')
    text = text.replace('：', ':')
    text = text.replace('？', '?')
    text = text.replace('《', '"')
    text = text.replace('》', '"')
    text = text.replace('）', ')')
    text = text.replace('！', '!')
    text = text.replace('（', '(')
    text = text.replace('；', ';')
    text = text.replace('１', '"')
    text = text.replace('」', '"')
    text = text.replace('「', '"')
    text = text.replace('０', '0')
    text = text.replace('３', '3')
    text = text.replace('２', '2')
    text = text.replace('５', '5')
    text = text.replace('６', '6')
    text = text.replace('９', '9')
    text = text.replace('７', '7')
    text = text.replace('８', '8')
    text = text.replace('４', '4')
    text = re.sub(r'．\s*', '. ', text)
    text = text.replace('～', '~')
    text = text.replace('’', '\'')
    text = text.replace('…', '...')
    text = text.replace('━', '-')
    text = text.replace('〈', '<')
    text = text.replace('〉', '>')
    text = text.replace('【', '[')
    text = text.replace('】', ']')
    text = text.replace('％', '%')
    return text


def remove_non_printing_char(text):
    '''
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    '''
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            continue
        output.append(char)
    return "".join(output)


def romanian_preprocessing(text):
    '''Sennrich's WMT16 scripts for Romanian preprocessing, used by model `xlm-mlm-enro-1024`'''
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py
    text = text.replace("\u0218", "S").replace("\u0219", "s") #s-comma
    text = text.replace("\u021a", "T").replace("\u021b", "t") #t-comma
    text = text.replace("\u0102", "A").replace("\u0103", "a")
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")
    return text


class XLMTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer for XLM

        - Moses preprocessing & tokenization for most supported languages

        - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP)

        - (optionally) lower case & normalize all inputs text

        - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
        (ex: "__classify__") to a vocabulary
        
        - `lang2id` attribute maps the languages supported by the model with their ids if provided (automatically set for pretrained vocabularies)

        - `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies)

        - `do_lowercase_and_remove_accent` controle lower casing and accent (automatically set for pretrained vocabularies)
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", bos_token="<s>",
                 sep_token="</s>", pad_token="<pad>", cls_token="</s>",
                 mask_token="<special1>", additional_special_tokens=["<special0>",
                 "<special1>", "<special2>", "<special3>", "<special4>", "<special5>",
                 "<special6>", "<special7>", "<special8>", "<special9>"],
                 lang2id=None, id2lang=None, do_lowercase_and_remove_accent=True,
                 **kwargs):
        super(XLMTokenizer, self).__init__(unk_token=unk_token, bos_token=bos_token,
                                           sep_token=sep_token, pad_token=pad_token,
                                           cls_token=cls_token, mask_token=mask_token,
                                           additional_special_tokens=additional_special_tokens,
                                           **kwargs)

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = dict()
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = dict()
        self.lang_with_custom_tokenizer = set(['zh', 'th', 'ja'])
        # True for current supported model (v1.2.0), False for XLM-17 & 100
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        self.encoder = json.load(open(vocab_file, encoding="utf-8"))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(merges_file, encoding='utf-8').read().split('\n')[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    def ja_tokenize(self, text):
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea
                self.ja_word_tokenizer = Mykytea.Mykytea('-model %s/local/share/kytea/model.bin' % os.path.expanduser('~'))
            except (AttributeError, ImportError) as e:
                logger.error("Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper (https://github.com/chezou/Mykytea-python) with the following steps")
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                raise e
        return list(self.ja_word_tokenizer.getWS(text))

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

    def _tokenize(self, text, lang='en', bypass_tokenizer=False):
        """
        Tokenize a string given language code. For Chinese, Japanese and Thai, we use a language specific tokenizerself. Otherwise, we use Moses.

        Details of tokenization:
        - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`
        - [pythainlp](https://github.com/PyThaiNLP/pythainlp): Thai tokenizer
            - Install with `pip install pythainlp`
        - [kytea](https://github.com/chezou/Mykytea-python): Japanese tokenizer, wrapper of [KyTea](https://github.com/neubig/kytea)
            - Install with the following steps:
            ```
            git clone git@github.com:neubig/kytea.git && cd kytea
            autoreconf -i
            ./configure --prefix=$HOME/local
            make && make install
            pip install kytea
            ```
        - [jieba](https://github.com/fxsjy/jieba): Chinese tokenizer *
            - Install with `pip install jieba`

        \* The original XLM used [Stanford Segmenter](https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip).
        However, the wrapper (`nltk.tokenize.stanford_segmenter`) is slow due to JVM overhead, and it will be deprecated.
        Jieba is a lot faster and pip-installable. Note there is some mismatch with the Stanford Segmenter. It should be fine
        if you fine-tune the model with Chinese supervisionself. If you want the same exact behaviour, use the original XLM
        [preprocessing script](https://github.com/facebookresearch/XLM/tree/master/tools) to tokenize the sentence externally,
        and set `bypass_tokenizer=True` to bypass the tokenizer.

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)  (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error("Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model.")
        if bypass_tokenizer:
            text = text.split()
        elif lang not in self.lang_with_custom_tokenizer:
            text = self.moses_pipeline(text, lang=lang)
            # TODO: make sure we are using `xlm-mlm-enro-1024`, since XLM-100 doesn't have this step
            if lang == 'ro':
                text = romanian_preprocessing(text)
            text = self.moses_tokenize(text, lang=lang)
        elif lang == 'th':
            text = self.moses_pipeline(text, lang=lang)
            try:
                if 'pythainlp' not in sys.modules:
                    from pythainlp.tokenize import word_tokenize as th_word_tokenize
            except (AttributeError, ImportError) as e:
                logger.error("Make sure you install PyThaiNLP (https://github.com/PyThaiNLP/pythainlp) with the following steps")
                logger.error("1. pip install pythainlp")
                raise e
            text = th_word_tokenize(text)
        elif lang == 'zh':
            try:
                if 'jieba' not in sys.modules:
                    import jieba
            except (AttributeError, ImportError) as e:
                logger.error("Make sure you install Jieba (https://github.com/fxsjy/jieba) with the following steps")
                logger.error("1. pip install jieba")
                raise e
            text = ' '.join(jieba.cut(text))
            text = self.moses_pipeline(text, lang=lang)
            text = text.split()
        elif lang == 'ja':
            text = self.moses_pipeline(text, lang=lang)
            text = self.ja_tokenize(text)
        else:
            raise ValueError('It should not reach here')

        if self.do_lowercase_and_remove_accent and not bypass_tokenizer:
            text = lowercase_and_remove_accent(text)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(' ')])

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
