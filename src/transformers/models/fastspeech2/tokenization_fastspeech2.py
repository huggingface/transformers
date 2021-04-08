# coding=utf-8
# Copyright 2021 The Ontocord team, the G2P, Melgan, Tacotron and Fastspeech2 Authors, and the HuggingFace Inc. team. All rights reserved.
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

# This software is based on other open source code. A huge thanks to
# the Huggingface team, and the authors of the Fastspeech2 and Melgan
# papers and the following authors who originally implemented the
# various modules, from which this code is based:

# Chung-Ming Chien's Fastspeech2 implementation is under the MIT license: https://github.com/ming024/FastSpeech2
# Seung-won Park 박승원's Meglan implementation is under BSD-3 license: https://github.com/seungwonpark/melgan
# Kyubyong Park's G2P implementation is under the Apache 2 license: https://github.com/Kyubyong/g2p, and also here for pytorch specifics https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb
# Some of the Text-preprocessing is from Tacotron https://github.com/keithito/tacotron 

"""Tokenization class for FastSpeech2."""

import json
import os
import sys
import warnings
from itertools import groupby, chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from transformers.file_utils import PaddingStrategy, TensorType, add_end_docstrings
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
import re
import inflect
import re

""" from https://github.com/keithito/tacotron 

    Cleaners are transformations that run over the input text at both training and eval time.

    Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
    hyperparameter. Some cleaners are English-specific. You'll typically want to use:
      1. "english_cleaners" for English text
      2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
      3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
"""

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


class FastSpeech2Tokenizer(PreTrainedTokenizer):


    """
    Constructs a Fastspeech tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains some of the main methods.
    Users should refer to the superclass for more information regarding such methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the .encoderulary.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the .encoderulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for defining the end of a word.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.

        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {}
    pretrained_init_configuration = {}
    max_model_input_sizes = {"ontocord/fastspeech2": sys.maxsize}
    model_input_names = ["input_ids", "attention_mask"]
    _base_letters = "abcdefghijklmnopqrstuvwxyz"
    _punctuation = '!\'(),.:;?'
    _special = '-'
    _silences = ['@sp', '@spn', '@sil']
    # Regular expression matching <> text
    _bracket_re = re.compile(r'(.*?)\<([^>]+)\>(.*)')

    _whitespace_re = re.compile(r'\s+')
    _alt_re = re.compile(r'\([0-9]+\)')
    _inflect = inflect.engine()
    _comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
    _decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
    _pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
    _dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
    _ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
    _number_re = re.compile(r'[0-9]+')

    def __init__(
        self,
        text_cleaners = ['english_cleaners'],
        vocab_file="vocab.json",
        abbreviations = [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort'),
            ],
        do_basic_tokenize=True,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token=' ',
        do_lower_case=True,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        self.text_cleaners = text_cleaners
        self.do_lower_case = do_lower_case
        self.word_delimiter_token = word_delimiter_token
        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in abbreviations]

        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.g2p = None

    def set_g2p(self, g2p):
      self.g2p = g2p

    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(self, name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)
        return text

    def _symbols_to_sequence(self, symbols):
        return [self.encoder[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(['@' + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.encoder and s is not '_' and s is not '~'


    def expand_abbreviations(self, text):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, ' ', text)


    def basic_cleaners(self, text):
        '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
        text = text.lower() if self.do_lower_case else text
        text = self.collapse_whitespace(text)
        return text


    def transliteration_cleaners(self, text):
        '''Pipeline for non-English text that transliterates to ASCII.'''
        text = text.lower() if self.do_lower_case else text
        text = self.collapse_whitespace(text)
        return text


    def english_cleaners(self, text):
        '''Pipeline for English text, including number and abbreviation expansion.'''
        text = text.lower() if self.do_lower_case else text
        text = self.collapse_whitespace(text)
        text = self.expand_numbers(text)
        text = self.expand_abbreviations(text)
        text = self.collapse_whitespace(text)
        return text

    def expand_numbers(self, text):
        def _remove_commas(m):
            return m.group(1).replace(',', '')


        def _expand_decimal_point(m):
            return m.group(1).replace('.', ' point ')

        def _expand_dollars(m):
            match = m.group(1)
            parts = match.split('.')
            if len(parts) > 2:
                return match + ' dollars'  # Unexpected format
            dollars = int(parts[0]) if parts[0] else 0
            cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            if dollars and cents:
                dollar_unit = 'dollar' if dollars == 1 else 'dollars'
                cent_unit = 'cent' if cents == 1 else 'cents'
                return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
            elif dollars:
                dollar_unit = 'dollar' if dollars == 1 else 'dollars'
                return '%s %s' % (dollars, dollar_unit)
            elif cents:
                cent_unit = 'cent' if cents == 1 else 'cents'
                return '%s %s' % (cents, cent_unit)
            else:
                return 'zero dollars'


        def _expand_ordinal(m):
            return self._inflect.number_to_words(m.group(0))


        def _expand_number(m):
            num = int(m.group(0))
            if num > 1000 and num < 3000:
                if num == 2000:
                    return 'two thousand'
                elif num > 2000 and num < 2010:
                    return 'two thousand ' + self._inflect.number_to_words(num % 100)
                elif num % 100 == 0:
                    return self._inflect.number_to_words(num // 100) + ' hundred'
                else:
                    return self._inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
            else:
                return self._inflect.number_to_words(num, andword='')


        text = re.sub(self._comma_number_re, _remove_commas, text)
        text = re.sub(self._pounds_re, r'\1 pounds', text)
        text = re.sub(self._dollars_re, _expand_dollars, text)
        text = re.sub(self._decimal_number_re, _expand_decimal_point, text)
        text = re.sub(self._ordinal_re, _expand_ordinal, text)
        text = re.sub(self._number_re,  _expand_number, text)
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # right pad for g2p
        def g2p_pad(batch):
          seqlens = [len(b) for b in batch]
          maxlen = max(seqlens)
          src_input = [sample+[self.pad_token_id]*(maxlen-len(sample)) for sample in batch]       
          return torch.LongTensor(src_input), seqlens

        if self.do_lower_case:
            text = text.lower()
        text = self._clean_text(text, self.text_cleaners)
        text = text.replace(" ", self.word_delimiter_token).replace(".", " . ").replace(",", " , ").replace("-", " - ").replace("?", " ? ").replace("!", " ! ").replace(",", " , ").replace("'", " ' ").replace("\"", " \" ").replace(":", " : ").replace(";", " ; ").replace("(", " ( ").replace(")", " ) ").strip()
        text = self.collapse_whitespace(text)        

        sequence = []
        # Check for <> braces
        while len(text):
            m = self._bracket_re.match(text)
            if not m:
                sequence.append(text.split())
                break
            sequence.append(m.group(1).split())
            sequence.append(["<"+m.group(2)+">"])
            text = m.group(3)
        if self.g2p is not None:
          #convert each word using g2p
          batch_output = []
          batch=[]
          sequence2 = []
          for i, segment in enumerate(sequence):
            for word in segment:
              if word[0] in ",.!?:;":
                sequence2.append(["@sp"])
              elif word==self.pad_token or word[0] in "()-' ":
                sequence2.append([word])
              elif (word[0] == "<" and word[-1] == ">"):
                pass
              else:
                ret=[]
                add_word = [self.encoder[a] for a in word if self.encoder.get(a) is not None]
                if add_word:
                  batch.append(add_word+[self.eos_token_id])
                  batch_output.append(ret)
                  sequence2.append(ret)
          if batch != []:
            is_training = self.g2p.training
            self.g2p.eval()
            with torch.no_grad():
              batch, seqlens = g2p_pad(batch)
              _, vals, _ = self.g2p(batch, seqlens)
            vals = vals.to('cpu').numpy().tolist()
            for ret, val in zip(batch_output, vals):
              if self.eos_token_id in val:
                val = val[:val.index(self.eos_token_id)]
              ret.extend([self.decoder[v] for v in val])
            if is_training: self.g2p.train()
            sequence = sequence2
            sequence = " ".join(list(chain(*sequence))).split()
        else:
          sequence = " ".join(list(chain(*sequence)))
          sequence = list(sequence)
        return sequence

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        return (vocab_file,)

    

    
