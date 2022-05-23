# coding=utf-8
# Copyright 2022 ArEnSc and The HuggingFace Inc. team. All rights reserved.
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


""" CMUDict
    from https://github.com/keithito/tacotron 
    from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/common/text/cmudict.py d0d427d357816d565b9732b6c40275a2a46d00e7
    Adapted for Hugging Face
"""
from posixpath import split
import re
from pathlib import Path
from typing import Union,Dict,Tuple
from typing_extensions import NotRequired

from defusedxml import NotSupportedError

class CMUDict:

    valid_symbols = [
        'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
        'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
        'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
    def __init__(self, file_or_path=None, heteronyms_path=None, keep_ambiguous=True):
        self._valid_symbol_set = set(CMUDict.valid_symbols)
        self._alt_re = re.compile(r'\([0-9]+\)')
        self._entries = {}
        self._entries_reversed = {}
        self.heteronyms = []
        if file_or_path is not None:
            self.initialize(file_or_path, heteronyms_path, keep_ambiguous)

    def initialize(self, file_or_path, heteronyms_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            if not Path(file_or_path).exists():
                print("CMUdict missing")
            with open(file_or_path, encoding='latin-1') as f:
                entries = self._parse_cmudict(f)
        else:
            entries = self._parse_cmudict(file_or_path)

        if not keep_ambiguous:
            entries = {word: pron for word, pron in self._entries.items() if len(pron) == 1}
        
        self._entries = entries

        for key in entries.keys():
    
            arpabet_list = self._entries[key]
            
            if len(arpabet_list) > 1:
                for arpabet in arpabet_list:
                    self._entries_reversed[arpabet] = key
            else:
                self._entries_reversed[arpabet_list[0]] = key

        if heteronyms_path is not None:
            with open(heteronyms_path, encoding='utf-8') as f:
                self.heteronyms = [l.rstrip() for l in f]

    def __len__(self):
        if len(self._entries) == 0:
            raise ValueError("CMUDict not initialized")
        return len(self._entries)

    def lookup(self, word:str)->Union[List[str], None]:
        '''Returns list of ARPAbet pronunciations of the given word.'''
        if len(self._entries) == 0:
            raise ValueError("CMUDict not initialized")
        return self._entries.get(word.upper())

    def arpabet_token_lookup(self,arpabet:str)->Union[str, None]:
        """Returns a word for a arpabet pronunciations."""
        entry = self._entries_reversed.get(arpabet.upper())
        return entry

    def _parse_cmudict(self,file):
        cmudict = {}
        for line in file:
            if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
                parts = line.split('  ')
                word = re.sub(self._alt_re, '', parts[0])
                pronunciation = self._get_pronunciation(parts[1])
                if pronunciation:
                    if word in cmudict:
                        cmudict[word].append(pronunciation)
                    else:
                        cmudict[word] = [pronunciation]
        return cmudict

    def _get_pronunciation(self,s):
        parts = s.strip().split(' ')
        for part in parts:
            if part not in self._valid_symbol_set:
                return None
        return ' '.join(parts)

class SymbolEncoder:
    arpabet = ['@' + s for s in CMUDict.valid_symbols]
    def __init__(self,symbol_set:str='english_basic', file_or_path:str="./cmudict-0.7b.txt", heteronyms_path:str = "./heteronyms"):
        self.cmudict = CMUDict(file_or_path=file_or_path,heteronyms_path=heteronyms_path)
        # Regular expression matching text enclosed in curly braces for encoding
        self._curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
        # Regular expression matching words and not words
        self._words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]{1,2}|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")
        # Regular expression separating words enclosed in curly braces for cleaning
        self._arpa_re = re.compile(r'{[^}]+}|\S+')

        self.symbols = self._get_symbols(symbol_set=symbol_set)
        # Vocab here is the symbols and their ids
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
    
    def encode_token_to_id(self,token:str)->str:
        """
        Token is either a arpabet or a symbol
        """
        return self._symbol_to_id[token]
    
    def decode_id_to_token(self,index:int)->str:
        return self._id_to_symbol[index]

    def match_arpabet(self,text:str)->Tuple[bool,str,str,str]:
        matches = self._curly_re.match(text)
        # Regular Symbols # Arpabet # Remaining Text
        if not matches:
            return False, "","",""
        else:
            return True, matches.group(1),matches.group(2),matches.group(3)

    def _get_symbols(self,symbol_set='english_basic'):
        """ 
            from https://github.com/keithito/tacotron 
            from https://github.com/NVIDIA/DeepLearningExamples/blob/3e8897f7855ff475d69699e66c8f668f6191dfa4/PyTorch/SpeechSynthesis/FastPitch/common/text/symbols.py#L14
            Adapted For Hugging Face

            Should this symbol set be read from a file? Question....
        """
        if symbol_set == 'english_basic':
            _pad = '_'
            _punctuation = '!\'(),.:;? '
            _special = '-'
            _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            symbols = list(_pad + _special + _punctuation + _letters) + SymbolEncoder.arpabet
        elif symbol_set == 'english_basic_lowercase':
            _pad = '_'
            _punctuation = '!\'"(),.:;? '
            _special = '-'
            _letters = 'abcdefghijklmnopqrstuvwxyz'
            symbols = list(_pad + _special + _punctuation + _letters) + SymbolEncoder.arpabet
        elif symbol_set == 'english_expanded':
            _punctuation = '!\'",.:;? '
            _math = '#%&*+-/[]()'
            _special = '_@©°½—₩€$'
            _accented = 'áçéêëñöøćž'
            _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            symbols = list(_punctuation + _math + _special + _accented + _letters) + SymbolEncoder.arpabet
        else:
            raise Exception("{} symbol set does not exist".format(symbol_set))

        return symbols
        
"""Tokenization classes for FastPitch."""
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

from ...file_utils import requires_backends

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "fast-pitch": "https://huggingface.co/fast-pitch/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "fast-pitch": 1024,
}

class FastPitchTokenizer(PreTrainedTokenizer):
    """
    Construct a FastPitch tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.add_prefix_space = False
        self.symbol_encoder = SymbolEncoder()

    def vocab_size(self) -> int:
        "Returns vocab size which is the symbol size in our case our vocab is infinite"
        return len(self.symbol_encoder.symbols)

    def get_vocab(self) -> Dict[str,int]:
        "Returns vocab as a dict"
        return self.symbol_encoder.symbol_to_id

    def _tokenize(self, text)->List[str]:
        """Returns a tokenized string."""
        tokenized_string = []

        while len(text):
            success, symbols, arpabet, remainder = self.symbol_encoder.match_arpabet(text)
            if success:
                tokenized_string.extend(symbols)
                tokenized_string.extend(['@'+ arpabet for arpabet in arpabet.split(' ')]) 
                text = remainder
            else:
                tokenized_string.extend(text)
                break
        return tokenized_string
       
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an symbols using the vocab."""
        symbol_id = self.symbol_encoder.encode_token_to_id(token)
        return symbol_id

    def _convert_id_to_token(self, index):
        """Converts an index id in a token (str) using the vocab."""
        token = self.symbol_encoder.decode_id_to_token(index)
        return token

    def convert_tokens_to_string(self, tokens:List[str])->str:
        """Converts a sequence of tokens (string) in a single string.
        """ 
        pass

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        raise NotSupportedError

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A FastPitch sequence has the following format:

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
        return token_ids_0 

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
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. FastPitch does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        return len(token_ids_0) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        return (text, kwargs)

class FastPitchTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" FastPitch tokenizer (backed by HuggingFace's *tokenizers* library).

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
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs
    ):
        super().__init__(
            ByteLevelBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. FastPitch does
        not make use of token type ids, therefore a list of zeros is returned.

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
