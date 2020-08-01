# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

import logging
from typing import List, Optional
from .tokenization_bert import BertTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer

from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES_EN = {"vocab_file": "vocab.txt"}
VOCAB_FILES_NAMES_CROSS_LINGUAL = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/prophetnet-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/prophetnet-large-uncased/vocab.txt",
        "microsoft/xprophetnet-large-wiki100-cased": "https://cdn.huggingface.co/microsoft/xprophetnet-large-wiki100-cased/sentencepiece.bpe.model"
    }
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/prophetnet-large-uncased": {"do_lower_case": True, "xprophetnet_tokenizer": False},
    "microsoft/xprophetnet-large-wiki100-cased": {"do_lower_case": False, "xprophetnet_tokenizer": True}
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/prophetnet-large-uncased": 512,
    "microsoft/xprophetnet-large-wiki100-cased": 512
}



class ProphetNetTokenizer(PreTrainedTokenizer):
    r"""
            ProphetNet inherit from BERT-tokenizer, xProphetNet  inherit from XLMR-tokenizer
        """

    vocab_files_names = VOCAB_FILES_NAMES_EN    # default english version rather than cross-lingual version.
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __new__(cls, **kwargs):
        xprophetnet_tokenizer = False if 'xprophetnet_tokenizer' not in kwargs.keys() else kwargs['xprophetnet_tokenizer']
        if xprophetnet_tokenizer:
            super_class = XLMRobertaTokenizer
        else:
            super_class = BertTokenizer
        cls = type(cls.__name__, (cls, super_class), {})
        if xprophetnet_tokenizer:
            cls.vocab_files_names = VOCAB_FILES_NAMES_CROSS_LINGUAL
        else:
            cls.vocab_files_names = VOCAB_FILES_NAMES_EN
        return super(ProphetNetTokenizer, cls).__new__(cls)

    def __init__(
        self,
        vocab_file,
        xprophetnet_tokenizer=False,
        do_lower_case=None,
        do_basic_tokenize=None,
        never_split=None,
        unk_token=None,
        sep_token=None,
        pad_token=None,
        cls_token=None,
        mask_token=None,
        tokenize_chinese_chars=None,
        bos_token=None,
        eos_token=None,
        **kwargs
    ):
        if not xprophetnet_tokenizer:
            # inherit from BERT tokenizer
            do_lower_case = True if do_lower_case is None else do_lower_case
            do_basic_tokenize = True if do_lower_case is None else do_basic_tokenize
            unk_token = "[UNK]" if unk_token is None else unk_token
            sep_token = "[SEP]" if sep_token is None else sep_token
            pad_token = "[PAD]" if pad_token is None else pad_token
            cls_token = "[SEP]" if cls_token is None else cls_token
            mask_token = "[MASK]" if mask_token is None else mask_token
            tokenize_chinese_chars = True if tokenize_chinese_chars is None else tokenize_chinese_chars
            super(ProphetNetTokenizer, self).__init__(
                vocab_file=vocab_file,
                do_lower_case=do_lower_case,
                do_basic_tokenize=do_basic_tokenize,
                never_split=never_split,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                tokenize_chinese_chars=tokenize_chinese_chars,
                xprophetnet_tokenizer=False,
                **kwargs
            )
            self.unique_no_split_tokens.append("[X_SEP]")
        else:
            # inherit from XLM-R tokenizer
            bos_token = "[SEP]" if bos_token is None else bos_token
            eos_token = "[SEP]" if eos_token is None else eos_token
            sep_token = "[SEP]" if sep_token is None else sep_token
            cls_token = "[SEP]" if cls_token is None else cls_token
            unk_token = "[UNK]" if unk_token is None else unk_token
            pad_token = "[PAD]" if pad_token is None else pad_token
            mask_token = "[MASK]" if mask_token is None else mask_token
            super(ProphetNetTokenizer, self).__init__(
                vocab_file=vocab_file,
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                xprophetnet_tokenizer=True,
                **kwargs
            )
            # Original fairseq vocab and spm vocab must be "aligned":
            # model    | '[PAD]'   | '[CLS]' | '<SEP>' | '[UNK]' | '[MASK]' | '[unused1]' | ......... | '[unused9]'  | ',' | '▁'
            # spm      | '<unk>'   | '<s>'   | '</s>'  | ','     | '.'   | '▁'  | 's' | '▁de'  | '-'   | '▁a'
            # put special tokens and [unused] tokens into the vocab
            self.fairseq_tokens_to_ids = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}
            for i in range(10):
                tok = '[unused{}]'.format(i)
                self.fairseq_tokens_to_ids[tok] = 5 + i

            # The first "real" token "," has position 15 in the embedding vocab and position 3 in the spm vocab
            self.fairseq_offset = 12
            self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
            for k in self.fairseq_tokens_to_ids.keys():
                self.unique_no_split_tokens.append(k)
        self.xprophetnet_tokenizer = xprophetnet_tokenizer

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A ProphetNet sequence has the following format:

        - single sequence: ``[SEP] X [SEP]``
        - pair of sequences: ``[SEP] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.sep_token_id] + token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        return sep + token_ids_0 + sep + token_ids_1 + sep



