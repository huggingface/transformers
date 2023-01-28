# coding=utf-8
# Copyright 2022 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ErnieM."""
#Copied from original paddlenlp repository.(https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie_m/fast_tokenizer.py)

import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union

from ...utils.log import logger
from ..tokenizer_utils_base import PaddingStrategy, TensorType, TruncationStrategy
from ..tokenizer_utils_fast import PretrainedFastTokenizer
from .tokenizer import ErnieMTokenizer

VOCAB_FILES_NAMES = {
    "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "vocab.txt",
    "tokenizer_file": "tokenizer.json",
}

SPIECE_UNDERLINE = "▁"


class ErnieMFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    slow_tokenizer_class = ErnieMTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration

    def __init__(
        self,
        vocab_file,
        sentencepiece_model_file,
        tokenizer_file=None,
        do_lower_case=True,
        encoding="utf8",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            vocab_file,
            sentencepiece_model_file=sentencepiece_model_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.do_lower_case = do_lower_case
        self.encoding = encoding
        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your faster tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_sentencepiece_model_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sentencepiece_model_file"],
        )
        if os.path.abspath(self.sentencepiece_model_file) != os.path.abspath(out_sentencepiece_model_file):
            copyfile(self.sentencepiece_model_file, out_sentencepiece_model_file)
        return (out_sentencepiece_model_file,)

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        return_position_ids: bool = True,
        return_token_type_ids: bool = False,
        return_attention_mask: bool = True,
        return_length: bool = False,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_dict: bool = True,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs
    ):
        return super(ErnieMFastTokenizer, self).__call__(
            text=text,
            text_pair=text_pair,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            padding=padding,
            truncation=truncation,
            return_position_ids=return_position_ids,
            # Ernie-M model doesn't have token_type embedding.
            # So set "return_token_type_ids" to False.
            return_token_type_ids=False,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_dict=return_dict,
            return_offsets_mapping=return_offsets_mapping,
            add_special_tokens=add_special_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            verbose=verbose,
            **kwargs,
        )