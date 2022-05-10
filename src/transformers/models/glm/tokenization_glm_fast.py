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
from typing import List, Optional

from tokenizers import normalizers

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_glm import GLMTokenizer
from typing import TYPE_CHECKING, List, Optional, Tuple

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "shunxing1234/GLM": "https://huggingface.co/shunxing1234/GLM/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "shunxing1234/GLM": "https://huggingface.co/shunxing1234/GLM/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "shunxing1234/GLM": 1024,
}


class GLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GLM tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = GLMTokenizer

    def __init__(
            self,
            vocab_file=None,
            do_lower_case=True,
            max_len=None,
            never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),
            pad_token='[PAD]',
            eos_token='[PAD]',
            sep_token='[SEP]',
            ENC_token='[CLS]',
            unk_token='[UNK]',
            MASK_token='[MASK]',
            sop_token='<|startofpiece|>',
            eop_token='<|endofpiece|>',
            gMASK_token='[gMASK]',
            sMASK_token='[sMASK]',
            **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            max_len=max_len,
            never_split=never_split,
            pad_token=pad_token,
            eos_token=eos_token,
            sep_token=sep_token,
            ENC_token=ENC_token,
            unk_token=unk_token,
            MASK_token=MASK_token,
            sop_token=sop_token,
            eop_token=eop_token,
            gMASK_token=gMASK_token,
            sMASK_token=sMASK_token,
            **kwargs
        )

        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (normalizer_state.get("lowercase", do_lower_case) != do_lower_case):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
