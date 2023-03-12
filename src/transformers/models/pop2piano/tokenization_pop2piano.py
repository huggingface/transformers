# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Pop2Piano."""

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_config":"tokenizer_config.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_config": {
        "susnato/pop2piano_dev": "https://huggingface.co/susnato/pop2piano_dev/blob/main/tokenizer_config.json",
    },
}

class Pop2PianoTokenizer(PreTrainedTokenizer):
    """
    Construct a Whisper tokenizer.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
     Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*, defaults to `None`):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftranscript|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    # model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_config,
        unk_token=None,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):

        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            pad_token=pad_token,
            **kwargs,
        )

        with open(tokenizer_config, encoding="utf-8") as f:
            self.tokenizer_cfg = json.load(f)

    def get_vocab(self):
        return self.tokenizer_cfg["vocab_size_time"] + self.tokenizer_cfg["vocab_size_velocity"] + \
            self.tokenizer_cfg["vocab_size_note"] + self.tokenizer_cfg["vocab_size_special"]
