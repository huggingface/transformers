# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Blueberry tokenizer based on Llama tokenizer"""

from typing import TYPE_CHECKING, List, Optional, Union

from ...tokenization_utils_base import AddedToken, EncodedInput, PreTokenizedInput
from ...utils import logging
from ..llama.tokenization_llama import LlamaTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # "dloring1988/blueberry": "https://huggingface.co/dloring1988/blueberry/resolve/main/tokenizer.model",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dloring1988/blueberry": 2048,
}


class BlueberryTokenizer(LlamaTokenizer):
    """
    Construct a Blueberry tokenizer. Based on byte-level BPE.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*):
            The token used for padding, if the model supports it.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. In this case, `model_file` must be passed.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to clean up the potential extra spaces in the text output.
        use_fast (`bool`, *optional*, defaults to `False`):
            Use a fast tokenizer.
        splitting_regex (`str`, *optional*, defaults to `r"\s+"`):
            A regex used to split raw input text into "words" that are then tokenized to subwords.
        non_concatenating_prefix_meta_tokens (`List[str]`, *optional*, defaults to `["##length"]`):
            A list of meta tokens that, when encountered, are not concatenated to the previous token when `add_bos_token=True`.
        legacy (`bool`, *optional*):
            Whether or not the tokenizer should run in legacy mode.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]