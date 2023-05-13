# coding=utf-8
# Copyright 2022 x-plug and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for MplugOwl."""

from ...utils import logging
from ..llama.tokenization_llama import LlamaTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "MAGAer13/mplug-owl-llama-7b": "https://huggingface.co/MAGAer13/mplug-owl-llama-7b/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "MAGAer13/mplug-owl-llama-7b": 1024,
}


class MplugOwlTokenizer(LlamaTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        sp_model_kwargs=None,
        add_bos_token=False,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            unk_token,
            bos_token,
            eos_token,
            pad_token,
            sp_model_kwargs,
            add_bos_token,
            add_eos_token,
            clean_up_tokenization_spaces,
            **kwargs,
        )
        self.eod_id = self.eos_token_id
