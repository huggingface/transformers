# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for YuE."""

from typing import Any, Optional

import sentencepiece as spm

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

# original in https://github.com/multimodal-art-projection/YuE/blob/main/inference/mm_tokenizer_v0.2_hf/tokenizer.model

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


@requires(backends=("sentencepiece",))
class YuETokenizer(PreTrainedTokenizer):
    """
    Construct YuE tokenizer based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token=None,
        eos_token=None,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)

        self.vocab_file = vocab_file

        self.sp_model.Load(self.vocab_file)

        special_tokens = ["<SOA>", "<EOA>", "<xcodec>", "<stage_1>", "<stage_2>"]

        if additional_special_tokens is None:
            additional_special_tokens = special_tokens
        else:
            additional_special_tokens = list(set(special_tokens + additional_special_tokens))

        unk_token = AddedToken(unk_token, special=True, normalized=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, special=True, normalized=False) if isinstance(pad_token, str) else pad_token
        additional_special_tokens = [
            AddedToken(token, special=True, normalized=False) for token in additional_special_tokens
        ]

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.soa_token_id = self.convert_tokens_to_ids("<SOA>")
        self.eoa_token_id = self.convert_tokens_to_ids("<EOA>")
        self.xcodec_token_id = self.convert_tokens_to_ids("<xcodec>")

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace("▁", " ").strip()
