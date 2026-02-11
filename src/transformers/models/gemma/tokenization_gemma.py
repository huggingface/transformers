# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from typing import Optional, Union

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class GemmaTokenizer(TokenizersBackend):
    """
    Construct a fast Gemma tokenizer (backed by HuggingFace's tokenizers library).

    This tokenizer uses a BPE model with byte fallback, no prefix space, and a normalizer that replaces
    spaces with "▁".

    Args:
        tokenizer_file (`str`, optional):
            A tokenizers JSON file containing the serialization of a tokenizer.
        unk_token (`str`, optional, defaults to "<unk>"):
            The unknown token.
        bos_token (`str`, optional, defaults to "<bos>"):
            The beginning of sequence token.
        eos_token (`str`, optional, defaults to "<eos>"):
            The end of sequence token.
        pad_token (`str`, optional, defaults to "<pad>"):
            The padding token.
        mask_token (`str`, optional, defaults to "<mask>"):
            The mask token.
        add_bos_token (`bool`, optional, defaults to True):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, optional, defaults to False):
            Whether or not to add an `eos_token` at the end of sequences.
        vocab (`str` or `dict[str, int]`, optional):
            Custom vocabulary dict. If not provided, a minimal vocabulary is created using the special tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[Union[str, dict[str, int]]] = None,
        merges: Optional[Union[str, list[str]]] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(eos_token): 1,
                str(bos_token): 2,
                str(unk_token): 3,
                str(mask_token): 4,
            }
        self._vocab = vocab
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                fuse_unk=True,
                unk_token=str(unk_token),
                dropout=None,
                byte_fallback=True,
            )
        )

        self._tokenizer.decoder = decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
        )
        self._tokenizer.normalizer = normalizers.Replace(" ", "▁")
        self._tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    def _unk_id(self) -> int:
        # Align with historical Gemma convention: pad, eos, bos, unk
        return 3


__all__ = ["GemmaTokenizer"]
