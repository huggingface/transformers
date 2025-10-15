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
from typing import Optional

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class GemmaTokenizer(TokenizersBackend):
    """
    Construct a fast Gemma tokenizer (backed by HuggingFace's tokenizers library).

    This tokenizer uses a Unigram model with ByteFallback, no prefix space, and a normalizer that replaces
    spaces with "▁". It supports creating a minimal, trainable tokenizer from scratch via `from_scratch=True`.

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
        add_bos_token (`bool`, optional, defaults to True):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, optional, defaults to False):
            Whether or not to add an `eos_token` at the end of sequences.
        from_scratch (`bool`, optional, defaults to False):
            When True, creates a minimal trainable tokenizer with only special tokens.
        vocab_scores (`list[tuple[str, float]]`, optional):
            Custom initial Unigram vocabulary with scores. If unset and `from_scratch=True`, a minimal
            vocabulary is created using the provided special tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = None
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        from_scratch: bool = False,
        vocab_scores: Optional[list[tuple[str, float]]] = None,
        vocab: Optional[dict] = None,
        merges: Optional[list[tuple[str, str]]] = None,
        **kwargs,
    ):
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token

        if vocab_scores is None and (vocab is not None or merges is not None):
            raise ValueError("GemmaTokenizer requires 'vocab_scores' when 'vocab' or 'merges' are provided.")

        # Convert to strings in case they're AddedToken objects
        self._vocab_scores = (
            vocab_scores
            if vocab_scores is not None
            else self._default_vocab_scores(str(pad_token), str(eos_token), str(bos_token), str(unk_token))
        )

        self._tokenizer = Tokenizer(Unigram(self._vocab_scores, unk_id=self._unk_id(), byte_fallback=True))
        self._tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
        self._tokenizer.normalizer = normalizers.Replace(" ", "▁")
        self._tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_file=None,
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self._post_init()

    def _post_init(self):
        """Post-initialization setup that needs to run after _tokenizer is set."""
        self.add_tokens(list(self.all_special_tokens), special_tokens=True)
        self.update_post_processor()

    def _unk_id(self) -> int:
        # Align with historical Gemma convention: pad, eos, bos, unk
        return 3

    def _default_vocab_scores(
        self, pad_token: str, eos_token: str, bos_token: str, unk_token: str
    ) -> list[tuple[str, float]]:
        return [
            (pad_token, 0.0),
            (eos_token, 0.0),
            (bos_token, 0.0),
            (unk_token, 0.0),
        ]


__all__ = ["GemmaTokenizer"]
