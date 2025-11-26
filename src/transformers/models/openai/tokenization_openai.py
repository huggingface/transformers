# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...convert_slow_tokenizer import generate_merges
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class OpenAIGPTTokenizer(TokenizersBackend):
    """
    Construct a GPT Tokenizer (backed by HuggingFace's *tokenizers* library). Based on Byte-Pair-Encoding with
    the following peculiarities:

    - lower case all inputs
    - uses BERT's BasicTokenizer for pre-BPE tokenization

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to a tokenizers JSON file containing the serialization of a tokenizer.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, a blank vocabulary is initialized.
        merges (`list`, *optional*):
            Custom merges list. If not provided, an empty list is used.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        vocab=None,
        merges=None,
        vocab_file=None,
        merges_file=None,
        **kwargs,
    ):
        # Initialize vocabulary
        if vocab is not None:
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
            )
        else:
            # Initialize minimal vocabulary with unk token
            self._vocab = {str(unk_token): 0}

        # Initialize merges
        if merges is not None:
            self._merges = merges if merges is not None else generate_merges(self._vocab)
        else:
            self._merges = []

        # Create BPE tokenizer
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        # Set normalizer and pre-tokenizer to mimic OpenAI GPT behavior
        # OpenAI GPT uses BERT BasicTokenizer with lower_case=True
        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFD(),
                normalizers.Lowercase(),
                normalizers.StripAccents(),
            ]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.merges_file = merges_file

    def _post_init(self):
        """Post-initialization to ensure tokenizer settings are applied correctly."""
        # Re-apply settings to ensure they're correct after loading from pretrained
        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFD(),
                normalizers.Lowercase(),
                normalizers.StripAccents(),
            ]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        # Call parent to handle AddedToken properties
        super()._post_init()

    @property
    def do_lower_case(self):
        return True


__all__ = ["OpenAIGPTTokenizer"]
