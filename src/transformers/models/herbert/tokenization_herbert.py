# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Allegro.pl, Facebook Inc. and the HuggingFace Inc. team.
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

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}


class HerbertTokenizer(TokenizersBackend):
    """
    Construct a BPE tokenizer for HerBERT (backed by HuggingFace's tokenizers library).

    Peculiarities:

    - uses BERT's pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of
      a punctuation character will be treated separately.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The mask token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary.
        merges (`list`, *optional*):
            Custom merges list.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab: Optional[dict] = None,
        merges: Optional[list] = None,
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        sep_token: str = "</s>",
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        **kwargs,
    ):
        if vocab is not None:
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
            )
        else:
            self._vocab = {}

        if merges is not None:
            # Convert lists to tuples if necessary (happens when loading from JSON)
            self._merges = [tuple(merge) if isinstance(merge, list) else merge for merge in merges]
        else:
            self._merges = []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
            )
        )

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            lowercase=False, strip_accents=False, clean_text=True, handle_chinese_chars=True
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        tokenizer_object = self._tokenizer

        self.vocab_file = vocab_file
        self.merges_file = merges_file

        super().__init__(
            tokenizer_object=tokenizer_object,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.BertProcessing(
            sep=(self.sep_token, 2),
            cls=(self.cls_token, 0),
        )


__all__ = ["HerbertTokenizer"]
