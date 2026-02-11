# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License
"""Tokenization classes for XLM-RoBERTa model (Tokenizers backend)."""

from typing import Optional, Union

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


class XLMRobertaTokenizer(TokenizersBackend):
    r"""
    Construct an XLM-RoBERTa tokenizer (backed by HuggingFace's tokenizers library). Based on SentencePiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, optional): Path to the vocabulary file.
        merges_file (`str`, optional): Path to the merges file.
        tokenizer_file (`str`, optional): Path to a tokenizers JSON file containing the serialization of a tokenizer.
        bos_token (`str`, optional, defaults to `"<s>"`): The beginning of sequence token.
        eos_token (`str`, optional, defaults to `"</s>"`): The end of sequence token.
        sep_token (`str`, optional, defaults to `"</s>"`): The separator token.
        cls_token (`str`, optional, defaults to `"<s>"`): The classifier token.
        unk_token (`str`, optional, defaults to `"<unk>"`): The unknown token.
        pad_token (`str`, optional, defaults to `"<pad>"`): The padding token.
        mask_token (`str`, optional, defaults to `"<mask>"`): The mask token.
        add_prefix_space (`bool`, optional, defaults to `True`): Whether to add an initial space.
        vocab (`str`, `dict` or `list`, optional): Custom vocabulary dictionary.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: Optional[Union[str, list[tuple[str, float]]]] = None,
        add_prefix_space: bool = True,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [
                (str(bos_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                (str(mask_token), 0.0),
            ]

        self._tokenizer = Tokenizer(Unigram(vocab=self._vocab, unk_id=3, byte_fallback=False))

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Strip(left=False, right=True),
                normalizers.Replace(" {2,}", "▁"),
            ]
        )

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme),
            ]
        )
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=[str(bos_token), "$A", str(eos_token)],
            pair=[str(bos_token), "$A", str(eos_token), "$B", str(eos_token)],
            special_tokens=[
                (str(bos_token), self.bos_token_id),
                (str(eos_token), self.eos_token_id),
            ],
        )


__all__ = ["XLMRobertaTokenizer"]
