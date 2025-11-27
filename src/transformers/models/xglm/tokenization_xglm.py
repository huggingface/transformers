# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for XGLM."""

from typing import Optional

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class XGLMTokenizer(TokenizersBackend):
    """
    Construct a XGLM tokenizer (backed by HuggingFace's tokenizers library). Based on BPE.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        tokenizer_file (`str`, *optional*):
            Path to a tokenizers JSON file containing the serialization of a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, a minimal vocabulary is created.
        merges (`list[tuple[str, str]]`, *optional*):
            Custom merge rules for BPE. If not provided, merges are generated from the vocabulary.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether to add a prefix space before encoding.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        vocab: Optional[dict] = None,
        merges: Optional[list[tuple[str, str]]] = None,
        add_prefix_space: bool = True,
        **kwargs,
    ):
        self.num_madeup_words = 7
        madeup_words = [f"<madeupword{i}>" for i in range(self.num_madeup_words)]
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            word for word in madeup_words if word not in kwargs["additional_special_tokens"]
        ]

        self.add_prefix_space = add_prefix_space

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [
                (str(bos_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
            ]

        self._tokenizer = Tokenizer(Unigram(vocab=self._vocab, unk_id=3, byte_fallback=False))

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"[\n\r\t]"), " "),
                normalizers.NFKC(),
                normalizers.Replace(Regex(r" {2,}"), " "),
            ]
        )
        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.eos_token} $A {self.eos_token}",
            pair=f"{self.eos_token} $A {self.eos_token} {self.eos_token} $B {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


__all__ = ["XGLMTokenizer"]
