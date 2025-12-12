# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Tokenization classes for Big Bird model."""

from typing import Optional, Union

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


SPIECE_UNDERLINE = "▁"


class BigBirdTokenizer(TokenizersBackend):
    """
    Construct a "fast" BigBird tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods

    Args:
        vocab (`str`, `dict` or `list`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            Path to a tokenizers JSON file containing the serialization of a tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: list[int] = []
    model = Unigram

    def __init__(
        self,
        vocab: Optional[Union[str, dict, list]] = None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        add_prefix_space=True,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.add_prefix_space = add_prefix_space

        # Convert vocab to list of (token, score) tuples
        if vocab is None:
            vocab = [(str(pad_token), 0.0), (str(eos_token), 0.0), (str(bos_token), 0.0), (str(unk_token), 0.0)]
            unk_id = 3
        elif isinstance(vocab, list):
            # vocab.insert(100, (str(unk_token), 0.0))  # Ensure unk_token is in vocab at index 100
            unk_id = vocab.index((str(unk_token), 0.0)) if (str(unk_token), 0.0) in vocab else 100

        self._tokenizer = Tokenizer(Unigram(vocab, unk_id=unk_id, byte_fallback=False))
        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Strip(left=False, right=False), normalizers.Replace(Regex(r" {2,}"), SPIECE_UNDERLINE)]
        )

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=prepend_scheme, split=True
        )
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme, split=True)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # Ensure cls_token and sep_token are in vocab
        cls_token_str = str(cls_token)
        sep_token_str = str(sep_token)
        cls_token_id = self.cls_token_id
        sep_token_id = self.sep_token_id

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token_str}:0 $A:0 {sep_token_str}:0",
            pair=f"{cls_token_str}:0 $A:0 {sep_token_str}:0 $B:1 {sep_token_str}:1",
            special_tokens=[(cls_token_str, cls_token_id), (sep_token_str, sep_token_id)],
        )


__all__ = ["BigBirdTokenizer"]
