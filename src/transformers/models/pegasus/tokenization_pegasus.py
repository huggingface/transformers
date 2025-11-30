# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
"""Tokenization class for model PEGASUS."""

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class PegasusTokenizer(TokenizersBackend):
    r"""
    Construct a PEGASUS tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (`str`, *optional*, defaults to `"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to *[MASK2]* in [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization](https://huggingface.co/papers/1912.08777).
        mask_token_sent (`str`, *optional*, defaults to `"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to *[MASK1]* in [PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization](https://huggingface.co/papers/1912.08777).
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the [original PEGASUS
            tokenizer](https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66)
            that uses the tokens 2 - 104 only for pretraining
        offset (`int`, *optional*, defaults to 103):
            Offset for additional special tokens.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, a blank vocabulary is initialized.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,
        vocab=None,
        vocab_file=None,
        **kwargs,
    ):
        self.offset = offset
        self.vocab_file = vocab_file

        if additional_special_tokens is None:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]

        if vocab is not None:
            # For Pegasus, insert special tokens at the beginning
            special_tokens_set = {pad_token, eos_token, mask_token_sent, mask_token, unk_token}
            special_tokens_set.update(additional_special_tokens)

            # Build special tokens in correct order
            _vocab_list = [
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
            ]
            if mask_token_sent:
                _vocab_list.append((str(mask_token_sent), 0.0))
            for token in additional_special_tokens:
                if token not in [pad_token, eos_token, mask_token_sent]:
                    _vocab_list.append((str(token), 0.0))
            if mask_token not in [t for t, _ in _vocab_list]:
                _vocab_list.append((str(mask_token), 0.0))
            _vocab_list.append((str(unk_token), 0.0))

            # Filter out special tokens from main vocab and combine
            filtered_vocab = [(t, s) for t, s in vocab if t not in special_tokens_set]
            _vocab_list = _vocab_list + filtered_vocab
        else:
            _vocab_list = [(str(unk_token), 0.0)]

        self._vocab = {token: idx for idx, (token, _) in enumerate(_vocab_list)}

        self._tokenizer = Tokenizer(Unigram(vocab=_vocab_list, unk_id=self._vocab.get(str(unk_token), 0)))

        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Replace(Regex(r"\n"), " "), normalizers.Replace(Regex(r" {2,}"), " ")]
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A {eos_token}",
            pair=f"$A $B {eos_token}",
            special_tokens=[(str(eos_token), self._vocab.get(str(eos_token), 1))],
        )

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)


__all__ = ["PegasusTokenizer"]
