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
"""Tokenization classes for Camembert model."""

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


SPIECE_UNDERLINE = "▁"


class CamembertTokenizer(TokenizersBackend):
    """
    Construct a "fast" CamemBERT tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=None,
        add_prefix_space=True,
        vocab_file=None,
        vocab=None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.add_prefix_space = add_prefix_space

        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        if additional_special_tokens is None:
            additional_special_tokens = ["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"]

        if vocab is not None and isinstance(vocab, list):
            self._vocab = list(vocab)
            unk_index = next(i for i, (tok, _) in enumerate(self._vocab) if tok == str(unk_token))
            self._tokenizer = Tokenizer(Unigram(self._vocab, unk_id=unk_index, byte_fallback=False))
        else:
            self._vocab = [
                ("<s>NOTUSED", 0.0),
                (str(pad_token), 0.0),
                ("</s>NOTUSED", 0.0),
                (str(unk_token), 0.0),
                ("<unk>NOTUSED", -100),
                (str(mask_token), 0.0),
            ]
            self._tokenizer = Tokenizer(Unigram(self._vocab, unk_id=3, byte_fallback=False))

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace("\n", " "),
                normalizers.Replace("\r", " "),
                normalizers.Replace("\t", " "),
                normalizers.Strip(left=False, right=True),
                normalizers.Replace(Regex(" {2,}"), "▁"),
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
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # always adds BOS/EOS with "</s> </s>" separator for pairs
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} {self.eos_token} $B {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


__all__ = ["CamembertTokenizer"]
