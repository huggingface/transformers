# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""
from ...utils import logging
from ..xlnet.tokenization_xlnet_fast import XLNetTokenizerFast


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/tokenizer.json",
    },
}


class CpmTokenizerFast(XLNetTokenizerFast):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    def __init__(self, *args, **kwargs):
        """
        Construct a CPM tokenizer. Based on [Jieba](https://pypi.org/project/jieba/) and
        [SentencePiece](https://github.com/google/sentencepiece).

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should
        refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether to lowercase the input when tokenizing.
            remove_space (`bool`, *optional*, defaults to `True`):
                Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (`bool`, *optional*, defaults to `False`):
                Whether to keep accents when tokenizing.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
                token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.

                </Tip>

            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the `sep_token`.

                </Tip>

            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            sep_token (`str`, *optional*, defaults to `"<sep>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
                for sequence classification or for a text and a question for question answering. It is also used as the
                last token of a sequence built with special tokens.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (`str`, *optional*, defaults to `"<cls>"`):
                The classifier token which is used when doing sequence classification (classification of the whole
                sequence instead of per-token classification). It is the first token of the sequence when built with
                special tokens.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
                Additional special tokens used by the tokenizer.

        Attributes:
            sp_model (`SentencePieceProcessor`):
                The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
        """
        super().__init__(*args, **kwargs)
        try:
            import jieba
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install jieba to use CpmTokenizer or CpmTokenizerFast. "
                "See https://pypi.org/project/jieba/ for installation."
            )
        self.jieba = jieba
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        batch_text_or_text_pairs = [
            " ".join([x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)])
            for text in batch_text_or_text_pairs
        ]
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        return text
