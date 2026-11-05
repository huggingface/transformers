# Copyright 2020 The HuggingFace Inc. team.
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
"""Tokenization class for Funnel Transformer."""

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]


class FunnelTokenizer(TokenizersBackend):
    r"""
    Construct a Funnel Transformer tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        bos_token (`str`, `optional`, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, `optional`, defaults to `"</s>"`):
            The end of sentence token.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model = WordPiece
    cls_token_type_id: int = 2

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        do_lower_case: bool = True,
        unk_token: str = "<unk>",
        sep_token: str = "<sep>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        clean_text: bool = True,
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        wordpieces_prefix: str = "##",
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.clean_text = clean_text
        self.wordpieces_prefix = wordpieces_prefix

        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
                str(bos_token): 5,
                str(eos_token): 6,
            }
        )

        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs,
        )
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token}:2 $A:0 {sep_token}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls_token}:2 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (str(cls_token), self.cls_token_id),
                (str(sep_token), self.sep_token_id),
            ],
        )


__all__ = ["FunnelTokenizer"]
