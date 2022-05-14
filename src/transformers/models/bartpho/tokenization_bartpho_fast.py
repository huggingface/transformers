# coding=utf-8
# Copyright 2021 VinAI Research and the HuggingFace Inc. team.
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
""" Tokenization classes for BARTpho-syllable model."""


from ...tokenization_utils import AddedToken
from ...utils import is_sentencepiece_available, logging
from ..xlm_roberta import XLMRobertaTokenizerFast


if is_sentencepiece_available():
    from .tokenization_bartpho import BartphoTokenizer
else:
    BartphoTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"vinai/bartpho-syllable": 1024}

# fmt: off
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
# fmt: on


class BartphoTokenizerFast(XLMRobertaTokenizerFast):
    """
    Construct a "fast" BARTpho tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`XLMRobertaTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
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
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = BartphoTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=None,
        **kwargs
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()

        if additional_special_tokens is not None:
            # Only add those special tokens if they are not already there.
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        self.add_special_tokens({"additional_special_tokens": _additional_special_tokens})
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }
