# coding=utf-8
# Copyright 2021 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]  # fmt: skip


class MBart50Tokenizer(TokenizersBackend):
    """
    Construct a MBart50 tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
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

    Examples:

    ```python
    >>> from transformers import MBart50Tokenizer

    >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    >>> # model(**model_inputs) should work
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        src_lang=None,
        tgt_lang=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        vocab=None,
        merges=None,  # Ignored for Unigram
        vocab_file=None,
        **kwargs,
    ):
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.vocab_file = vocab_file

        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        # MBart50 uses fairseq vocab alignment: <s>=0, <pad>=1, </s>=2, <unk>=3, then SPM pieces[3:], lang codes, <mask>
        if vocab is not None:
            vocab = [(str(item[0]), float(item[1])) for item in vocab]

            # Reorder to fairseq: <s>, <pad>, </s>, <unk>, ... (rest of vocab from SPM[3:])
            vocab_list = []
            vocab_list.append((str(cls_token), 0.0))   # 0: <s>
            vocab_list.append((str(pad_token), 0.0))   # 1: <pad>
            vocab_list.append((str(eos_token), 0.0))   # 2: </s>
            vocab_list.append((str(unk_token), 0.0))   # 3: <unk>
            
            vocab_list.extend(vocab[3:])
            
            for lang_code in FAIRSEQ_LANGUAGE_CODES:
                vocab_list.append((str(lang_code), 0.0))
            
            vocab_list.append((str(mask_token), 0.0))
            
            self._vocab_scores = vocab_list
        else:
            self._vocab_scores = [
                (str(cls_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                ("▁", -2.0),
            ]
            for lang_code in FAIRSEQ_LANGUAGE_CODES:
                self._vocab_scores.append((lang_code, 0.0))
            self._vocab_scores.append((str(mask_token), 0.0))

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=3,
                byte_fallback=False,
            )
        )

        self._tokenizer.normalizer = None

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True),
        ])

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_offset = 1

        # Build fairseq token mappings for backward compatibility
        self.fairseq_tokens_to_ids = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_tokens_to_ids["<mask>"] = self.convert_tokens_to_ids(str(mask_token))
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.tgt_lang = tgt_lang
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)


    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[list[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if self.tgt_lang is None:
            self.tgt_lang = self._src_lang
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.convert_tokens_to_ids(src_lang)
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.convert_tokens_to_ids(tgt_lang)
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs


__all__ = ["MBart50Tokenizer"]

# Backward alias
MBart50TokenizerFast = MBart50Tokenizer
