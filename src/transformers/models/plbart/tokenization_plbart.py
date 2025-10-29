# coding=utf-8
# Copyright 2022, UCLA NLP, The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from ...tokenization_python import AddedToken, BatchEncoding
from ...tokenization_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


FAIRSEQ_LANGUAGE_CODES = {
    "base": ["__java__", "__python__", "__en_XX__"],
    "multi": ["__java__", "__python__", "__en_XX__", "__javascript__", "__php__", "__ruby__", "__go__"],
}

FAIRSEQ_LANGUAGE_CODES_MAP = {
    "java": "__java__",
    "python": "__python__",
    "en_XX": "__en_XX__",
    "javascript": "__javascript__",
    "php": "__php__",
    "ruby": "__ruby__",
    "go": "__go__",
}


class PLBartTokenizer(TokenizersBackend):
    """
    Construct a PLBart tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The start of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The cls token, which is a special token used as the first token for all tasks.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token(`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masking tasks. This
            is only used in the `"base"` tokenizer type. For `"multi"` tokenizer, masking is never done for the
            downstream tasks.
        language_codes (`str`, *optional*, defaults to `"base"`):
            What language codes to use. Should be one of `"base"` or `"multi"`.
        vocab (`list`, *optional*):
            Vocabulary as list of (token, score) tuples from SentencePieceExtractor.

    Examples:

    ```python
    >>> from transformers import PLBartTokenizer

    >>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
    >>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
    >>> expected_translation_english = "Returns the maximum value of a b c."
    >>> inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        language_codes="base",
        src_lang=None,
        tgt_lang=None,
        additional_special_tokens=None,
        vocab=None,
        merges=None,  # Ignored for Unigram
        vocab_file=None,
        **kwargs,
    ):
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.vocab_file = vocab_file
        self.language_codes = language_codes

        src_lang = self._convert_lang_code_special_format(src_lang)
        tgt_lang = self._convert_lang_code_special_format(tgt_lang)

        # Get language codes based on type ("base" or "multi")
        fairseq_language_codes = FAIRSEQ_LANGUAGE_CODES[self.language_codes]

        _additional_special_tokens = list(fairseq_language_codes)
        if additional_special_tokens is not None:
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        # PLBart uses fairseq vocab alignment like MBart: <s>=0, <pad>=1, </s>=2, <unk>=3, then SPM pieces[3:], lang codes, optionally <mask>
        if vocab is not None:
            #TODO: Ita handle diff vocab types
            # Ensure vocab is list of (str, float) tuples
            vocab = [(str(item[0]), float(item[1])) for item in vocab]
            
            # Reorder to fairseq: <s>, <pad>, </s>, <unk>, ... (rest of vocab from SPM[3:])
            vocab_list = []
            vocab_list.append((str(bos_token), 0.0))
            vocab_list.append((str(pad_token), 0.0))
            vocab_list.append((str(eos_token), 0.0))
            vocab_list.append((str(unk_token), 0.0))
            
            # Add the rest of the SentencePiece vocab (skipping first 3: <unk>, <s>, </s>)
            vocab_list.extend(vocab[3:])
            
            for lang_code in fairseq_language_codes:
                vocab_list.append((str(lang_code), 0.0))
            
            if self.language_codes == "base":
                vocab_list.append((str(mask_token), 0.0))
            
            self._vocab_scores = vocab_list
        else:
            self._vocab_scores = [
                (str(bos_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                ("▁", -2.0),
            ]
            for lang_code in fairseq_language_codes:
                self._vocab_scores.append((str(lang_code), 0.0))
            if self.language_codes == "base":
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
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            language_codes=language_codes,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            **kwargs,
        )

        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in fairseq_language_codes
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_offset = 1

        self.fairseq_tokens_to_ids = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        if self.language_codes == "base":
            self.fairseq_tokens_to_ids["<mask>"] = self.convert_tokens_to_ids(str(mask_token))
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        if self.language_codes == "base":
            self._src_lang = src_lang
            self.cur_lang_code_id = (
                self.lang_code_to_id[self._src_lang] if self._src_lang is not None else self._src_lang
            )
        else:
            self._src_lang = src_lang if src_lang is not None else "__en_XX__"
            self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]

        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        new_src_lang = self._convert_lang_code_special_format(new_src_lang)
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)


    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(self.tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[list[str]] = None,
        tgt_lang: str = "python",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        src_lang = self._convert_lang_code_special_format(src_lang)
        self.cur_lang_code = self.lang_code_to_id[src_lang] if src_lang is not None else None
        self.prefix_tokens = []
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        lang = self._convert_lang_code_special_format(lang)

        self.cur_lang_code = self.lang_code_to_id[lang] if lang is not None else None
        self.prefix_tokens = []
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def _convert_lang_code_special_format(self, lang: str) -> str:
        """Convert Language Codes to format tokenizer uses if required"""
        lang = FAIRSEQ_LANGUAGE_CODES_MAP.get(lang, lang)
        return lang


__all__ = ["PLBartTokenizer"]
