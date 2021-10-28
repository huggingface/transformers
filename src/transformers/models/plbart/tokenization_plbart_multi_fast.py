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

import os
from contextlib import contextmanager
from shutil import copyfile
from typing import List, Optional, Tuple

from tokenizers import processors

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_plbart import PLBartMultiTokenizer
else:
    PLBartMultiTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "uclanlp/plbart-multi_task-all": "https://huggingface.co/uclanlp/plbart-multi_task-all/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-compiled": "https://huggingface.co/uclanlp/plbart-multi_task-compiled/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-dynamic": "https://huggingface.co/uclanlp/plbart-multi_task-dynamic/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-go": "https://huggingface.co/uclanlp/plbart-multi_task-go/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-interpreted": "https://huggingface.co/uclanlp/plbart-multi_task-interpreted/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-java": "https://huggingface.co/uclanlp/plbart-multi_task-java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-js": "https://huggingface.co/uclanlp/plbart-multi_task-js/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-php": "https://huggingface.co/uclanlp/plbart-multi_task-php/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-python": "https://huggingface.co/uclanlp/plbart-multi_task-python/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-ruby": "https://huggingface.co/uclanlp/plbart-multi_task-ruby/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-static": "https://huggingface.co/uclanlp/plbart-multi_task-static/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-strong": "https://huggingface.co/uclanlp/plbart-multi_task-strong/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-weak": "https://huggingface.co/uclanlp/plbart-multi_task-weak/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-all-generation": "https://huggingface.co/uclanlp/plbart-single_task-all-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-all-summarization": "https://huggingface.co/uclanlp/plbart-single_task-all-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-compiled-generation": "https://huggingface.co/uclanlp/plbart-single_task-compiled-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-compiled-summarization": "https://huggingface.co/uclanlp/plbart-single_task-compiled-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-dynamic-generation": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-dynamic-summarization": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_go": "https://huggingface.co/uclanlp/plbart-single_task-en_go/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_java": "https://huggingface.co/uclanlp/plbart-single_task-en_java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_js": "https://huggingface.co/uclanlp/plbart-single_task-en_js/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_php": "https://huggingface.co/uclanlp/plbart-single_task-en_php/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_python": "https://huggingface.co/uclanlp/plbart-single_task-en_python/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_ruby": "https://huggingface.co/uclanlp/plbart-single_task-en_ruby/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-go_en": "https://huggingface.co/uclanlp/plbart-single_task-go_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-interpreted-generation": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-interpreted-summarization": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-java_en": "https://huggingface.co/uclanlp/plbart-single_task-java_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-js_en": "https://huggingface.co/uclanlp/plbart-single_task-js_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-php_en": "https://huggingface.co/uclanlp/plbart-single_task-php_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-python_en": "https://huggingface.co/uclanlp/plbart-single_task-python_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-ruby_en": "https://huggingface.co/uclanlp/plbart-single_task-ruby_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-static-generation": "https://huggingface.co/uclanlp/plbart-single_task-static-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-static-summarization": "https://huggingface.co/uclanlp/plbart-single_task-static-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-strong-generation": "https://huggingface.co/uclanlp/plbart-single_task-strong-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-strong-summarization": "https://huggingface.co/uclanlp/plbart-single_task-strong-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-weak-generation": "https://huggingface.co/uclanlp/plbart-single_task-weak-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-weak-summarization": "https://huggingface.co/uclanlp/plbart-single_task-weak-summarization/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "uclanlp/plbart-multi_task-all": "https://huggingface.co/uclanlp/plbart-multi_task-all/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-compiled": "https://huggingface.co/uclanlp/plbart-multi_task-compiled/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-dynamic": "https://huggingface.co/uclanlp/plbart-multi_task-dynamic/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-go": "https://huggingface.co/uclanlp/plbart-multi_task-go/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-interpreted": "https://huggingface.co/uclanlp/plbart-multi_task-interpreted/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-java": "https://huggingface.co/uclanlp/plbart-multi_task-java/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-js": "https://huggingface.co/uclanlp/plbart-multi_task-js/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-php": "https://huggingface.co/uclanlp/plbart-multi_task-php/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-python": "https://huggingface.co/uclanlp/plbart-multi_task-python/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-ruby": "https://huggingface.co/uclanlp/plbart-multi_task-ruby/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-static": "https://huggingface.co/uclanlp/plbart-multi_task-static/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-strong": "https://huggingface.co/uclanlp/plbart-multi_task-strong/resolve/main/tokenizer.json",
        "uclanlp/plbart-multi_task-weak": "https://huggingface.co/uclanlp/plbart-multi_task-weak/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-all-generation": "https://huggingface.co/uclanlp/plbart-single_task-all-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-all-summarization": "https://huggingface.co/uclanlp/plbart-single_task-all-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-compiled-generation": "https://huggingface.co/uclanlp/plbart-single_task-compiled-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-compiled-summarization": "https://huggingface.co/uclanlp/plbart-single_task-compiled-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-dynamic-generation": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-dynamic-summarization": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_go": "https://huggingface.co/uclanlp/plbart-single_task-en_go/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_java": "https://huggingface.co/uclanlp/plbart-single_task-en_java/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_js": "https://huggingface.co/uclanlp/plbart-single_task-en_js/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_php": "https://huggingface.co/uclanlp/plbart-single_task-en_php/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_python": "https://huggingface.co/uclanlp/plbart-single_task-en_python/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-en_ruby": "https://huggingface.co/uclanlp/plbart-single_task-en_ruby/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-go_en": "https://huggingface.co/uclanlp/plbart-single_task-go_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-interpreted-generation": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-interpreted-summarization": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-java_en": "https://huggingface.co/uclanlp/plbart-single_task-java_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-js_en": "https://huggingface.co/uclanlp/plbart-single_task-js_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-php_en": "https://huggingface.co/uclanlp/plbart-single_task-php_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-python_en": "https://huggingface.co/uclanlp/plbart-single_task-python_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-ruby_en": "https://huggingface.co/uclanlp/plbart-single_task-ruby_en/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-static-generation": "https://huggingface.co/uclanlp/plbart-single_task-static-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-static-summarization": "https://huggingface.co/uclanlp/plbart-single_task-static-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-strong-generation": "https://huggingface.co/uclanlp/plbart-single_task-strong-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-strong-summarization": "https://huggingface.co/uclanlp/plbart-single_task-strong-summarization/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-weak-generation": "https://huggingface.co/uclanlp/plbart-single_task-weak-generation/resolve/main/tokenizer.json",
        "uclanlp/plbart-single_task-weak-summarization": "https://huggingface.co/uclanlp/plbart-single_task-weak-summarization/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "uclanlp/plbart-multi_task-all": 1024,
    "uclanlp/plbart-multi_task-compiled": 1024,
    "uclanlp/plbart-multi_task-dynamic": 1024,
    "uclanlp/plbart-multi_task-go": 1024,
    "uclanlp/plbart-multi_task-interpreted": 1024,
    "uclanlp/plbart-multi_task-java": 1024,
    "uclanlp/plbart-multi_task-js": 1024,
    "uclanlp/plbart-multi_task-php": 1024,
    "uclanlp/plbart-multi_task-python": 1024,
    "uclanlp/plbart-multi_task-ruby": 1024,
    "uclanlp/plbart-multi_task-static": 1024,
    "uclanlp/plbart-multi_task-strong": 1024,
    "uclanlp/plbart-multi_task-weak": 1024,
    "uclanlp/plbart-single_task-all-generation": 1024,
    "uclanlp/plbart-single_task-all-summarization": 1024,
    "uclanlp/plbart-single_task-compiled-generation": 1024,
    "uclanlp/plbart-single_task-compiled-summarization": 1024,
    "uclanlp/plbart-single_task-dynamic-generation": 1024,
    "uclanlp/plbart-single_task-dynamic-summarization": 1024,
    "uclanlp/plbart-single_task-en_go": 1024,
    "uclanlp/plbart-single_task-en_java": 1024,
    "uclanlp/plbart-single_task-en_js": 1024,
    "uclanlp/plbart-single_task-en_php": 1024,
    "uclanlp/plbart-single_task-en_python": 1024,
    "uclanlp/plbart-single_task-en_ruby": 1024,
    "uclanlp/plbart-single_task-go_en": 1024,
    "uclanlp/plbart-single_task-interpreted-generation": 1024,
    "uclanlp/plbart-single_task-interpreted-summarization": 1024,
    "uclanlp/plbart-single_task-java_en": 1024,
    "uclanlp/plbart-single_task-js_en": 1024,
    "uclanlp/plbart-single_task-php_en": 1024,
    "uclanlp/plbart-single_task-python_en": 1024,
    "uclanlp/plbart-single_task-ruby_en": 1024,
    "uclanlp/plbart-single_task-static-generation": 1024,
    "uclanlp/plbart-single_task-static-summarization": 1024,
    "uclanlp/plbart-single_task-strong-generation": 1024,
    "uclanlp/plbart-single_task-strong-summarization": 1024,
    "uclanlp/plbart-single_task-weak-generation": 1024,
    "uclanlp/plbart-single_task-weak-summarization": 1024,
}

FAIRSEQ_LANGUAGE_CODES = [
    "java",
    "python",
    "en_XX",
    "js",
    "php",
    "ruby",
    "go",
]


class PLBartMultiTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" PLBART tokenizer for PLBART-Multilingual (backed by HuggingFace's `tokenizers` library). Based
    on `BPE <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        src_lang (:obj:`str`, `optional`):
            A string representing the source language.
        tgt_lang (:obj:`str`, `optional`):
            A string representing the target language.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.

    Examples::

        >>> from transformers import PLBartMultiTokenizerFast
        >>> tokenizer = PLBartMultiTokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
        >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
        >>> tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> model_inputs = tokenizer(src_text, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
        >>> # model(**model_inputs, labels=labels) should work
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = PLBartMultiTokenizer

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        src_lang=None,
        tgt_lang=None,
        tokenizer_file=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", [])
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        super().__init__(
            vocab_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An PLBART sequence has the following format, where ``X`` represents the sequence:

        - ``input_ids`` (for encoder) ``[src_lang_code] X [eos]``
        - ``labels``: (for decoder) ``[tgt_lang_code] X [eos]``

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

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
        """Reset the special tokens to the target language setting. prefix=[src_lang_code] and suffix=[eos]."""
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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
