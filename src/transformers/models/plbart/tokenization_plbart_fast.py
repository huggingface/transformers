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

from contextlib import contextmanager
from typing import List, Optional

from tokenizers import processors

from ...file_utils import is_sentencepiece_available
from ...tokenization_utils import BatchEncoding
from ...utils import logging
from ..xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast


if is_sentencepiece_available():
    from .tokenization_plbart import PLBartTokenizer
else:
    PLBartTokenizer = None


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-c-cpp-defect-detection": "https://huggingface.co/uclanlp/plbart-c-cpp-defect-detection/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-cs-java": "https://huggingface.co/uclanlp/plbart-cs-java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-en_XX-java": "https://huggingface.co/uclanlp/plbart-en_XX-java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-go-en_XX": "https://huggingface.co/uclanlp/plbart-go-en_XX/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-java-clone-detection": "https://huggingface.co/uclanlp/plbart-java-clone-detection/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-java-cs": "https://huggingface.co/uclanlp/plbart-java-cs/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-java-en_XX": "https://huggingface.co/uclanlp/plbart-java-en_XX/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-javascript-en_XX": "https://huggingface.co/uclanlp/plbart-javascript-en_XX/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-php-en_XX": "https://huggingface.co/uclanlp/plbart-php-en_XX/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-python-en_XX": "https://huggingface.co/uclanlp/plbart-python-en_XX/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-refine-java-medium": "https://huggingface.co/uclanlp/plbart-refine-java-medium/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-refine-java-small": "https://huggingface.co/uclanlp/plbart-refine-java-small/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-ruby-en_XX": "https://huggingface.co/uclanlp/plbart-ruby-en_XX/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/tokenizer.json",
        "uclanlp/plbart-c-cpp-defect-detection": "https://huggingface.co/uclanlp/plbart-c-cpp-defect-detection/resolve/main/tokenizer.json",
        "uclanlp/plbart-cs-java": "https://huggingface.co/uclanlp/plbart-cs-java/resolve/main/tokenizer.json",
        "uclanlp/plbart-en_XX-java": "https://huggingface.co/uclanlp/plbart-en_XX-java/resolve/main/tokenizer.json",
        "uclanlp/plbart-go-en_XX": "https://huggingface.co/uclanlp/plbart-go-en_XX/resolve/main/tokenizer.json",
        "uclanlp/plbart-java-clone-detection": "https://huggingface.co/uclanlp/plbart-java-clone-detection/resolve/main/tokenizer.json",
        "uclanlp/plbart-java-cs": "https://huggingface.co/uclanlp/plbart-java-cs/resolve/main/tokenizer.json",
        "uclanlp/plbart-java-en_XX": "https://huggingface.co/uclanlp/plbart-java-en_XX/resolve/main/tokenizer.json",
        "uclanlp/plbart-javascript-en_XX": "https://huggingface.co/uclanlp/plbart-javascript-en_XX/resolve/main/tokenizer.json",
        "uclanlp/plbart-php-en_XX": "https://huggingface.co/uclanlp/plbart-php-en_XX/resolve/main/tokenizer.json",
        "uclanlp/plbart-python-en_XX": "https://huggingface.co/uclanlp/plbart-python-en_XX/resolve/main/tokenizer.json",
        "uclanlp/plbart-refine-java-medium": "https://huggingface.co/uclanlp/plbart-refine-java-medium/resolve/main/tokenizer.json",
        "uclanlp/plbart-refine-java-small": "https://huggingface.co/uclanlp/plbart-refine-java-small/resolve/main/tokenizer.json",
        "uclanlp/plbart-ruby-en_XX": "https://huggingface.co/uclanlp/plbart-ruby-en_XX/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "uclanlp/plbart-base": 1024,
    "uclanlp/plbart-c-cpp-defect-detection": 1024,
    "uclanlp/plbart-cs-java": 1024,
    "uclanlp/plbart-en_XX-java": 1024,
    "uclanlp/plbart-go-en_XX": 1024,
    "uclanlp/plbart-java-clone-detection": 1024,
    "uclanlp/plbart-java-cs": 1024,
    "uclanlp/plbart-java-en_XX": 1024,
    "uclanlp/plbart-javascript-en_XX": 1024,
    "uclanlp/plbart-php-en_XX": 1024,
    "uclanlp/plbart-python-en_XX": 1024,
    "uclanlp/plbart-refine-java-medium": 1024,
    "uclanlp/plbart-refine-java-small": 1024,
    "uclanlp/plbart-ruby-en_XX": 1024,
}

FAIRSEQ_LANGUAGE_CODES = [
    "java",
    "python",
    "en_XX",
]


class PLBartTokenizerFast(XLMRobertaTokenizerFast):
    """
    Construct a "fast" PLBART tokenizer (backed by HuggingFace's *tokenizers* library). Based on [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    [`PLBartTokenizerFast`] is a subclass of [`XLMRobertaTokenizerFast`]. Refer
    to superclass [`XLMRobertaTokenizerFast`] for usage examples and documentation concerning the
    initialization parameters and other methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and ``<language code>
    <tokens> <eos>``` for target language documents.

    Examples:

    ```python
    >>> from transformers import PLBartTokenizerFast >>> tokenizer =
    PLBartTokenizerFast.from_pretrained('uclanlp/plbart-python-en_XX', src_lang="python", tgt_lang="en_XX") >>>
    example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])" >>>
    expected_translation_english = "Returns the maximum value of a b c." >>> inputs =
    tokenizer(example_python_phrase, return_tensors="pt) >>> with tokenizer.as_target_tokenizer(): ... labels =
    tokenizer(expected_translation_english, return_tensors="pt") >>> inputs["labels"] = labels["input_ids"]
    ```
"""

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    slow_tokenizer_class = PLBartTokenizer

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        additional_special_tokens=None,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

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

        self._src_lang = src_lang
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang) if self._src_lang is not None else None
        self.tgt_lang = tgt_lang
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

        An PLBART sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

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

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "python",
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

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang) if src_lang is not None else None
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
        self.cur_lang_code = self.convert_tokens_to_ids(lang) if lang is not None else None
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
