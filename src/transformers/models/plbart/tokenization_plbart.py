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

from typing import Any

from ...tokenization_python import BatchEncoding
from ...tokenization_utils_base import AddedToken
from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

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


@requires(backends=("sentencepiece",))
class PLBartTokenizer(SentencePieceBackend):
    """
    Construct an PLBART tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Args:
        vocab_file (`str`):
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
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:
            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

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

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        vocab_file,
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
        sp_model_kwargs: dict[str, Any] | None = None,
        additional_special_tokens=None,
        clean_up_tokenization_spaces=True,
        **kwargs,
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        src_lang = self._convert_lang_code_special_format(src_lang)
        tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        self.language_codes = language_codes
        fairseq_language_codes = FAIRSEQ_LANGUAGE_CODES[self.language_codes]

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.vocab_file = vocab_file
        self.lang_code_to_id = {}
        self.id_to_lang_code = {}
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        self.fairseq_offset = 1
        _additional_special_tokens = list(fairseq_language_codes)

        if additional_special_tokens is not None:
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            language_codes=language_codes,
            special_tokens_pattern="prefix_suffix",
            token_type_ids_pattern="all_zeros",
            **kwargs,
        )

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(fairseq_language_codes)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        if self.language_codes == "base":
            self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        reserved_tokens = {"<s>", "<pad>", "</s>", "<unk>", "<mask>"}
        reserved_tokens.update(FAIRSEQ_LANGUAGE_CODES[self.language_codes])

        removed = False
        for token in reserved_tokens:
            idx = self._added_tokens_encoder.pop(token, None)
            if idx is not None:
                self._added_tokens_decoder.pop(idx, None)
                removed = True
        if removed:
            self._update_trie()
            self._update_total_vocab_size()

        synced = False
        for token, idx in self._added_tokens_encoder.items():
            if idx in self._added_tokens_decoder:
                continue
            self._added_tokens_decoder[idx] = AddedToken(
                token, special=True, normalized=False, lstrip=False, rstrip=False
            )
            synced = True
        if synced:
            self._update_trie()
            self._update_total_vocab_size()

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
    def vocab_size(self):
        lang_code_count = len(getattr(self, "lang_code_to_id", {}))
        fairseq_offset = getattr(self, "fairseq_offset", 1)
        base_vocab = len(self.sp_model) if hasattr(self, "sp_model") else 0
        if getattr(self, "language_codes", "base") == "base":
            return base_vocab + lang_code_count + fairseq_offset + 1  # +1 for mask token
        return base_vocab + lang_code_count + fairseq_offset

    def get_vocab(self):
        """Override to use fairseq vocabulary structure"""
        vocab = self.fairseq_tokens_to_ids.copy()
        for i in range(self.sp_model.get_piece_size()):
            sp_token = self.sp_model.IdToPiece(i)
            # Map SP token to fairseq ID: SP ID 0 maps to unk_token_id, others map to SP_ID + fairseq_offset
            vocab_id = self.unk_token_id if i == 0 else (i + self.fairseq_offset)
            if sp_token not in vocab:
                vocab[sp_token] = vocab_id
        # Add any additional tokens
        vocab.update({token: idx for token, idx in self._added_tokens_encoder.items() if token not in vocab})
        return vocab

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        new_src_lang = self._convert_lang_code_special_format(new_src_lang)
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: str | None, tgt_lang: str | None, **extra_kwargs
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

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "en_XX",
        tgt_texts: list[str] | None = None,
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

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        lang = self._convert_lang_code_special_format(lang)

        self.cur_lang_code = self.lang_code_to_id[lang] if lang is not None else None
        self.prefix_tokens = []
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.suffix_tokens = [self.eos_token_id]

    def _convert_lang_code_special_format(self, lang: str) -> str:
        """Convert Language Codes to format tokenizer uses if required"""
        lang = FAIRSEQ_LANGUAGE_CODES_MAP.get(lang, lang)
        return lang

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
        """Override to use self.clean_up_tokenization_spaces as default for batched input."""
        return super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["PLBartTokenizer"]
