# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for SeamlessM4T."""

from typing import Optional, Union

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...tokenization_tokenizers import TokenizersBackend
from ...utils import PaddingStrategy, logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


class SeamlessM4TTokenizer(TokenizersBackend):
    """
    Construct a SeamlessM4T tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language
    code> <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import SeamlessM4TTokenizer

    >>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab (`list`, *optional*):
            List of (token, score) tuples for Unigram model. If not provided, uses default vocab.
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
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        vocab: Optional[list] = None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        src_lang="eng",
        tgt_lang="fra",
        additional_special_tokens=None,
        keep_accents=None,
        vocab_file=None,
        **kwargs,
    ):
        # SeamlessM4T uses fairseq vocab alignment: <pad>=0, <unk>=1, <s>=2, </s>=3, then SPM pieces[3:]
        if vocab is not None:
            # Ensure vocab is list of (str, float) tuples
            vocab = [(str(item[0]), float(item[1])) for item in vocab]

            # Reorder to fairseq: <pad>, <unk>, <s>, </s>, ... (rest of vocab from SPM[3:])
            vocab_list = []
            vocab_list.append((str(pad_token), 0.0))
            vocab_list.append((str(unk_token), 0.0))
            vocab_list.append((str(bos_token), 0.0))
            vocab_list.append((str(eos_token), 0.0))

            vocab_list.extend(vocab[3:])

            self._vocab_scores = vocab_list
        else:
            self._vocab_scores = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(bos_token), 0.0),
                (str(eos_token), 0.0),
                ("▁", -2.0),
            ]

        # Get unk_token_id from vocab (should be 1 based on fairseq alignment)
        unk_id = 1

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=unk_id,
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

        # Wrap language codes with "__" if needed
        if "__" not in src_lang:
            src_lang = f"__{src_lang}__"
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"

        super().__init__(
            tokenizer_object=tokenizer_object,
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            keep_accents=keep_accents,
            **kwargs,
        )

        # Build fairseq mappings for backward compatibility
        self.fairseq_offset = 1
        self.fairseq_tokens_to_ids = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if "__" not in new_src_lang:
            self._src_lang = f"__{new_src_lang}__"
        else:
            self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        if "__" not in new_tgt_lang:
            self._tgt_lang = f"__{new_tgt_lang}__"
        else:
            self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "eng",
        tgt_texts: Optional[list[str]] = None,
        tgt_lang: str = "fra",
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: Optional[str] = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        if max_length is None:
            max_length = self.model_max_length
        
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        
        if tgt_texts is None:
            return model_inputs
            
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
            
        # Switch to target mode to set the right special tokens
        self._switch_to_target_mode()
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )
        model_inputs["labels"] = labels["input_ids"]
        
        # Switch back to input mode
        self._switch_to_input_mode()
        
        return model_inputs

    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)

        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`src_lang={src_lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        """
        self.cur_lang_code = self.convert_tokens_to_ids(lang)

        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]
        ] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            text (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            src_lang (`str`, *optional*):
                A string representing the source language. If not specified, the last `src_lang` specified (either
                during initialization or when calling this tokenizer) will be used.
            tgt_lang (`str`, *optional*):
                A string representing the target language. If not specified, the last `tgt_lang` specified (either
                during initialization or when calling this tokenizer) will be used.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`TokenizersBackend.__call__`].
        """
        if src_lang is not None:
            self.src_lang = src_lang
        if tgt_lang is not None:
            self.tgt_lang = tgt_lang

        output = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )

        return output


__all__ = ["SeamlessM4TTokenizer"]
