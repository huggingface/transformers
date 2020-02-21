# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RoBERTa."""


import logging

from tokenizers.processors import RobertaProcessing

from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
        "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
        "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
        "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    },
    "merges_file": {
        "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
        "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
        "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
        "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
        "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
        "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class RobertaTokenizer(GPT2Tokenizer):
    """
    RoBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        RoBERTa does not make use of token type ids, therefore a list of zeros is returned.

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, add_special_tokens=False, **kwargs):
        if "add_prefix_space" in kwargs:
            add_prefix_space = kwargs["add_prefix_space"]
        else:
            add_prefix_space = add_special_tokens
        if add_prefix_space and not text[0].isspace():
            text = " " + text
        return text


class RobertaTokenizerFast(GPT2TokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs
    ):
        kwargs.setdefault("pad_token", pad_token)
        kwargs.setdefault("sep_token", sep_token)
        kwargs.setdefault("cls_token", cls_token)
        kwargs.setdefault("mask_token", mask_token)

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.tokenizer._tokenizer.post_processor = RobertaProcessing(
            (sep_token, self.sep_token_id), (cls_token, self.cls_token_id)
        )

        # As we override the post_processor post super.__init__ the computed num_added_tokens is wrong in super().
        # We need to recompute max_len according to the newly register post_processor to get real values.
        self.max_len_single_sentence = self.max_len - self.num_added_tokens(False)  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - self.num_added_tokens(True)  # take into account special tokens

        logger.warning(
            "RobertaTokenizerFast has an issue when working on mask language modeling "
            "where it introduces an extra encoded space before the mask token."
            "See https://github.com/huggingface/transformers/pull/2778 for more information."
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
