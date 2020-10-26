# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
from typing import List, Optional

from .file_utils import add_start_docstrings, is_sentencepiece_available
from .tokenization_reformer_fast import ReformerTokenizerFast
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding


if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer
else:
    PegasusTokenizer = None


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://cdn.huggingface.co/google/pegasus-xsum/spiece.model"},
    "tokenizer_file": {"google/pegasus-xsum": "https://cdn.huggingface.co/google/pegasus-xsum/tokenizer.json"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}


class PegasusTokenizerFast(ReformerTokenizerFast):
    offset = 103  # entries 2-104 are only used for pretraining
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = PegasusTokenizer

    # def num_special_tokens_to_add(self, pair=False):
    #     """Just EOS"""
    #     return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        assert all_special_ids == set([0, 1])
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

        - single sequence: ``X </s>``
        - pair of sequences: ``A B </s>`` (not intended use)

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        return_tensors: str = "pt",
        truncation=True,
        padding="longest",
        **unused,
    ) -> BatchEncoding:
        if "" in src_texts:
            raise ValueError(f"found empty string in src_texts: {src_texts}")
        tokenizer_kwargs = dict(
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        model_inputs: BatchEncoding = self(src_texts, **tokenizer_kwargs)
        if tgt_texts is None:
            return model_inputs
        if max_target_length is not None:
            tokenizer_kwargs["max_length"] = max_target_length
        labels: BatchEncoding = self(tgt_texts, **tokenizer_kwargs)["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs
