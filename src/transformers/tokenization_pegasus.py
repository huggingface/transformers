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
from typing import Dict, List, Optional

from .file_utils import add_start_docstrings
from .tokenization_reformer import ReformerTokenizer
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding


class PegasusTokenizer(ReformerTokenizer):
    offset = 103  # entries 2-104 are only used for pretraining
    vocab_files_names = {"vocab_file": "spiece.model"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dont use reserved words added_token_encoder, added_tokens_decoder because of
        # AssertionError: Non-consecutive added token '1' found. in from_pretrained
        assert len(self.added_tokens_decoder) == 0
        self.encoder: Dict[int, str] = {0: self.pad_token, 1: self.eos_token}
        # entries 2-104 are only used for pretraining and called unk_2, ...unk_104
        self.encoder.update({i: f"unk_{i}" for i in range(2, self.offset + 2)})
        self.decoder: Dict[str, int] = {v: k for k, v in self.encoder.items()}

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        if token in self.decoder:
            return self.decoder[token]
        elif token in self.added_tokens_decoder:
            return self.added_tokens_decoder[token]
        sp_id = self.sp_model.piece_to_id(token)
        return sp_id + self.offset

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.encoder:
            return self.encoder[index]
        elif index in self.added_tokens_encoder:
            return self.added_tokens_encoder[index]
        else:
            # assert index > self.offset, f"cannot decode ids between 2 and {self.offset}. Got {index}"
            token = self.sp_model.IdToPiece(index - self.offset)
        return token

    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.offset

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def num_special_tokens_to_add(self, pair=False):
        """Just EOS"""
        return 1

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
        - pair of sequences: ``A B </s>``  (not intended use)

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
        """
        Prepare model inputs for summarization or translation.

        """
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
        # TODO(@sshleifer): maybe tgt_texts = [self.pad_token + t for t in tgt_texts]  # add decoder_start_token_id
        labels: BatchEncoding = self(tgt_texts, **tokenizer_kwargs)["input_ids"]
        model_inputs["labels"] = labels
        # for k, v in decoder_inputs.items():
        #    model_inputs[f"decoder_{k}"] = v
        return model_inputs
