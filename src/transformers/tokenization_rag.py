# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RAG."""
import os
from typing import List, Optional

from .configuration_rag import RagConfig
from .file_utils import add_start_docstrings
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)


class RagTokenizer:
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder
        self.generator = generator

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise ValueError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        question_encoder_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_path = os.path.join(save_directory, "generator_tokenizer")
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # dynamically import AutoTokenizer
        from .tokenization_auto import AutoTokenizer

        config = kwargs.pop("config", None)

        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        question_encoder_path = os.path.join(pretrained_model_name_or_path, "question_encoder_tokenizer")
        generator_path = os.path.join(pretrained_model_name_or_path, "generator_tokenizer")
        question_encoder = AutoTokenizer.from_pretrained(question_encoder_path, config=config.question_encoder)
        generator = AutoTokenizer.from_pretrained(generator_path, config=config.generator)
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kwargs):
        return self.question_encoder(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.generator.batch_decode(*args, **kwargs)

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = "np",
        truncation=True,
        **kwargs,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.question_encoder.model_max_length
        model_inputs: BatchEncoding = self.question_encoder(
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
            max_target_length = self.generator.model_max_length
        labels = self.generator(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs
