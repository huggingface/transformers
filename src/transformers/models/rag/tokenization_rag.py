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
import warnings
from contextlib import contextmanager
from typing import List, Optional

from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig


logger = logging.get_logger(__name__)


class RagTokenizer:
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder
        self.generator = generator
        self.current_tokenizer = self.question_encoder

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
        from ..auto.tokenization_auto import AutoTokenizer

        config = kwargs.pop("config", None)

        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.question_encoder, subfolder="question_encoder_tokenizer"
        )
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kwargs):
        return self.current_tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.generator.decode(*args, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.current_tokenizer = self.generator
        yield
        self.current_tokenizer = self.question_encoder

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        warnings.warn(
            "`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of 🤗 Transformers. Use the "
            "regular `__call__` method to prepare your inputs and the tokenizer under the `with_target_tokenizer` "
            "context manager to prepare your targets. See the documentation of your specific tokenizer for more "
            "details",
            FutureWarning,
        )
        if max_length is None:
            max_length = self.current_tokenizer.model_max_length
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
        with self.as_target_tokenizer():
            if max_target_length is None:
                max_target_length = self.current_tokenizer.model_max_length
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
        return model_inputs
