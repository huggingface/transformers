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

from .configuration_rag import RagConfig
from .tokenization_auto import AutoTokenizer
from .utils import logging


logger = logging.get_logger(__name__)


class RagTokenizer:
    def __init__(self, question_encoder_tokenizer, generator_tokenizer):
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.generator_tokenizer = generator_tokenizer

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        question_encoder_tokenizer_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_tokenizer_path = os.path.join(save_directory, "generator_tokenizer")
        self.question_encoder_tokenizer.save_pretrained(question_encoder_tokenizer_path)
        self.generator_tokenizer.save_pretrained(generator_tokenizer_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        question_encoder_tokenizer_path = os.path.join(pretrained_model_name_or_path, "question_encoder_tokenizer")
        generator_tokenizer_path = os.path.join(pretrained_model_name_or_path, "generator_tokenizer")
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(
            question_encoder_tokenizer_path, config=config.question_encoder
        )
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path, config=config.generator)
        return cls(question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer)

    def __call__(self, *args, **kwargs):
        # TODO
        pass
