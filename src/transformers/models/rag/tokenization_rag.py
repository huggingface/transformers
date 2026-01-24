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
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
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

    def encode(self, *args, **kwargs):
        return self.current_tokenizer.encode(*args, **kwargs)

    @property
    def patch_token(self):
        return getattr(self.current_tokenizer, "patch_token", None)

    @patch_token.setter
    def patch_token(self, value):
        setattr(self.current_tokenizer, "patch_token", value)

    @property
    def patch_token_id(self):
        patch_token_id = getattr(self.current_tokenizer, "patch_token_id", None)
        if patch_token_id is not None:
            return patch_token_id

        patch_token = getattr(self.current_tokenizer, "patch_token", None)
        if patch_token is None:
            return None

        convert = getattr(self.current_tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None

        return convert(patch_token)

    @patch_token_id.setter
    def patch_token_id(self, value):
        # Store/forward to the currently active tokenizer
        setattr(self.current_tokenizer, "patch_token_id", value)

    def batch_decode(self, *args, **kwargs):
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.generator.decode(*args, **kwargs)

    def _switch_to_input_mode(self):
        self.current_tokenizer = self.question_encoder

    def _switch_to_target_mode(self):
        self.current_tokenizer = self.generator


__all__ = ["RagTokenizer"]
