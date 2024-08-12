# coding=utf-8
# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Mamba2."""

from ...utils import logging
from ..gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast


logger = logging.get_logger(__name__)


class Mamba2TokenizerFast(GPTNeoXTokenizerFast):
    """
    Utility class to overwrite the padding side of a GPTNeoXTokenizerFast tokenizer.
    """

    padding_side = "left"

    def __init__(self, *args, **kwargs):
        # Silently remove padding side on init
        kwargs.pop("padding_side", None)

        # Otherwise we take over all other parameters
        super().__init__(*args, **kwargs, padding_side="left")
