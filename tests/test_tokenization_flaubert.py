# coding=utf-8
# Copyright 2020 Huggingface
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


from transformers import (
    FlaubertTokenizer,
    FLAUBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST
)

from .test_tokenization_xlm import XLMTokenizationTest


class FlaubertTokenizationTest(XLMTokenizationTest):

    pretrained_vocab_checkpoints = FLAUBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST
    tokenizer_class = FlaubertTokenizer
