# coding=utf-8
# Copyright 2020 HuggingFace Inc..
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


import unittest

import torch

from transformers import AutoTokenizer

from .utils import require_torch


@require_torch
class TokenizerUtilsBatchTest(unittest.TestCase):
    def test_batch_tokenizers(self):
        pretrained = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        text = "My"
        text1 = "features are ok"

        batch = [text, text1]

        batch_encoding = tokenizer.batch_encode_plus(batch, return_tensors="pt", add_special_tokens=False)

        text_encoding = tokenizer.encode_plus(
            text, return_tensors="pt", add_special_tokens=False, max_length=3, pad_to_max_length=True
        )

        self.assertTrue(torch.all(batch_encoding["token_type_ids"][0].eq(text_encoding["token_type_ids"][0])))
