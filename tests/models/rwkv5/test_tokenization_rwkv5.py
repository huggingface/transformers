# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import os
import unittest

from transformers.testing_utils import get_tests_dir, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers import RWKVWorldTokenizer


@require_torch
class RWKVWorldTokenizationTest(unittest.TestCase):
    def test_rwkv_world_tokenizer_encode(self):
        tokenizer = RWKVWorldTokenizer.from_pretrained("RWKV/rwkv-5-world-169m")
        s1 = tokenizer("Hello")["input_ids"]
        self.assertListEqual(s1, [33155])
        s2 = tokenizer("S:2")["input_ids"]
        self.assertListEqual(s2, [84, 59, 51])
        s3 = tokenizer("Made in China")["input_ids"]
        self.assertListEqual(s3, [23897, 4596, 36473])
        s4 = tokenizer("今天天气不错")["input_ids"]
        self.assertListEqual(s4, [10381, 11639, 11639, 13655, 10260, 17631])
        s5 = tokenizer("男：听说你们公司要派你去南方工作?")["input_ids"]
        self.assertListEqual(
            s5,
            [
                14601,
                19151,
                11065,
                16735,
                10464,
                10402,
                10678,
                11029,
                16503,
                13818,
                10464,
                10985,
                10934,
                13036,
                12137,
                10460,
                64,
            ],
        )
        s6 = tokenizer("Pré")["input_ids"]
        self.assertListEqual(s6, [1371, 2503])
