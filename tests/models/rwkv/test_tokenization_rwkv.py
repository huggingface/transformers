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


import json
import os
import unittest

from transformers import AutoTokenizer, RWKVWorldTokenizer
from transformers.models.gpt2.tokenization_rwkv import VOCAB_FILES_NAMES
from transformers.testing_utils import require_jinja, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class RWKVWorldTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RWKVWorldTokenizer
    from_pretrained_kwargs = {"add_prefix_space": False}
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        url = "https://huggingface.co/RWKV/rwkv-5-world-169m/blob/main/rwkv_vocab_v20230424.json"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_name = os.path.basename(url)
            file_path = os.path.join(self.tmpdirname, file_name)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File downloaded successfully at {file_path}")
        else:
            print(f"Failed to download the file. HTTP Status Code: {response.status_code}")

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])

    def get_tokenizer(self, **kwargs):
        return RWKVWorldTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_rwkv_world_tokenizer(self):
        tokenizer = RWKVWorldTokenizer(self.vocab_file)
        s1 = tokenizer("Hello")['input_ids']
        self.assertListEqual(s1, [33155])
        s2 = tokenizer("S:2")['input_ids']
        self.assertListEqual(s2, [84, 59, 51])
        s3 = tokenizer("Made in China")['input_ids']
        self.assertListEqual(s3, [23897, 4596, 36473])
        s4 = tokenizer("今天天气不错")['input_ids']
        self.assertListEqual(s4, [10381, 11639, 11639, 13655, 10260, 17631])
        s5 = tokenizer("男：听说你们公司要派你去南方工作?")['input_ids']
        self.assertListEqual(s5, [14601, 19151, 11065, 16735, 10464, 10402, 10678, 11029, 16503, 13818, 10464, 10985, 10934, 13036, 12137, 10460, 64])
        s6 = tokenizer("Pré")['input_ids']
        self.assertListEqual(s6, [1371, 2503])


