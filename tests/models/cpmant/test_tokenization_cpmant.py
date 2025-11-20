# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team.
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

from transformers.models.cpmant.tokenization_cpmant import VOCAB_FILES_NAMES, CpmAntTokenizer
from transformers.testing_utils import require_rjieba, tooslow

from ...test_tokenization_common import TokenizerTesterMixin


@require_rjieba
class CPMAntTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openbmb/cpm-ant-10b"
    tokenizer_class = CpmAntTokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab_tokens = [
            "<d>",
            "</d>",
            "<s>",
            "</s>",
            "</_>",
            "<unk>",
            "<pad>",
            "</n>",
            "我",
            "是",
            "C",
            "P",
            "M",
            "A",
            "n",
            "t",
        ]
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    @tooslow
    def test_pre_tokenization(self):
        tokenizer = CpmAntTokenizer.from_pretrained("openbmb/cpm-ant-10b")
        texts = "今天天气真好！"
        rjieba_tokens = ["今天", "天气", "真", "好", "！"]
        tokens = tokenizer.tokenize(texts)
        self.assertListEqual(tokens, rjieba_tokens)
        normalized_text = "今天天气真好！"
        input_tokens = [tokenizer.bos_token] + tokens

        input_rjieba_tokens = [6, 9802, 14962, 2082, 831, 244]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_rjieba_tokens)

        reconstructed_text = tokenizer.decode(input_rjieba_tokens)
        self.assertEqual(reconstructed_text, normalized_text)
