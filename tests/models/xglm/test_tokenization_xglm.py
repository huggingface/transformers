# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.xglm.tokenization_xglm import XGLMTokenizer
from transformers.testing_utils import require_tokenizers


@require_tokenizers
class XGLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["facebook/xglm-564M"]
    tokenizer_class = XGLMTokenizer
    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [1018, 67, 11, 3194, 6, 61533, 44, 254, 23572, 22, 465, 13323, 4, 53, 319, 67, 84785, 185, 5, 6, 63782, 2530, 3, 322, 2751, 31227, 2751, 31227, 31227, 6, 0, 1075, 0, 1193, 268, 12894, 44036, 2817, 113, 77749, 29, 21257, 72, 13, 31227, 5, 2079, 246, 72, 53, 10845, 246, 72, 30937, 20933, 1271, 256, 206, 7667]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活的', '真', '<unk>', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<s>', '▁hi', '<s>', '▁there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真<unk>是 Hi Hello Hi Hello Hello <s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "facebook/xglm-564M"

        tokenizer = XGLMTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tokenizer]
