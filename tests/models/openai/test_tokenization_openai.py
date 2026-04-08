# Copyright 2018 The Google AI Language Team Authors.
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

from transformers import OpenAIGPTTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class OpenAIGPTTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openai-community/openai-gpt"

    tokenizer_class = OpenAIGPTTokenizer
    integration_expected_tokens = ['this</w>', 'is</w>', 'a</w>', 'test</w>', '<unk>', 'i</w>', 'was</w>', 'born</w>', 'in</w>', '9', '2000</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'false</w>', '.</w>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', 'hi</w>', 'hello</w>', 'hi</w>', 'hello</w>', 'hello</w>', '<</w>', 's</w>', '></w>', 'hi</w>', '<</w>', 's</w>', '></w>', 'there</w>', 'the</w>', 'following</w>', 'string</w>', 'should</w>', 'be</w>', 'properly</w>', 'en', 'coded</w>', ':</w>', 'hello</w>', '.</w>', 'but</w>', 'ird</w>', 'and</w>', '<unk>', 'ird</w>', '<unk>', 'hey</w>', 'how</w>', 'are</w>', 'you</w>', 'doing</w>']  # fmt: skip
    integration_expected_token_ids = [616, 544, 246, 2345, 0, 249, 509, 3105, 500, 53, 28654, 240, 488, 616, 544, 6843, 239, 0, 0, 0, 0, 0, 0, 3569, 3570, 3569, 3570, 3570, 295, 252, 290, 3569, 295, 252, 290, 655, 481, 2890, 6422, 994, 580, 6506, 496, 20925, 271, 3570, 239, 568, 13926, 488, 0, 13926, 0, 2229, 718, 640, 512, 1273]  # fmt: skip
    expected_tokens_from_ids = ['this</w>', 'is</w>', 'a</w>', 'test</w>', '<unk>', 'i</w>', 'was</w>', 'born</w>', 'in</w>', '9', '2000</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'false</w>', '.</w>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', 'hi</w>', 'hello</w>', 'hi</w>', 'hello</w>', 'hello</w>', '<</w>', 's</w>', '></w>', 'hi</w>', '<</w>', 's</w>', '></w>', 'there</w>', 'the</w>', 'following</w>', 'string</w>', 'should</w>', 'be</w>', 'properly</w>', 'en', 'coded</w>', ':</w>', 'hello</w>', '.</w>', 'but</w>', 'ird</w>', 'and</w>', '<unk>', 'ird</w>', '<unk>', 'hey</w>', 'how</w>', 'are</w>', 'you</w>', 'doing</w>']  # fmt: skip
    integration_expected_decoded_text = "this is a test <unk>i was born in 92000 , and this is false . <unk><unk><unk><unk><unk><unk>hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello . but ird and <unk>ird <unk>hey how are you doing"
