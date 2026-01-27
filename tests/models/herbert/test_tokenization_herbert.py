# Copyright 2020 HuggingFace Inc. team.
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

from transformers import HerbertTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class HerbertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "allegro/herbert-base-cased"
    tokenizer_class = HerbertTokenizer

    integration_expected_tokens = ['T', 'his</w>', 'is</w>', 'a</w>', 'test</w>', '<unk>', 'I</w>', 'was</w>', 'bor', 'n</w>', 'in</w>', '9', '2000</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'fal', 's', 'é</w>', '.</w>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '是</w>', 'H', 'i</w>', 'Hel', 'lo</w>', 'H', 'i</w>', 'Hel', 'lo</w>', 'Hel', 'lo</w>', '<s>', 'hi</w>', '<s>', 'ther', 'e</w>', 'The</w>', 'fol', 'low', 'ing</w>', 'str', 'ing</w>', 'sho', 'uld</w>', 'be</w>', 'pro', 'per', 'ly</w>', 'en', 'c', 'ode', 'd</w>', ':</w>', 'Hel', 'lo</w>', '.</w>', 'Bu', 't</w>', 'ir', 'd</w>', 'and</w>', '<unk>', 'ี</w>', 'ir', 'd</w>', 'ด</w>', 'He', 'y</w>', 'ho', 'w</w>', 'are</w>', 'you</w>', 'do', 'ing</w>']  # fmt: skip
    integration_expected_token_ids = [56, 22855, 6869, 1011, 14825, 3, 1056, 9873, 2822, 1016, 2651, 29, 3450, 1947, 7158, 48846, 6869, 7355, 87, 1093, 1899, 3, 3, 3, 3, 3, 1776, 44, 1009, 12156, 6170, 44, 1009, 12156, 6170, 12156, 6170, 0, 21566, 0, 40445, 1015, 7117, 9929, 13194, 5129, 15948, 5129, 14924, 48273, 11072, 2088, 3040, 8172, 2058, 71, 3909, 1038, 1335, 12156, 6170, 1899, 3025, 1026, 17435, 1038, 7158, 3, 1085, 17435, 1038, 1579, 4596, 1005, 3145, 1019, 25720, 20254, 2065, 5129]  # fmt: skip
    expected_tokens_from_ids = ['T', 'his</w>', 'is</w>', 'a</w>', 'test</w>', '<unk>', 'I</w>', 'was</w>', 'bor', 'n</w>', 'in</w>', '9', '2000</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'fal', 's', 'é</w>', '.</w>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '是</w>', 'H', 'i</w>', 'Hel', 'lo</w>', 'H', 'i</w>', 'Hel', 'lo</w>', 'Hel', 'lo</w>', '<s>', 'hi</w>', '<s>', 'ther', 'e</w>', 'The</w>', 'fol', 'low', 'ing</w>', 'str', 'ing</w>', 'sho', 'uld</w>', 'be</w>', 'pro', 'per', 'ly</w>', 'en', 'c', 'ode', 'd</w>', ':</w>', 'Hel', 'lo</w>', '.</w>', 'Bu', 't</w>', 'ir', 'd</w>', 'and</w>', '<unk>', 'ี</w>', 'ir', 'd</w>', 'ด</w>', 'He', 'y</w>', 'ho', 'w</w>', 'are</w>', 'you</w>', 'do', 'ing</w>']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk>I was born in 92000 , and this is falsé . <unk><unk><unk><unk><unk>是 Hi Hello Hi Hello Hello <s>hi <s>there The following string should be properly encoded : Hello . But ird and <unk>ี ird ด Hey how are you doing"
