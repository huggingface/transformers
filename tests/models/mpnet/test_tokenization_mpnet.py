# Copyright 2020 The HuggingFace Inc. team, Microsoft Corporation.
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

from transformers.models.mpnet.tokenization_mpnet import MPNetTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class MPNetTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/mpnet-base"
    tokenizer_class = MPNetTokenizer

    integration_expected_tokens = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<s>', 'hi', '<s>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [2027, 2007, 1041, 3235, 104, 1049, 2005, 2145, 2003, 6231, 8893, 2696, 1014, 2002, 2027, 2007, 6274, 1016, 1914, 104, 1920, 1925, 104, 104, 7636, 7596, 7636, 7596, 7596, 0, 7636, 0, 2049, 2000, 2210, 5168, 2327, 2026, 7923, 12363, 1028, 7596, 1016, 2025, 20872, 2098, 2002, 104, 20872, 2098, 104, 4935, 2133, 2028, 2021, 2729]  # fmt: skip
    expected_tokens_from_ids = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<s>', 'hi', '<s>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_decoded_text = "this is a test [UNK] i was born in 92000, and this is false. 生 [UNK] 的 真 [UNK] [UNK] hi hello hi hello hello <s> hi <s> there the following string should be properly encoded : hello. but ird and [UNK] ird [UNK] hey how are you doing"
