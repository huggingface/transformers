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

from transformers import FunnelTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class FunnelTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "funnel-transformer/small"
    tokenizer_class = FunnelTokenizer

    integration_expected_tokens = ['<unk>', 'is', 'a', 'test', '<unk>', '<unk>', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', '<unk>', '.', '生', '<unk>', '的', '真', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<s>', 'hi', '<s>', 'there', '<unk>', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', '<unk>', '.', '<unk>', 'ir', '##d', 'and', '<unk>', 'ir', '##d', '<unk>', '<unk>', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [100, 2003, 1037, 3231, 100, 100, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 100, 1012, 1910, 100, 1916, 1921, 100, 100, 100, 100, 100, 100, 100, 96, 7632, 96, 2045, 100, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 100, 1012, 100, 20868, 2094, 1998, 100, 20868, 2094, 100, 100, 2129, 2024, 2017, 2725]  # fmt: skip
    expected_tokens_from_ids = ['<unk>', 'is', 'a', 'test', '<unk>', '<unk>', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', '<unk>', '.', '生', '<unk>', '的', '真', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<s>', 'hi', '<s>', 'there', '<unk>', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', '<unk>', '.', '<unk>', 'ir', '##d', 'and', '<unk>', 'ir', '##d', '<unk>', '<unk>', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_decoded_text = "<unk> is a test <unk> <unk> was born in 92000, and this is <unk>. 生 <unk> 的 真 <unk> <unk> <unk> <unk> <unk> <unk> <unk> <s> hi <s> there <unk> following string should be properly encoded : <unk>. <unk> ird and <unk> ird <unk> <unk> how are you doing"
