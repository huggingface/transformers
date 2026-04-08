# Copyright 2024 The HuggingFace Inc. team.
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
from transformers.models.splinter.tokenization_splinter import SplinterTokenizer


class SplinterTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SplinterTokenizer
    from_pretrained_id = "tau/splinter-base"
    test_tokenizer_from_extractor = (
        False  # Splinter already only provides a vocab.txt file, so we don't need to extract vocab to retest this
    )

    integration_expected_tokens = ['This', 'is', 'a', 'test', '[UNK]', 'I', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'f', '##als', '##é', '.', '生', '[UNK]', '[UNK]', '真', '[UNK]', '[UNK]', 'Hi', 'Hello', 'Hi', 'Hello', 'Hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'The', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'Hello', '.', 'But', 'i', '##rd', 'and', '[UNK]', 'i', '##rd', '[UNK]', 'Hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [1188, 1110, 170, 2774, 100, 146, 1108, 1255, 1107, 5556, 7629, 1568, 117, 1105, 1142, 1110, 175, 7264, 2744, 119, 1056, 100, 100, 1061, 100, 100, 8790, 8667, 8790, 8667, 8667, 133, 188, 135, 20844, 133, 188, 135, 1175, 1109, 1378, 5101, 1431, 1129, 7513, 12544, 131, 8667, 119, 1252, 178, 2956, 1105, 100, 178, 2956, 100, 4403, 1293, 1132, 1128, 1833]  # fmt: skip
    expected_tokens_from_ids = ['This', 'is', 'a', 'test', '[UNK]', 'I', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'f', '##als', '##é', '.', '生', '[UNK]', '[UNK]', '真', '[UNK]', '[UNK]', 'Hi', 'Hello', 'Hi', 'Hello', 'Hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'The', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'Hello', '.', 'But', 'i', '##rd', 'and', '[UNK]', 'i', '##rd', '[UNK]', 'Hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test [UNK] I was born in 92000, and this is falsé. 生 [UNK] [UNK] 真 [UNK] [UNK] Hi Hello Hi Hello Hello < s > hi < s > there The following string should be properly encoded : Hello. But ird and [UNK] ird [UNK] Hey how are you doing"
