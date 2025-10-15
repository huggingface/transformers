# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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
from transformers import AutoTokenizer, SqueezeBertTokenizer
from transformers.testing_utils import require_tokenizers


# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""

expected_tokens = ['this', 'is', 'a', 'test', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']
expected_token_ids = [101, 2023, 2003, 1037, 3231, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 1910, 100, 1916, 1921, 100, 100, 7632, 7592, 7632, 7592, 7592, 1026, 1055, 1028, 7632, 1026, 1055, 1028, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725, 102]


@require_tokenizers
class SqueezeBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SqueezeBertTokenizer
    rust_tokenizer_class = SqueezeBertTokenizer
    from_pretrained_id = "squeezebert/squeezebert-uncased"
    test_slow_tokenizer = True
    test_rust_tokenizer = False  # we're going to just test the fast one I'll remove this
    space_between_special_tokens = False
    from_pretrained_kwargs = {}
    test_seq2seq = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "squeezebert/squeezebert-uncased"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

    def test_integration_expected_tokens(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.tokenize(input_string), expected_tokens)
        self.assertEqual(tokenizer.encode(input_string), expected_token_ids)

    def test_integration_expected_token_ids(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.encode(input_string), expected_token_ids)
