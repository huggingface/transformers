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


from transformers import AutoTokenizer
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
from transformers.testing_utils import require_tokenizers

from ..bert import test_tokenization_bert


# TODO: Ita remove this test file?
@require_tokenizers
class DistilBertTokenizationTest(test_tokenization_bert.BertTokenizationTest):
    tokenizer_class = DistilBertTokenizer
    rust_tokenizer_class = DistilBertTokenizer
    test_rust_tokenizer = False
    from_pretrained_id = "distilbert/distilbert-base-uncased"

    integration_expected_tokens = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [2023, 2003, 1037, 3231, 100, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 1910, 100, 1916, 1921, 100, 100, 7632, 7592, 7632, 7592, 7592, 1026, 1055, 1028, 7632, 1026, 1055, 1028, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725]  # fmt: skip
    expected_tokens_from_ids = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_decoded_text = "this is a test [UNK] i was born in 92000, and this is false. 生 [UNK] 的 真 [UNK] [UNK] hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello. but ird and [UNK] ird [UNK] hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "distilbert/distilbert-base-uncased"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]
