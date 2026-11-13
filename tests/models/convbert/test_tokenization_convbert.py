# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import ConvBertTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class ConvBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "YituTech/conv-bert-base"
    tokenizer_class = ConvBertTokenizer

    # Matches ConvBertTokenizer.from_pretrained("YituTech/conv-bert-base"). ConvBERT uses its own WordPiece
    # vocabulary, so these differ from BERT's despite the shared implementation.
    integration_expected_tokens = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [2023, 2003, 1037, 3231, 100, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 100, 100, 100, 100, 100, 100, 7632, 7592, 7632, 7592, 7592, 1026, 1055, 1028, 7632, 1026, 1055, 1028, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725]  # fmt: skip
    integration_expected_decoded_text = "this is a test [UNK] i was born in 92000, and this is false. [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello. but ird and [UNK] ird [UNK] hey how are you doing"

    def test_tokenizer_is_bert_subclass(self):
        # ConvBertTokenizer adds no behavior of its own; it exists so that the ConvBERT checkpoints resolve to a
        # tokenizer named after the model. Guard the inheritance so the class cannot silently drift from BERT's.
        self.assertTrue(issubclass(ConvBertTokenizer, BertTokenizer))
