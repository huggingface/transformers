# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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

from transformers import PreTrainedTokenizer, BertTokenizer, BertTokenizerFast
from transformers.tokenization_gpt2 import GPT2Tokenizer

from .utils import slow


class TokenizerUtilsTest(unittest.TestCase):
    def check_tokenizer_from_pretrained(self, tokenizer_class):
        s3_models = list(tokenizer_class.max_model_input_sizes.keys())
        for model_name in s3_models[:1]:
            tokenizer = tokenizer_class.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, tokenizer_class)
            self.assertIsInstance(tokenizer, PreTrainedTokenizer)

            for special_tok in tokenizer.all_special_tokens:
                self.assertIsInstance(special_tok, str)
                special_tok_id = tokenizer.convert_tokens_to_ids(special_tok)
                self.assertIsInstance(special_tok_id, int)

    @slow
    def test_pretrained_tokenizers(self):
        self.check_tokenizer_from_pretrained(GPT2Tokenizer)

    def test_batch_encoding_pickle(self):
        from pickle import loads, dumps

        # Get a slow & a fast tokenizer
        tok_slow = BertTokenizer.from_pretrained("bert-base-cased")
        tok_fast = BertTokenizerFast.from_pretrained("bert-base-cased")

        # Encode a sentence
        be_slow = tok_slow.encode_plus("This is a dummy input sentence")
        be_fast = tok_fast.encode_plus("This is a dummy input sentence")

        # Make sure both are pickable
        be_slow_data = dumps(be_slow)
        be_fast_data = dumps(be_fast)

        # Try to restore
        be_slow_pickled = loads(be_slow_data)
        be_fast_pickled = loads(be_fast_data)

        # Ensure pickled objects keeps the is_fast attribute
        self.assertFalse(be_slow_pickled.is_fast)
        self.assertTrue(be_fast_pickled.is_fast)

        # Ensure .data match
        self.assertDictEqual(be_slow_pickled.data, be_slow.data)
        self.assertDictEqual(be_fast_pickled.data, be_fast.data)

        # Ensure .encodings match
        self.assertIsNone(be_slow_pickled.encodings)
        self.assertEqual(len(be_fast_pickled.encodings), len(be_fast.encodings))