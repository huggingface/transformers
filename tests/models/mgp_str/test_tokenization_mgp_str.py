# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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


import json
import os
import unittest

from transformers import MgpstrTokenizer
from transformers.models.mgp_str.tokenization_mgp_str import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class MgpstrTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "alibaba-damo/mgp-str-base"
    tokenizer_class = MgpstrTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {}
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab = ['[GO]', '[s]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']  # fmt: skip
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs):
        return MgpstrTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "tester"
        output_text = "tester"
        return input_text, output_text

    @unittest.skip("MGP-STR always lower cases letters.")
    def test_added_tokens_do_lower_case(self):
        pass

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode([special_token], add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2.replace(" ", ""), output_text)

    @unittest.skip("MGP-STR tokenizer only handles one sequence.")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip("inputs cannot be pretokenized in MgpstrTokenizer")
    def test_pretokenized_inputs(self):
        pass
