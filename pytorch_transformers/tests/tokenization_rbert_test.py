# coding=utf-8
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from io import open

from pytorch_transformers.tokenization_bert import VOCAB_FILES_NAMES
from pytorch_transformers.tokenization_rbert import (RBertTokenizer)
from .tokenization_tests_commons import CommonTestCases


class RBertTokenizationTest(CommonTestCases.CommonTokenizerTester):
    tokenizer_class = RBertTokenizer

    def setUp(self):
        super(RBertTokenizationTest, self).setUp()

        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ",", "low", "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, "w", encoding='utf-8') as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        return RBertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self):
        input_text = u"UNwant\u00E9d,running"
        output_text = u"unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = RBertTokenizer(self.vocab_file)

        tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_sequence_builders(self):
        tokenizer = RBertTokenizer.from_pretrained("bert-base-uncased")

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.add_special_tokens_single_sentence(text)
        encoded_pair = tokenizer.add_special_tokens_sentences_pair(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_relationship_sequence_builder(self):
        tokenizer = RBertTokenizer.from_pretrained("bert-base-uncased")

        text_encoded = tokenizer.encode_with_relationship("The first entity is GENE1 which is related to GENE2",
                                                          e1_offset_tup=(20, 25,),
                                                          e2_offset_tup=(46, 51,),
                                                          add_special_tokens=True)

        ent1_token_id = tokenizer._convert_token_to_id(tokenizer.ent1_sep_token)
        ent2_token_id = tokenizer._convert_token_to_id(tokenizer.ent2_sep_token)

        special_tokens_ordered = list(filter(lambda n: n in [101, ent1_token_id, ent2_token_id, 102], text_encoded))

        assert special_tokens_ordered[0] == 101
        assert special_tokens_ordered[1] == ent1_token_id
        assert special_tokens_ordered[2] == ent1_token_id
        assert special_tokens_ordered[3] == ent2_token_id
        assert special_tokens_ordered[4] == ent2_token_id
        assert special_tokens_ordered[5] == 102


if __name__ == '__main__':
    unittest.main()
