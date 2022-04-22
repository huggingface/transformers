# coding=utf-8
# Copyright 2022 Leon Derczynski. All rights reserved.
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
""" Testing suite for the MobileBERT tokenizer. """


import filecmp
import unittest
from tempfile import NamedTemporaryFile

import requests
from transformers import MobileBertTokenizer, MobileBertTokenizerFast
from transformers.testing_utils import require_tokenizers

from ..bert.test_tokenization_bert import BertTokenizationTest


# disable duplicate incorporation of tests from parent class in this module
BertTokenizationTest.__test__ = False


@require_tokenizers
class MobileBertTokenizationTest(BertTokenizationTest, unittest.TestCase):

    # override this module's disabling of parent
    __test__ = True

    tokenizer_class = MobileBertTokenizer
    test_slow_tokenizer = True

    rust_tokenizer_class = MobileBertTokenizerFast
    test_rust_tokenizer = True

    pre_trained_model_path = "google/mobilebert-uncased"

    def setUp(self):
        super().setUp()
        self.tokenizers_list = [
            (tokenizer_def[0], self.pre_trained_model_path, tokenizer_def[2])  # else the 'google/' prefix is stripped
            for tokenizer_def in self.tokenizers_list
        ]

    def test_mobilebert_tokenizer_uses_bert_params(self):
        with NamedTemporaryFile(buffering=0) as bert_vocab_file, NamedTemporaryFile(
            buffering=0
        ) as bert_merge_file, NamedTemporaryFile(buffering=0) as mobilebert_vocab_file, NamedTemporaryFile(
            buffering=0
        ) as mobilebert_merge_file:  # buffering=0 is shorter than four flush()es before cmp()
            bert_merge_file.write(requests.get("https://huggingface.co/bert-base-uncased/raw/main/merges.txt").content)
            mobilebert_merge_file.write(
                requests.get("https://huggingface.co/google/mobilebert-uncased/raw/main/merges.txt").content
            )
            self.assertTrue(filecmp.cmp(bert_merge_file.name, mobilebert_merge_file.name), "Merge files don't match")

            bert_vocab_file.write(requests.get("https://huggingface.co/bert-base-uncased/raw/main/vocab.json").content)
            mobilebert_vocab_file.write(
                requests.get("https://huggingface.co/google/mobilebert-uncased/raw/main/vocab.json").content
            )
            self.assertTrue(filecmp.cmp(bert_vocab_file.name, mobilebert_vocab_file.name), "Vocab files don't match")
