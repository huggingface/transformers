# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import inspect
import json
import os
import tempfile
import unittest

from transformers.models.speech_to_text_2 import Speech2Text2Tokenizer
from transformers.models.speech_to_text_2.tokenization_speech_to_text_2 import VOCAB_FILES_NAMES

from .test_tokenization_common import TokenizerTesterMixin


class SpeechToTextTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Speech2Text2Tokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = "<s> <pad> </s> <unk> here@@ a couple of@@ words for the he@@ re@@ vocab".split(" ")
        merges = ["he re</w> 123", "here a 1456"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "vocab")
        self.assertEqual(len(vocab_keys), 14)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 14)

    def test_tokenizer_decode(self):
        tokenizer = Speech2Text2Tokenizer.from_pretrained(self.tmpdirname)

        # make sure @@ is correctly concatenated
        token_ids = [4, 6, 8, 7, 10]  # ["here@@", "couple", "words", "of@@", "the"]
        output_string = tokenizer.decode(token_ids)

        self.assertTrue(output_string == "herecouple words ofthe")

    def test_load_no_merges_file(self):
        tokenizer = Speech2Text2Tokenizer.from_pretrained(self.tmpdirname)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            tokenizer.save_pretrained(tmp_dirname)
            os.remove(os.path.join(tmp_dirname, "merges.txt"))

            # load tokenizer without merges file should not throw an error
            tokenizer = Speech2Text2Tokenizer.from_pretrained(tmp_dirname)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            # save tokenizer and load again
            tokenizer.save_pretrained(tmp_dirname)
            tokenizer = Speech2Text2Tokenizer.from_pretrained(tmp_dirname)

        self.assertIsNotNone(tokenizer)

    # overwrite since merges_file is optional
    def test_tokenizer_slow_store_full_signature(self):
        if not self.test_slow_tokenizer:
            return

        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty and parameter_name != "merges_file":
                self.assertIn(parameter_name, tokenizer.init_kwargs)
