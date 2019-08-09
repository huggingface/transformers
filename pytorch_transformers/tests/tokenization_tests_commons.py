# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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
import sys
from io import open
import tempfile
import shutil
import unittest

if sys.version_info[0] == 2:
    import cPickle as pickle

    class TemporaryDirectory(object):
        """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""
        def __enter__(self):
            self.name = tempfile.mkdtemp()
            return self.name
        def __exit__(self, exc_type, exc_value, traceback):
            shutil.rmtree(self.name)
else:
    import pickle
    TemporaryDirectory = tempfile.TemporaryDirectory
    unicode = str


class CommonTestCases:

    class CommonTokenizerTester(unittest.TestCase):

        tokenizer_class = None

        def setUp(self):
            self.tmpdirname = tempfile.mkdtemp()

        def tearDown(self):
            shutil.rmtree(self.tmpdirname)

        def get_tokenizer(self):
            raise NotImplementedError

        def get_input_output_texts(self):
            raise NotImplementedError

        def test_save_and_load_tokenizer(self):
            tokenizer = self.get_tokenizer()

            before_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running")

            with TemporaryDirectory() as tmpdirname:
                tokenizer.save_pretrained(tmpdirname)
                tokenizer = tokenizer.from_pretrained(tmpdirname)

            after_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running")
            self.assertListEqual(before_tokens, after_tokens)

        def test_pickle_tokenizer(self):
            tokenizer = self.get_tokenizer()
            self.assertIsNotNone(tokenizer)

            text = u"Munich and Berlin are nice cities"
            subwords = tokenizer.tokenize(text)

            with TemporaryDirectory() as tmpdirname:

                filename = os.path.join(tmpdirname, u"tokenizer.bin")
                pickle.dump(tokenizer, open(filename, "wb"))

                tokenizer_new = pickle.load(open(filename, "rb"))

            subwords_loaded = tokenizer_new.tokenize(text)

            self.assertListEqual(subwords, subwords_loaded)


        def test_add_tokens_tokenizer(self):
            tokenizer = self.get_tokenizer()

            vocab_size = tokenizer.vocab_size
            all_size = len(tokenizer)

            self.assertNotEqual(vocab_size, 0)
            self.assertEqual(vocab_size, all_size)

            new_toks = ["aaaaabbbbbb", "cccccccccdddddddd"]
            added_toks = tokenizer.add_tokens(new_toks)
            vocab_size_2 = tokenizer.vocab_size
            all_size_2 = len(tokenizer)

            self.assertNotEqual(vocab_size_2, 0)
            self.assertEqual(vocab_size, vocab_size_2)
            self.assertEqual(added_toks, len(new_toks))
            self.assertEqual(all_size_2, all_size + len(new_toks))

            tokens = tokenizer.encode("aaaaabbbbbb low cccccccccdddddddd l")
            self.assertGreaterEqual(len(tokens), 4)
            self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

            new_toks_2 = {'eos_token': ">>>>|||<||<<|<<",
                        'pad_token': "<<<<<|||>|>>>>|>"}
            added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
            vocab_size_3 = tokenizer.vocab_size
            all_size_3 = len(tokenizer)

            self.assertNotEqual(vocab_size_3, 0)
            self.assertEqual(vocab_size, vocab_size_3)
            self.assertEqual(added_toks_2, len(new_toks_2))
            self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

            tokens = tokenizer.encode(">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l")

            self.assertGreaterEqual(len(tokens), 6)
            self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[0], tokens[1])
            self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[-2], tokens[-3])
            self.assertEqual(tokens[0], tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
            self.assertEqual(tokens[-2], tokenizer.convert_tokens_to_ids(tokenizer.pad_token))


        def test_required_methods_tokenizer(self):
            tokenizer = self.get_tokenizer()
            input_text, output_text = self.get_input_output_texts()

            tokens = tokenizer.tokenize(input_text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids_2 = tokenizer.encode(input_text)
            self.assertListEqual(ids, ids_2)

            tokens_2 = tokenizer.convert_ids_to_tokens(ids)
            text_2 = tokenizer.decode(ids)

            self.assertEqual(text_2, output_text)

            self.assertNotEqual(len(tokens_2), 0)
            self.assertIsInstance(text_2, (str, unicode))


        def test_pretrained_model_lists(self):
            weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
            weights_lists_2 = []
            for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items():
                weights_lists_2.append(list(map_list.keys()))

            for weights_list_2 in weights_lists_2:
                self.assertListEqual(weights_list, weights_list_2)
