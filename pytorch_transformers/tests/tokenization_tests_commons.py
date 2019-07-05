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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from io import open

if sys.version_info[0] == 3:
    unicode = str

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def create_and_check_save_and_load_tokenizer(tester, tokenizer_class, *inputs, **kwargs):
    tokenizer = tokenizer_class(*inputs, **kwargs)

    before_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running")

    vocab_path="/tmp/"
    output_files = tokenizer.save_vocabulary(vocab_path=vocab_path)
    tokenizer = tokenizer.from_pretrained(vocab_path)

    for f in output_files:
        os.remove(f)

    after_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running")
    tester.assertListEqual(before_tokens, after_tokens)

def create_and_check_pickle_tokenizer(tester, tokenizer_class, *inputs, **kwargs):
    tokenizer = tokenizer_class(*inputs, **kwargs)

    text = "Munich and Berlin are nice cities"
    filename = u"/tmp/tokenizer.bin"

    subwords = tokenizer.tokenize(text)

    pickle.dump(tokenizer, open(filename, "wb"))

    tokenizer_new = pickle.load(open(filename, "rb"))
    subwords_loaded = tokenizer_new.tokenize(text)

    tester.assertListEqual(subwords, subwords_loaded)


def create_and_check_required_methods_tokenizer(tester, tokenizer_class, *inputs, **kwargs):
    tokenizer = tokenizer_class(*inputs, **kwargs)

    text = u"He is very happy, UNwant\u00E9d,running"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids_2 = tokenizer.encode(text)
    tester.assertListEqual(ids, ids_2)

    tokens_2 = tokenizer.convert_ids_to_tokens(ids)
    text_2 = tokenizer.decode(ids)

    tester.assertNotEqual(len(tokens_2), 0)
    tester.assertIsInstance(text_2, (str, unicode))

def create_and_check_tokenizer_commons(tester, tokenizer_class, *inputs, **kwargs):
    create_and_check_required_methods_tokenizer(tester, tokenizer_class, *inputs, **kwargs)
    create_and_check_save_and_load_tokenizer(tester, tokenizer_class, *inputs, **kwargs)
    create_and_check_pickle_tokenizer(tester, tokenizer_class, *inputs, **kwargs)
