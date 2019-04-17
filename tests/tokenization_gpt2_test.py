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
import json
import shutil
import pytest

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


class GPT2TokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n",
                 "lo", "low", "er",
                 "low", "lowest", "newer", "wider"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r", ""]
        with open("/tmp/openai_tokenizer_vocab_test.json", "w") as fp:
            fp.write(json.dumps(vocab_tokens))
            vocab_file = fp.name
        with open("/tmp/openai_tokenizer_merges_test.txt", "w") as fp:
            fp.write("\n".join(merges))
            merges_file = fp.name

        tokenizer = GPT2Tokenizer(vocab_file, merges_file, special_tokens=["<unk>", "<pad>"])
        os.remove(vocab_file)
        os.remove(merges_file)

        text = "lower"
        bpe_tokens = ["low", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [13, 12, 16]
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

        vocab_file, merges_file, special_tokens_file = tokenizer.save_vocabulary(vocab_path="/tmp/")
        tokenizer_2 = GPT2Tokenizer.from_pretrained("/tmp/")
        os.remove(vocab_file)
        os.remove(merges_file)
        os.remove(special_tokens_file)

        self.assertListEqual(
            [tokenizer.encoder, tokenizer.decoder, tokenizer.bpe_ranks,
             tokenizer.special_tokens, tokenizer.special_tokens_decoder],
            [tokenizer_2.encoder, tokenizer_2.decoder, tokenizer_2.bpe_ranks,
             tokenizer_2.special_tokens, tokenizer_2.special_tokens_decoder])

    # @pytest.mark.slow
    def test_tokenizer_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(PRETRAINED_VOCAB_ARCHIVE_MAP.keys())[:1]:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(tokenizer)

if __name__ == '__main__':
    unittest.main()
