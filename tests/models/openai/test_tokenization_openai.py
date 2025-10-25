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


import json
import os
import unittest

from transformers import OpenAIGPTTokenizer, OpenAIGPTTokenizerFast
from transformers.models.openai.tokenization_openai import VOCAB_FILES_NAMES
from transformers.testing_utils import require_ftfy, require_spacy, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class OpenAIGPTTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openai-community/openai-gpt"
    """Tests OpenAIGPTTokenizer that uses BERT BasicTokenizer."""

    tokenizer_class = OpenAIGPTTokenizer
    rust_tokenizer_class = OpenAIGPTTokenizerFast
    test_rust_tokenizer = True
    test_seq2seq = False
    
    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['this</w>', 'is</w>', 'a</w>', 'test</w>', '<unk>', 'i</w>', 'was</w>', 'born</w>', 'in</w>', '9', '2000</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'false</w>', '.</w>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', 'hi</w>', 'hello</w>', 'hi</w>', 'hello</w>', 'hello</w>', '<</w>', 's</w>', '></w>', 'hi</w>', '<</w>', 's</w>', '></w>', 'there</w>', 'the</w>', 'following</w>', 'string</w>', 'should</w>', 'be</w>', 'properly</w>', 'en', 'coded</w>', ':</w>', 'hello</w>', '.</w>', 'but</w>', 'ird</w>', 'and</w>', '<unk>', 'ird</w>', '<unk>', 'hey</w>', 'how</w>', 'are</w>', 'you</w>', 'doing</w>']
    integration_expected_token_ids = [616, 544, 246, 2345, 0, 249, 509, 3105, 500, 53, 28654, 240, 488, 616, 544, 6843, 239, 0, 0, 0, 0, 0, 0, 3569, 3570, 3569, 3570, 3570, 295, 252, 290, 3569, 295, 252, 290, 655, 481, 2890, 6422, 994, 580, 6506, 496, 20925, 271, 3570, 239, 568, 13926, 488, 0, 13926, 0, 2229, 718, 640, 512, 1273]
    integration_expected_decoded_text = 'this is a test <unk>i was born in 92000 , and this is false . <unk><unk><unk><unk><unk><unk>hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello . but ird and <unk>ird <unk>hey how are you doing'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "w</w>",
            "r</w>",
            "t</w>",
            "lo",
            "low",
            "er</w>",
            "low</w>",
            "lowest</w>",
            "newer</w>",
            "wider</w>",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w", "e r</w>", ""]

        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        cls.merges_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(cls.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(cls.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def get_input_output_texts(self, tokenizer):
        return "lower newer", "lower newer"

    def test_full_tokenizer(self):
        tokenizer = OpenAIGPTTokenizer(self.vocab_file, self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_padding(self, max_length=15):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_rust_tokenizer(pretrained_name, **kwargs)

                # Simple input
                s = "This is a simple input"
                s2 = ["This is a simple input 1", "This is a simple input 2"]
                p = ("This is a simple input", "This is a pair")
                p2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    @unittest.skip(reason="tokenizer has no padding token")
    def test_padding_different_model_input_name(self):
        pass


@require_ftfy
@require_spacy
@require_tokenizers
class OpenAIGPTTokenizationTestWithSpacy(OpenAIGPTTokenizationTest):
    """Tests OpenAIGPTTokenizer that uses SpaCy and ftfy."""

    pass
