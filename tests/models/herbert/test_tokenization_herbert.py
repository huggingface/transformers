# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Allegro.pl and The HuggingFace Inc. team.
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

from transformers import HerbertTokenizer, HerbertTokenizerFast
from transformers.models.herbert.tokenization_herbert import VOCAB_FILES_NAMES
from transformers.testing_utils import get_tests_dir, require_sacremoses, require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_sacremoses
@require_tokenizers
class HerbertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "allegro/herbert-base-cased"
    tokenizer_class = HerbertTokenizer
    rust_tokenizer_class = HerbertTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        # Use a simpler test file without japanese/chinese characters
        with open(f"{get_tests_dir()}/fixtures/sample_text_no_unicode.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        vocab = [
            "<s>",
            "</s>",
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
            ",</w>",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(vocab_file=self.vocab_file, merges_file=self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [16, 17, 23]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "lower,newer"

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("allegro/herbert-base-cased")

        text = tokenizer.encode("konstruowanie sekwencji", add_special_tokens=False)
        text_2 = tokenizer.encode("konstruowanie wielu sekwencji", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [0] + text + [2]
        assert encoded_pair == [0] + text + [2] + text_2 + [2]

    @unittest.skip(
        "Test passes if run individually but not with the full tests (internal state of the tokenizer is modified). Will fix later"
    )
    def test_training_new_tokenizer_with_special_tokens_change(self):
        pass

    @unittest.skip(
        "Test passes if run individually but not with the full tests (internal state of the tokenizer is modified). Will fix later"
    )
    def test_training_new_tokenizer(self):
        pass
