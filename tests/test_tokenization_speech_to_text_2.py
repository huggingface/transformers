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

import json
import os
import tempfile
import unittest

from transformers.models.speech_to_text_2 import Speech2Text2Tokenizer
from transformers.models.speech_to_text_2.tokenization_speech_to_text_2 import VOCAB_FILES_NAMES
from transformers.testing_utils import is_pt_tf_cross_test

from .test_tokenization_common import TokenizerTesterMixin


class SpeechToTextTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Speech2Text2Tokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = "<s> <pad> </s> <unk> here@@ a couple of@@ words for the vocab".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "vocab")
        self.assertEqual(len(vocab_keys), 12)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 12)

    def test_tokenizer_decode(self):
        tokenizer = Speech2Text2Tokenizer.from_pretrained(self.tmpdirname)

        # make sure @@ is correctly concatenated
        token_ids = [4, 6, 8, 7, 10]  # ["here@@", "couple", "words", "of@@", "the"]
        output_string = tokenizer.decode(token_ids)

        self.assertTrue(output_string == "herecouple words ofthe")

    # currently tokenizer cannot do encoding, but just decoding
    def test_add_special_tokens(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_add_tokens_tokenizer(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_added_tokens_do_lower_case(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_batch_encode_plus_batch_sequence_length(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_batch_encode_plus_overflowing_tokens(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_batch_encode_plus_padding(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_call(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_encode_plus_with_padding(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_internal_consistency(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_maximum_encoding_length_pair_input(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_maximum_encoding_length_single_input(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_number_of_added_tokens(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_padding_to_max_length(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_padding_to_multiple_of(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_pickle_tokenizer(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_prepare_for_model(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_pretokenized_inputs(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_right_and_left_padding(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_save_and_load_tokenizer(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_special_tokens_mask(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_special_tokens_mask_input_pairs(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_token_type_ids(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    def test_added_token_are_matched_longest_first(self):
        pass

    # currently tokenizer cannot do encoding, but just decoding
    @is_pt_tf_cross_test
    def test_batch_encode_plus_tensors(self):
        pass
