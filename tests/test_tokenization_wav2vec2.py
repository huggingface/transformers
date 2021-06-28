# coding=utf-8
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
"""Tests for the Wav2Vec2 tokenizer."""
import inspect
import json
import os
import random
import shutil
import tempfile
import unittest

import numpy as np

from transformers import (
    WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Tokenizer,
)
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class Wav2Vec2TokenizerTest(unittest.TestCase):
    tokenizer_class = Wav2Vec2Tokenizer

    def setUp(self):
        super().setUp()

        vocab = "<pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Wav2Vec2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizer_decode(self):
        # TODO(PVP) - change to facebook
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77],
        ]
        tokens = tokenizer.decode(sample_ids[0])
        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["HELLO<unk>", "BYE BYE<unk>"])

    def test_tokenizer_decode_special(self):
        # TODO(PVP) - change to facebook
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77],
        ]
        sample_ids_2 = [
            [11, 5, 5, 5, 5, 5, 15, 15, 15, tokenizer.pad_token_id, 15, 8, 98],
            [
                24,
                22,
                5,
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
                tokenizer.word_delimiter_token_id,
                24,
                22,
                5,
                77,
                tokenizer.word_delimiter_token_id,
            ],
        ]

        batch_tokens = tokenizer.batch_decode(sample_ids)
        batch_tokens_2 = tokenizer.batch_decode(sample_ids_2)
        self.assertEqual(batch_tokens, batch_tokens_2)
        self.assertEqual(batch_tokens, ["HELLO<unk>", "BYE BYE<unk>"])

    def test_tokenizer_decode_added_tokens(self):
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer.add_tokens(["!", "?"])
        tokenizer.add_special_tokens({"cls_token": "$$$"})

        sample_ids = [
            [
                11,
                5,
                15,
                tokenizer.pad_token_id,
                15,
                8,
                98,
                32,
                32,
                33,
                tokenizer.word_delimiter_token_id,
                32,
                32,
                33,
                34,
                34,
            ],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77, tokenizer.pad_token_id, 34, 34],
        ]
        batch_tokens = tokenizer.batch_decode(sample_ids)

        self.assertEqual(batch_tokens, ["HELLO<unk>!?!?$$$", "BYE BYE<unk>$$$"])

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizer = self.get_tokenizer()
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test not batched input
        encoded_sequences_1 = tokenizer(speech_inputs[0], return_tensors="np").input_values
        encoded_sequences_2 = tokenizer(np_speech_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = tokenizer(speech_inputs, return_tensors="np").input_values
        encoded_sequences_2 = tokenizer(np_speech_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_padding(self, max_length=50):
        def _input_values_have_equal_length(input_values):
            length = len(input_values[0])
            for input_values_slice in input_values[1:]:
                if len(input_values_slice) != length:
                    return False
            return True

        def _input_values_are_equal(input_values_1, input_values_2):
            if len(input_values_1) != len(input_values_2):
                return False

            for input_values_slice_1, input_values_slice_2 in zip(input_values_1, input_values_2):
                if not np.allclose(np.asarray(input_values_slice_1), np.asarray(input_values_slice_2), atol=1e-3):
                    return False
            return True

        tokenizer = self.get_tokenizer()
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        input_values_1 = tokenizer(speech_inputs).input_values
        input_values_2 = tokenizer(speech_inputs, padding="longest").input_values
        input_values_3 = tokenizer(speech_inputs, padding="longest", max_length=1600).input_values

        self.assertFalse(_input_values_have_equal_length(input_values_1))
        self.assertTrue(_input_values_have_equal_length(input_values_2))
        self.assertTrue(_input_values_have_equal_length(input_values_3))
        self.assertTrue(_input_values_are_equal(input_values_2, input_values_3))
        self.assertTrue(len(input_values_1[0]) == 800)
        self.assertTrue(len(input_values_2[0]) == 1200)
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_values_2[0])[800:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_2[1])[1000:])) < 1e-3)

        input_values_4 = tokenizer(speech_inputs, padding="max_length").input_values
        input_values_5 = tokenizer(speech_inputs, padding="max_length", max_length=1600).input_values

        self.assertTrue(_input_values_are_equal(input_values_1, input_values_4))
        self.assertTrue(input_values_5.shape, (3, 1600))
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_values_5[0])[800:1200])) < 1e-3)

        input_values_6 = tokenizer(speech_inputs, pad_to_multiple_of=500).input_values
        input_values_7 = tokenizer(speech_inputs, padding="longest", pad_to_multiple_of=500).input_values
        input_values_8 = tokenizer(
            speech_inputs, padding="max_length", pad_to_multiple_of=500, max_length=2400
        ).input_values

        self.assertTrue(_input_values_are_equal(input_values_1, input_values_6))
        self.assertTrue(input_values_7.shape, (3, 1500))
        self.assertTrue(input_values_8.shape, (3, 2500))
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_values_7[0])[800:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_7[1])[1000:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_7[2])[1200:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_8[0])[800:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_8[1])[1000:])) < 1e-3)
        self.assertTrue(abs(sum(np.asarray(input_values_8[2])[1200:])) < 1e-3)

    def test_save_pretrained(self):
        pretrained_name = list(self.tokenizer_class.pretrained_vocab_files_map["vocab_file"].keys())[0]
        tokenizer = self.tokenizer_class.from_pretrained(pretrained_name)
        tmpdirname2 = tempfile.mkdtemp()

        tokenizer_files = tokenizer.save_pretrained(tmpdirname2)
        self.assertSequenceEqual(
            sorted(tuple(VOCAB_FILES_NAMES.values()) + ("special_tokens_map.json", "added_tokens.json")),
            sorted(tuple(x.split(os.path.sep)[-1] for x in tokenizer_files)),
        )

        # Checks everything loads correctly in the same way
        tokenizer_p = self.tokenizer_class.from_pretrained(tmpdirname2)

        # Check special tokens are set accordingly on Rust and Python
        for key in tokenizer.special_tokens_map:
            self.assertTrue(key in tokenizer_p.special_tokens_map)

        shutil.rmtree(tmpdirname2)

    def test_get_vocab(self):
        tokenizer = self.get_tokenizer()
        vocab_dict = tokenizer.get_vocab()
        self.assertIsInstance(vocab_dict, dict)
        self.assertGreaterEqual(len(tokenizer), len(vocab_dict))

        vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
        self.assertEqual(len(vocab), len(tokenizer))

        tokenizer.add_tokens(["asdfasdfasdfasdf"])
        vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
        self.assertEqual(len(vocab), len(tokenizer))

    def test_save_and_load_tokenizer(self):
        tokenizer = self.get_tokenizer()
        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        sample_ids = [0, 1, 4, 8, 9, 0, 12]
        before_tokens = tokenizer.decode(sample_ids)
        before_vocab = tokenizer.get_vocab()
        tokenizer.save_pretrained(tmpdirname)

        after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
        after_tokens = after_tokenizer.decode(sample_ids)
        after_vocab = after_tokenizer.get_vocab()

        self.assertEqual(before_tokens, after_tokens)
        self.assertDictEqual(before_vocab, after_vocab)

        shutil.rmtree(tmpdirname)

        tokenizer = self.get_tokenizer()

        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        before_len = len(tokenizer)
        sample_ids = [0, 1, 4, 8, 9, 0, 12, before_len, before_len + 1, before_len + 2]
        tokenizer.add_tokens(["?", "!"])
        additional_special_tokens = tokenizer.additional_special_tokens
        additional_special_tokens.append("&")
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        before_tokens = tokenizer.decode(sample_ids)
        before_vocab = tokenizer.get_vocab()
        tokenizer.save_pretrained(tmpdirname)

        after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
        after_tokens = after_tokenizer.decode(sample_ids)
        after_vocab = after_tokenizer.get_vocab()

        self.assertEqual(before_tokens, after_tokens)
        self.assertDictEqual(before_vocab, after_vocab)

        self.assertTrue(len(tokenizer), before_len + 3)
        self.assertTrue(len(tokenizer), len(after_tokenizer))
        shutil.rmtree(tmpdirname)

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_zero_mean_unit_variance_normalization(self):
        tokenizer = self.get_tokenizer(do_normalize=True)
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        processed = tokenizer(speech_inputs, padding="longest")
        input_values = processed.input_values

        def _check_zero_mean_unit_variance(input_vector):
            self.assertTrue(np.abs(np.mean(input_vector)) < 1e-3)
            self.assertTrue(np.abs(np.var(input_vector) - 1) < 1e-3)

        _check_zero_mean_unit_variance(input_values[0, :800])
        _check_zero_mean_unit_variance(input_values[1, :1000])
        _check_zero_mean_unit_variance(input_values[2])

    def test_return_attention_mask(self):
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        # default case -> no attention_mask is returned
        tokenizer = self.get_tokenizer()
        processed = tokenizer(speech_inputs)
        self.assertNotIn("attention_mask", processed)

        # wav2vec2-lv60 -> return attention_mask
        tokenizer = self.get_tokenizer(return_attention_mask=True)
        processed = tokenizer(speech_inputs, padding="longest")

        self.assertIn("attention_mask", processed)
        self.assertListEqual(list(processed.attention_mask.shape), list(processed.input_values.shape))
        self.assertListEqual(processed.attention_mask.sum(-1).tolist(), [800, 1000, 1200])

    @slow
    @require_torch
    def test_pretrained_checkpoints_are_set_correctly(self):
        # this test makes sure that models that are using
        # group norm don't have their tokenizer return the
        # attention_mask
        for model_id in WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST:
            config = Wav2Vec2Config.from_pretrained(model_id)
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)

            # only "layer" feature extraction norm should make use of
            # attention_mask
            self.assertEqual(tokenizer.return_attention_mask, config.feat_extract_norm == "layer")


class Wav2Vec2CTCTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Wav2Vec2CTCTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        vocab = "<pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.special_tokens_map = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Wav2Vec2CTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizer_add_token_chars(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")

        # check adding a single token
        tokenizer.add_tokens("x")
        token_ids = tokenizer("C x A").input_ids
        self.assertEqual(token_ids, [19, 4, 32, 4, 7])

        tokenizer.add_tokens(["a", "b", "c"])
        token_ids = tokenizer("C a A c").input_ids
        self.assertEqual(token_ids, [19, 4, 33, 4, 7, 4, 35])

        tokenizer.add_tokens(["a", "b", "c"])
        token_ids = tokenizer("CaA c").input_ids
        self.assertEqual(token_ids, [19, 33, 7, 4, 35])

    def test_tokenizer_add_token_words(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")

        # check adding a single token
        tokenizer.add_tokens("xxx")
        token_ids = tokenizer("C xxx A B").input_ids
        self.assertEqual(token_ids, [19, 4, 32, 4, 7, 4, 24])

        tokenizer.add_tokens(["aaa", "bbb", "ccc"])
        token_ids = tokenizer("C aaa A ccc B B").input_ids
        self.assertEqual(token_ids, [19, 4, 33, 4, 7, 4, 35, 4, 24, 4, 24])

        tokenizer.add_tokens(["aaa", "bbb", "ccc"])
        token_ids = tokenizer("CaaaA ccc B B").input_ids
        self.assertEqual(token_ids, [19, 33, 7, 4, 35, 4, 24, 4, 24])

    def test_tokenizer_decode(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")

        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77],
        ]
        tokens = tokenizer.decode(sample_ids[0])
        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["HELLO<unk>", "BYE BYE<unk>"])

    def test_tokenizer_decode_special(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")

        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77],
        ]
        sample_ids_2 = [
            [11, 5, 5, 5, 5, 5, 15, 15, 15, tokenizer.pad_token_id, 15, 8, 98],
            [
                24,
                22,
                5,
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
                tokenizer.word_delimiter_token_id,
                24,
                22,
                5,
                77,
                tokenizer.word_delimiter_token_id,
            ],
        ]

        batch_tokens = tokenizer.batch_decode(sample_ids)
        batch_tokens_2 = tokenizer.batch_decode(sample_ids_2)
        self.assertEqual(batch_tokens, batch_tokens_2)
        self.assertEqual(batch_tokens, ["HELLO<unk>", "BYE BYE<unk>"])

    def test_tokenizer_decode_added_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer.add_tokens(["!", "?"])
        tokenizer.add_special_tokens({"cls_token": "$$$"})

        sample_ids = [
            [
                11,
                5,
                15,
                tokenizer.pad_token_id,
                15,
                8,
                98,
                32,
                32,
                33,
                tokenizer.word_delimiter_token_id,
                32,
                32,
                33,
                34,
                34,
            ],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77, tokenizer.pad_token_id, 34, 34],
        ]
        batch_tokens = tokenizer.batch_decode(sample_ids)

        self.assertEqual(batch_tokens, ["HELLO<unk>!?!?$$$", "BYE BYE<unk>$$$"])

    def test_special_characters_in_vocab(self):
        sent = "ʈʰ æ æ̃ ˧ kʰ"

        vocab_dict = {k: v for v, k in enumerate({phoneme for phoneme in sent.split()})}
        vocab_file = os.path.join(self.tmpdirname, "vocab_special.json")

        with open(vocab_file, "w") as f:
            json.dump(vocab_dict, f)

        tokenizer = Wav2Vec2CTCTokenizer(vocab_file)

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

        tokenizer.save_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

    def test_pretrained_model_lists(self):
        # Wav2Vec2Model has no max model length => no testing
        pass

    # overwrite from test_tokenization_common
    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokens[-4])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-3], tokenizer.pad_token_id)
