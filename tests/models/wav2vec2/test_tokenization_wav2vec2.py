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
    AddedToken,
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Tokenizer,
)
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES, Wav2Vec2CTCTokenizerOutput
from transformers.testing_utils import require_torch, slow

from ...test_tokenization_common import TokenizerTesterMixin


global_rng = random.Random()


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
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
        batch_tokens_2 = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

        self.assertEqual(batch_tokens, ["HELLO<unk>!?!?$$$", "BYE BYE<unk>$$$"])
        self.assertEqual(batch_tokens_2, ["HELO!?!?", "BYE BYE"])

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

        # Test 2-D numpy arrays are batched.
        speech_inputs = [floats_list((1, x))[0] for x in (800, 800, 800)]
        np_speech_inputs = np.asarray(speech_inputs)
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
        self.assertEqual(input_values_5.shape, (3, 1600))
        # padding should be 0.0
        self.assertTrue(abs(sum(np.asarray(input_values_5[0])[800:1200])) < 1e-3)

        input_values_6 = tokenizer(speech_inputs, pad_to_multiple_of=500).input_values
        input_values_7 = tokenizer(speech_inputs, padding="longest", pad_to_multiple_of=500).input_values
        input_values_8 = tokenizer(
            speech_inputs, padding="max_length", pad_to_multiple_of=500, max_length=2400
        ).input_values

        self.assertTrue(_input_values_are_equal(input_values_1, input_values_6))
        self.assertEqual(input_values_7.shape, (3, 1500))
        self.assertEqual(input_values_8.shape, (3, 2500))
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
            sorted(x.split(os.path.sep)[-1] for x in tokenizer_files),
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
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}, replace_additional_special_tokens=False
        )
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
        model_id = "facebook/wav2vec2-base-960h"
        config = Wav2Vec2Config.from_pretrained(model_id)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)

        # only "layer" feature extraction norm should make use of
        # attention_mask
        self.assertEqual(tokenizer.return_attention_mask, config.feat_extract_norm == "layer")


class Wav2Vec2CTCTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/wav2vec2-base-960h"
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

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77],
        ]
        sample_ids_2 = [
            [11, 5, 5, 5, 5, 5, 15, 15, 15, tokenizer.pad_token_id, 15, 8, 98],
            [24, 22, 5, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.word_delimiter_token_id, 24, 22, 5, 77, tokenizer.word_delimiter_token_id],
        ]
        # fmt: on

        batch_tokens = tokenizer.batch_decode(sample_ids)
        batch_tokens_2 = tokenizer.batch_decode(sample_ids_2)
        self.assertEqual(batch_tokens, batch_tokens_2)
        self.assertEqual(batch_tokens, ["HELLO<unk>", "BYE BYE<unk>"])

    def test_tokenizer_decode_added_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer.add_tokens(["!", "?", "<new_tokens>"])
        tokenizer.add_special_tokens({"cls_token": "$$$"})

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 8, 98, 32, 32, 33, tokenizer.word_delimiter_token_id, 32, 32, 33, 34, 34, 35, 35],
            [24, 22, 5, tokenizer.word_delimiter_token_id, 24, 22, 5, 77, tokenizer.pad_token_id, 34, 34, 35, 35],
        ]
        # fmt: on
        batch_tokens = tokenizer.batch_decode(sample_ids)
        batch_tokens_2 = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

        self.assertEqual(batch_tokens, ["HELLO<unk>!?!?<new_tokens>$$$", "BYE BYE<unk><new_tokens>$$$"])
        self.assertEqual(batch_tokens_2, ["HELO!?!?<new_tokens>", "BYE BYE<new_tokens>"])

    def test_special_characters_in_vocab(self):
        sent = "ʈʰ æ æ̃ ˧ kʰ"

        vocab_dict = {k: v for v, k in enumerate(set(sent.split()))}
        vocab_file = os.path.join(self.tmpdirname, "vocab_special.json")

        with open(vocab_file, "w") as f:
            json.dump(vocab_dict, f)

        tokenizer = Wav2Vec2CTCTokenizer(vocab_file)  # , unk_token="<unk>")

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

        tokenizer.save_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

    @staticmethod
    def get_from_offsets(offsets, key):
        retrieved_list = [d[key] for d in offsets]
        return retrieved_list

    def test_offsets(self):
        tokenizer = self.get_tokenizer()

        # fmt: off
        # HEEEEE||LLL<pad>LO<unk> => HE LLO<unk>
        # 1H + 5E + 2| + 3L + 1<pad> + 1L + 1O + 1<unk>
        sample_ids = [11, 5, 5, 5, 5, 5, 4, 4, 15, 15, 15, tokenizer.pad_token_id, 15, 8, 98]
        # fmt: on

        outputs_char = tokenizer.decode(sample_ids, output_char_offsets=True)
        # check Wav2Vec2CTCTokenizerOutput keys for char
        self.assertEqual(len(outputs_char.keys()), 2)
        self.assertTrue("text" in outputs_char)
        self.assertTrue("char_offsets" in outputs_char)
        self.assertTrue(isinstance(outputs_char, Wav2Vec2CTCTokenizerOutput))

        outputs_word = tokenizer.decode(sample_ids, output_word_offsets=True)
        # check Wav2Vec2CTCTokenizerOutput keys for word
        self.assertEqual(len(outputs_word.keys()), 2)
        self.assertTrue("text" in outputs_word)
        self.assertTrue("word_offsets" in outputs_word)
        self.assertTrue(isinstance(outputs_word, Wav2Vec2CTCTokenizerOutput))

        outputs = tokenizer.decode(sample_ids, output_char_offsets=True, output_word_offsets=True)
        # check Wav2Vec2CTCTokenizerOutput keys for both
        self.assertEqual(len(outputs.keys()), 3)
        self.assertTrue("text" in outputs)
        self.assertTrue("char_offsets" in outputs)
        self.assertTrue("word_offsets" in outputs)
        self.assertTrue(isinstance(outputs, Wav2Vec2CTCTokenizerOutput))

        # check that order of chars is correct and identical for both outputs
        self.assertEqual("".join(self.get_from_offsets(outputs["char_offsets"], "char")), outputs.text)
        self.assertEqual(
            self.get_from_offsets(outputs["char_offsets"], "char"), ["H", "E", " ", "L", "L", "O", "<unk>"]
        )
        self.assertListEqual(
            self.get_from_offsets(outputs["char_offsets"], "char"),
            self.get_from_offsets(outputs_char["char_offsets"], "char"),
        )

        # check that order of words is correct and identical to both outputs
        self.assertEqual(" ".join(self.get_from_offsets(outputs["word_offsets"], "word")), outputs.text)
        self.assertListEqual(self.get_from_offsets(outputs["word_offsets"], "word"), ["HE", "LLO<unk>"])
        self.assertListEqual(
            self.get_from_offsets(outputs["word_offsets"], "word"),
            self.get_from_offsets(outputs_word["word_offsets"], "word"),
        )

        # check that offsets are actually correct for char
        # 0 is H, 1 is E, 6 is | (" "),  8 is 1st L,  12 is 2nd L, 13 is O, 14 is <unk>
        self.assertListEqual(self.get_from_offsets(outputs["char_offsets"], "start_offset"), [0, 1, 6, 8, 12, 13, 14])
        # 1 is H, 6 is E, 8 is | (" "),  11 is 1st L (note due to <pad>
        # different begin of 2nd L), 13 is 2nd L, 14 is O, 15 is <unk>
        self.assertListEqual(self.get_from_offsets(outputs["char_offsets"], "end_offset"), [1, 6, 8, 11, 13, 14, 15])

        # check that offsets are actually correct for word
        # H is at 1st position of first word, first L is at 8th position of second word
        self.assertListEqual(self.get_from_offsets(outputs["word_offsets"], "start_offset"), [0, 8])
        # last E is at 6th position of first word, first L is at last (15th) position of second word
        self.assertListEqual(self.get_from_offsets(outputs["word_offsets"], "end_offset"), [6, 15])

    def test_word_offsets_from_char_offsets(self):
        tokenizer = self.get_tokenizer()

        char_offsets = [
            {"char": "H", "start_offset": 0, "end_offset": 1},
            {"char": "I", "start_offset": 1, "end_offset": 2},
            {"char": " ", "start_offset": 2, "end_offset": 3},
            {"char": "L", "start_offset": 3, "end_offset": 4},
            {"char": "I", "start_offset": 4, "end_offset": 5},
        ]
        word_offsets = tokenizer._get_word_offsets(char_offsets, tokenizer.replace_word_delimiter_char)

        self.assertEqual(
            word_offsets,
            [{"word": "HI", "start_offset": 0, "end_offset": 2}, {"word": "LI", "start_offset": 3, "end_offset": 5}],
        )

        # Double spaces don't get counted
        char_offsets = [
            {"char": " ", "start_offset": 0, "end_offset": 1},
            {"char": "H", "start_offset": 1, "end_offset": 2},
            {"char": "I", "start_offset": 2, "end_offset": 3},
            {"char": " ", "start_offset": 3, "end_offset": 4},
            {"char": " ", "start_offset": 4, "end_offset": 5},
            {"char": "L", "start_offset": 5, "end_offset": 6},
            {"char": "I", "start_offset": 6, "end_offset": 7},
            {"char": "I", "start_offset": 7, "end_offset": 8},
            {"char": " ", "start_offset": 8, "end_offset": 9},
            {"char": " ", "start_offset": 9, "end_offset": 10},
        ]
        word_offsets = tokenizer._get_word_offsets(char_offsets, tokenizer.replace_word_delimiter_char)
        self.assertEqual(
            word_offsets,
            [{"word": "HI", "start_offset": 1, "end_offset": 3}, {"word": "LII", "start_offset": 5, "end_offset": 8}],
        )

    def test_offsets_batch(self):
        tokenizer = self.get_tokenizer()

        def check_list_tuples_equal(outputs_batch, outputs_list):
            self.assertTrue(isinstance(outputs_batch, Wav2Vec2CTCTokenizerOutput))
            self.assertTrue(isinstance(outputs_list[0], Wav2Vec2CTCTokenizerOutput))

            # transform list to ModelOutput
            outputs_batch_2 = Wav2Vec2CTCTokenizerOutput({k: [d[k] for d in outputs_list] for k in outputs_list[0]})

            self.assertListEqual(outputs_batch["text"], outputs_batch_2["text"])

            def recursive_check(list_or_dict_1, list_or_dict_2):
                if isinstance(list_or_dict_1, list):
                    [recursive_check(l1, l2) for l1, l2 in zip(list_or_dict_1, list_or_dict_2)]
                self.assertEqual(list_or_dict_1, list_or_dict_2)

            if "char_offsets" in outputs_batch:
                recursive_check(outputs_batch["char_offsets"], outputs_batch_2["char_offsets"])

            if "word_offsets" in outputs_batch:
                recursive_check(outputs_batch["word_offsets"], outputs_batch_2["word_offsets"])

        # fmt: off
        sample_ids = [
            [11, 5, 15, tokenizer.pad_token_id, 15, 4, 8, 98, 32, 32, 32, 32, 4, 33, tokenizer.word_delimiter_token_id, 32, 32, 33, 34, 34],
            [24, 22, 5, tokenizer.word_delimiter_token_id, tokenizer.word_delimiter_token_id, 24, 22, 22, 22, 4, 5, 77, tokenizer.pad_token_id, 22, 22, 4, 34, 34, 34, 34],
        ]
        # fmt: on

        # We assume that `decode` works as expected. All we will check now is
        # the output type is correct and the output is identical to `decode`

        # char
        outputs_char_batch = tokenizer.batch_decode(sample_ids, output_char_offsets=True)
        outputs_char = [tokenizer.decode(ids, output_char_offsets=True) for ids in sample_ids]
        check_list_tuples_equal(outputs_char_batch, outputs_char)

        # word
        outputs_word_batch = tokenizer.batch_decode(sample_ids, output_word_offsets=True)
        outputs_word = [tokenizer.decode(ids, output_word_offsets=True) for ids in sample_ids]
        check_list_tuples_equal(outputs_word_batch, outputs_word)

        # both
        outputs_batch = tokenizer.batch_decode(sample_ids, output_char_offsets=True, output_word_offsets=True)
        outputs = [tokenizer.decode(ids, output_word_offsets=True, output_char_offsets=True) for ids in sample_ids]
        check_list_tuples_equal(outputs_batch, outputs)

    def test_offsets_integration(self):
        tokenizer = self.tokenizer_class.from_pretrained("facebook/wav2vec2-base-960h")
        # pred_ids correspond to the following code
        # ```
        #        from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
        #        from datasets import load_dataset
        #        import datasets
        #        import torch
        #        model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        #        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        #
        #        ds = load_dataset("common_voice", "en", split="train", streaming=True)
        #        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        #        ds_iter = iter(ds)
        #        sample = next(ds_iter)
        #
        #        input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
        #        logits = model(input_values).logits
        #        pred_ids = torch.argmax(logits, axis=-1).cpu().tolist()
        # ```
        # fmt: off
        pred_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 11, 0, 0, 0, 22, 0, 0, 4, 4, 4, 14, 0, 0, 0, 0, 0, 8, 8, 0, 5, 5, 0, 12, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 10, 0, 0, 0, 15, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0, 0, 7, 0, 9, 0, 0, 14, 0, 0, 0, 13, 0, 7, 0, 0, 4, 4, 0, 15, 8, 8, 0, 0, 8, 0, 26, 0, 0, 4, 4, 0, 0, 15, 0, 0, 0, 0, 0, 0, 10, 0, 26, 5, 5, 0, 4, 4, 0, 0, 12, 11, 0, 0, 5, 4, 4, 4, 0, 18, 0, 0, 0, 7, 9, 9, 0, 6, 0, 12, 12, 4, 4, 0, 6, 0, 0, 8, 0, 4, 4, 4, 0, 19, 0, 0, 8, 9, 9, 0, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 0, 16, 16, 0, 0, 17, 5, 5, 5, 0, 4, 4, 4, 0, 0, 29, 29, 0, 0, 0, 0, 8, 11, 0, 9, 9, 0, 0, 0, 4, 4, 0, 12, 12, 0, 0, 0, 9, 0, 0, 0, 0, 0, 8, 18, 0, 0, 0, 4, 4, 0, 0, 8, 9, 0, 4, 4, 0, 6, 11, 5, 0, 4, 4, 0, 13, 13, 0, 0, 0, 10, 0, 0, 25, 0, 0, 6, 0, 4, 4, 0, 0, 0, 0, 7, 0, 0, 23, 0, 0, 4, 4, 0, 0, 0, 6, 11, 0, 5, 4, 4, 18, 0, 0, 0, 0, 0, 0, 7, 15, 0, 0, 0, 15, 15, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # wav2vec2-base downsamples input audio by a factor of 320
        # sampling rate for wav2vec2-base is 16_000
        time_offset_wav2vec2_base = 320 / 16_000

        expected_char_time_stamps_text = ['W', 'H', 'Y', ' ', 'D', 'O', 'E', 'S', ' ', 'M', 'I', 'L', 'I', 'S', 'A', 'N', 'D', 'R', 'A', ' ', 'L', 'O', 'O', 'K', ' ', 'L', 'I', 'K', 'E', ' ', 'S', 'H', 'E', ' ', 'W', 'A', 'N', 'T', 'S', ' ', 'T', 'O', ' ', 'C', 'O', 'N', 'S', 'U', 'M', 'E', ' ', 'J', 'O', 'H', 'N', ' ', 'S', 'N', 'O', 'W', ' ', 'O', 'N', ' ', 'T', 'H', 'E', ' ', 'R', 'I', 'V', 'T', ' ', 'A', 'P', ' ', 'T', 'H', 'E', ' ', 'W', 'A', 'L', 'L', ' ']
        expected_char_time_stamps_start = [1.42, 1.44, 1.52, 1.58, 1.64, 1.76, 1.82, 1.88, 1.92, 2.26, 2.32, 2.4, 2.46, 2.54, 2.66, 2.7, 2.76, 2.84, 2.88, 2.94, 3.0, 3.02, 3.1, 3.14, 3.2, 3.28, 3.42, 3.46, 3.48, 3.54, 3.62, 3.64, 3.7, 3.72, 3.8, 3.88, 3.9, 3.96, 4.0, 4.04, 4.1, 4.16, 4.2, 4.28, 4.34, 4.36, 4.48, 4.66, 4.74, 4.76, 4.84, 4.94, 5.06, 5.08, 5.12, 5.22, 5.28, 5.38, 5.5, 5.52, 5.6, 5.68, 5.7, 5.74, 5.8, 5.82, 5.84, 5.88, 5.94, 6.04, 6.1, 6.16, 6.2, 6.32, 6.38, 6.44, 6.54, 6.56, 6.6, 6.62, 6.66, 6.8, 6.82, 6.9, 6.96]
        expected_char_time_stamps_end = [1.44, 1.46, 1.54, 1.64, 1.66, 1.8, 1.86, 1.9, 2.06, 2.28, 2.34, 2.42, 2.48, 2.56, 2.68, 2.72, 2.78, 2.86, 2.9, 2.98, 3.02, 3.06, 3.12, 3.16, 3.24, 3.3, 3.44, 3.48, 3.52, 3.58, 3.64, 3.66, 3.72, 3.78, 3.82, 3.9, 3.94, 3.98, 4.04, 4.08, 4.12, 4.18, 4.26, 4.3, 4.36, 4.4, 4.52, 4.7, 4.76, 4.82, 4.9, 4.98, 5.08, 5.1, 5.16, 5.26, 5.32, 5.4, 5.52, 5.54, 5.64, 5.7, 5.72, 5.78, 5.82, 5.84, 5.86, 5.92, 5.98, 6.06, 6.12, 6.18, 6.24, 6.34, 6.4, 6.48, 6.56, 6.58, 6.62, 6.66, 6.68, 6.82, 6.84, 6.94, 7.02]

        expected_word_time_stamps_text = ['WHY', 'DOES', 'MILISANDRA', 'LOOK', 'LIKE', 'SHE', 'WANTS', 'TO', 'CONSUME', 'JOHN', 'SNOW', 'ON', 'THE', 'RIVT', 'AP', 'THE', 'WALL']
        expected_word_time_stamps_start = [1.42, 1.64, 2.26, 3.0, 3.28, 3.62, 3.8, 4.1, 4.28, 4.94, 5.28, 5.68, 5.8, 5.94, 6.32, 6.54, 6.66]
        expected_word_time_stamps_end = [1.54, 1.9, 2.9, 3.16, 3.52, 3.72, 4.04, 4.18, 4.82, 5.16, 5.54, 5.72, 5.86, 6.18, 6.4, 6.62, 6.94]
        # fmt: on

        output = tokenizer.batch_decode(pred_ids, output_char_offsets=True, output_word_offsets=True)

        char_offsets_text = self.get_from_offsets(output["char_offsets"][0], "char")
        char_offsets_start = self.get_from_offsets(output["char_offsets"][0], "start_offset")
        char_offsets_end = self.get_from_offsets(output["char_offsets"][0], "end_offset")

        word_offsets_text = self.get_from_offsets(output["word_offsets"][0], "word")
        word_offsets_start = self.get_from_offsets(output["word_offsets"][0], "start_offset")
        word_offsets_end = self.get_from_offsets(output["word_offsets"][0], "end_offset")

        # let's transform offsets to time stamps in seconds
        char_time_stamps_start = [round(c * time_offset_wav2vec2_base, 2) for c in char_offsets_start]
        char_time_stamps_end = [round(c * time_offset_wav2vec2_base, 2) for c in char_offsets_end]

        word_time_stamps_start = [round(w * time_offset_wav2vec2_base, 2) for w in word_offsets_start]
        word_time_stamps_end = [round(w * time_offset_wav2vec2_base, 2) for w in word_offsets_end]

        # NOTE: you can verify the above results by checking out the dataset viewer
        # on https://huggingface.co/datasets/common_voice/viewer/en/train and
        # downloading / playing the sample `common_voice_en_100038.mp3`. As
        # you can hear the time-stamps match more or less

        self.assertListEqual(expected_char_time_stamps_text, char_offsets_text)
        self.assertListEqual(expected_char_time_stamps_start, char_time_stamps_start)
        self.assertListEqual(expected_char_time_stamps_end, char_time_stamps_end)

        self.assertListEqual(expected_word_time_stamps_text, word_offsets_text)
        self.assertListEqual(expected_word_time_stamps_start, word_time_stamps_start)
        self.assertListEqual(expected_word_time_stamps_end, word_time_stamps_end)

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

                new_toks_2 = {
                    "eos_token": AddedToken(">>>>|||<||<<|<<", lstrip=False, rstrip=False),
                    "pad_token": AddedToken("<<<<<|||>|>>>>|>", rstrip=False, lstrip=False),
                }
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

    @unittest.skip("The tokenizer shouldn't be used to encode input IDs (except for labels), only to decode.")
    def test_tf_encode_plus_sent_to_model(self):
        pass

    @unittest.skip("The tokenizer shouldn't be used to encode input IDs (except for labels), only to decode.")
    def test_torch_encode_plus_sent_to_model(self):
        pass

    def test_convert_tokens_to_string_format(self):
        # The default common tokenizer tests assumes that the output of `convert_tokens_to_string` is a string which
        # is not the case for Wav2vec2.
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokens = ["T", "H", "I", "S", "|", "I", "S", "|", "A", "|", "T", "E", "X", "T"]
                output = tokenizer.convert_tokens_to_string(tokens)

                self.assertIsInstance(output["text"], str)

    def test_nested_vocab(self):
        eng_vocab = {"a": 7, "b": 8}
        spa_vocab = {"a": 23, "c": 88}
        ita_vocab = {"a": 6, "d": 9}

        nested_vocab = {"eng": eng_vocab, "spa": spa_vocab, "ita": ita_vocab}

        def check_tokenizer(tokenizer, check_ita_first=False):
            if check_ita_first:
                self.assertEqual(tokenizer.decode([6, 9, 9]), "ad")
                self.assertEqual(tokenizer.encoder, ita_vocab)
                tokenizer.set_target_lang("eng")

            self.assertEqual(tokenizer.encoder, eng_vocab)
            self.assertEqual(tokenizer.decode([7, 8, 7]), "aba")

            tokenizer.set_target_lang("spa")
            self.assertEqual(tokenizer.decode([23, 88, 23]), "aca")
            self.assertEqual(tokenizer.encoder, spa_vocab)

            tokenizer.set_target_lang("eng")
            self.assertEqual(tokenizer.encoder, eng_vocab)
            self.assertEqual(tokenizer.decode([7, 7, 8]), "ab")

            tokenizer.set_target_lang("ita")
            self.assertEqual(tokenizer.decode([6, 9, 9]), "ad")
            self.assertEqual(tokenizer.encoder, ita_vocab)

        with tempfile.TemporaryDirectory() as tempdir:
            tempfile_path = os.path.join(tempdir, "vocab.json")
            with open(tempfile_path, "w") as temp_file:
                json.dump(nested_vocab, temp_file)

            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tempdir, target_lang="eng")

        check_tokenizer(tokenizer)

        with tempfile.TemporaryDirectory() as tempdir:
            # should have saved target lang as "ita" since it was last one
            tokenizer.save_pretrained(tempdir)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tempdir)

            self.assertEqual(tokenizer.target_lang, "ita")
            check_tokenizer(tokenizer, check_ita_first=True)
