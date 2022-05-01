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
"""Tests for the MCTC tokenizer."""
import inspect
import json
import os
import random
import shutil
import tempfile
import unittest

import numpy as np

from transformers import MCTC_PRETRAINED_MODEL_ARCHIVE_LIST, MCTCConfig, MCTCTokenizer
from transformers.models.mctc.tokenization_mctc import VOCAB_FILES_NAMES, MCTCTokenizerOutput
from transformers.testing_utils import require_torch, slow

from ..test_tokenization_common import TokenizerTesterMixin


class MCTCTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MCTCTokenizer
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
        return MCTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def test_tokenizer_add_token_chars(self):
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")

        # check adding a single token
        tokenizer.add_tokens("x")
        token_ids = tokenizer("C x A").input_ids
        self.assertEqual(token_ids, [24, 81, 77, 81, 22])

        tokenizer.add_tokens(["a", "b", "c"])
        token_ids = tokenizer("C a A c").input_ids
        self.assertEqual(token_ids, [24, 81, 54, 81, 22, 81, 56])

        tokenizer.add_tokens(["a", "b", "c"])
        token_ids = tokenizer("CaA c").input_ids
        self.assertEqual(token_ids, [24, 54, 22, 81, 56])

    def test_tokenizer_add_token_words(self):
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")

        # check adding a single token
        tokenizer.add_tokens("xxx")
        token_ids = tokenizer("C xxx A B").input_ids
        self.assertEqual(token_ids, [24, 81, 8067, 81, 22, 81, 23])

        tokenizer.add_tokens(["aaa", "bbb", "ccc"])
        token_ids = tokenizer("C aaa A ccc B B").input_ids
        self.assertEqual(token_ids, [24, 81, 8068, 81, 22, 81, 8070, 81, 23, 81, 23])

        tokenizer.add_tokens(["aaa", "bbb", "ccc"])
        token_ids = tokenizer("CaaaA ccc B B").input_ids
        self.assertEqual(token_ids, [24, 8068, 22, 81, 8070, 81, 23, 81, 23])

    def test_tokenizer_decode(self):
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")

        sample_ids = [
            [29, 26, 33, tokenizer.pad_token_id, 33, 36],
            [23, 46, 26, tokenizer.word_delimiter_token_id, 81, 23, 46, 26],
        ]
        tokens = tokenizer.decode(sample_ids[0])
        batch_tokens = tokenizer.batch_decode(sample_ids)
        self.assertEqual(tokens, batch_tokens[0])
        self.assertEqual(batch_tokens, ["HELLO", "BYE BYE"])

    def test_tokenizer_decode_special(self):
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")
        # fmt: off
        sample_ids = [
            [29, 26, 33, tokenizer.pad_token_id, 33, 36],
            [23, 46, 26, tokenizer.word_delimiter_token_id, 81, 23, 46, 26],
        ]
        sample_ids_2 = [
            [29, 26, 26, 26, 26, 26, 33, 33, 33, 33, 33, tokenizer.pad_token_id, 33, 36],
            [23, 46, 26, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.word_delimiter_token_id, 81, 23, 46, 26, tokenizer.word_delimiter_token_id],
        ]
        # fmt: on

        batch_tokens = tokenizer.batch_decode(sample_ids)
        batch_tokens_2 = tokenizer.batch_decode(sample_ids_2)
        self.assertEqual(batch_tokens, batch_tokens_2)
        self.assertEqual(batch_tokens, ["HELLO", "BYE BYE"])

    def test_tokenizer_decode_added_tokens(self):
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")
        tokenizer.add_special_tokens({"cls_token": "$$$"})
        token_ids = tokenizer("HELLO$$$").input_ids
        print("token_ids", token_ids)

        # fmt: off
        sample_ids = [
            [29, 26, 33, tokenizer.pad_token_id, 33, 33, 33, 36, 36, tokenizer.word_delimiter_token_id, 36, 36, 8067, 8067],
        ]
        # fmt: on
        batch_tokens = tokenizer.batch_decode(sample_ids)

        self.assertEqual(batch_tokens, ["HELLO O$$$"])

    def test_special_characters_in_vocab(self):
        sent = "ʈʰ æ æ̃ ˧ kʰ"

        vocab_dict = {k: v for v, k in enumerate({phoneme for phoneme in sent.split()})}
        vocab_file = os.path.join(self.tmpdirname, "vocab_special.json")

        with open(vocab_file, "w") as f:
            json.dump(vocab_dict, f)

        tokenizer = MCTCTokenizer(vocab_file)

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

        tokenizer.save_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))
        tokenizer = MCTCTokenizer.from_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)


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

    @slow
    @require_torch
    def test_pretrained_checkpoints_are_set_correctly(self):
        # this test makes sure that models that are using
        # group norm don't have their tokenizer return the
        # attention_mask
        for model_id in MCTC_PRETRAINED_MODEL_ARCHIVE_LIST:
            config = MCTCConfig.from_pretrained(model_id)
            tokenizer = MCTCTokenizer.from_pretrained(model_id)

            # only "layer" feature extraction norm should make use of
            # attention_mask
            self.assertEqual(tokenizer.return_attention_mask, config.feat_extract_norm == "layer")

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
        # check MCTCTokenizerOutput keys for char
        self.assertTrue(len(outputs_char.keys()), 2)
        self.assertTrue("text" in outputs_char)
        self.assertTrue("char_offsets" in outputs_char)
        self.assertTrue(isinstance(outputs_char, MCTCTokenizerOutput))

        outputs_word = tokenizer.decode(sample_ids, output_word_offsets=True)
        # check MCTCTokenizerOutput keys for word
        self.assertTrue(len(outputs_word.keys()), 2)
        self.assertTrue("text" in outputs_word)
        self.assertTrue("word_offsets" in outputs_word)
        self.assertTrue(isinstance(outputs_word, MCTCTokenizerOutput))

        outputs = tokenizer.decode(sample_ids, output_char_offsets=True, output_word_offsets=True)
        # check MCTCTokenizerOutput keys for both
        self.assertTrue(len(outputs.keys()), 3)
        self.assertTrue("text" in outputs)
        self.assertTrue("char_offsets" in outputs)
        self.assertTrue("word_offsets" in outputs)
        self.assertTrue(isinstance(outputs, MCTCTokenizerOutput))

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
            self.assertTrue(isinstance(outputs_batch, MCTCTokenizerOutput))
            self.assertTrue(isinstance(outputs_list[0], MCTCTokenizerOutput))

            # transform list to ModelOutput
            outputs_batch_2 = MCTCTokenizerOutput({k: [d[k] for d in outputs_list] for k in outputs_list[0]})

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
        tokenizer = self.tokenizer_class.from_pretrained("cwkeam/mctc-large")
        # pred_ids correspond to the following code
        # ```
        # from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
        # from datasets import load_dataset
        # import datasets
        # import torch
        # model = AutoModelForCTC.from_pretrained("cwkeam/mctc-large")
        # feature_extractor = AutoFeatureExtractor.from_pretrained("cwkeam/mctc-large")

        # ds = load_dataset("common_voice", "en", split="train", streaming=True)
        # ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        # ds_iter = iter(ds)
        # sample = next(ds_iter)

        # input_features = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_features
        # logits = model(input_features).logits
        # pred_ids = torch.argmax(logits, axis=-1).cpu().tolist()
        # print(pred_ids)
        # ```
        # fmt: off
        pred_ids = [[81, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 44, 8064, 61, 8064, 78, 8064, 8064, 8064, 8064, 81, 8064, 57, 8064, 8064, 68, 8064, 58, 8064, 72, 8064, 8064, 8064, 8064, 81, 81, 81, 8064, 8064, 8064, 34, 8064, 58, 8064, 8064, 8064, 65, 8064, 62, 72, 8064, 8064, 72, 8064, 8064, 54, 8064, 67, 8064, 8064, 57, 71, 71, 54, 8064, 8064, 8064, 81, 65, 8064, 68, 8064, 68, 64, 8064, 8064, 8064, 81, 65, 62, 8064, 64, 58, 8064, 8064, 8064, 8064, 81, 81, 72, 61, 58, 58, 81, 81, 76, 54, 8064, 67, 73, 73, 72, 72, 81, 81, 73, 8064, 68, 81, 81, 8064, 8064, 56, 8064, 68, 8064, 67, 8064, 72, 8064, 8064, 74, 8064, 66, 8064, 58, 8064, 81, 81, 81, 8064, 8064, 8064, 31, 8064, 68, 61, 61, 67, 67, 8064, 8064, 81, 81, 8064, 40, 8064, 67, 8064, 68, 8064, 76, 8064, 8064, 81, 8064, 68, 67, 8064, 81, 8064, 73, 61, 61, 58, 81, 81, 8064, 71, 8064, 62, 60, 8064, 61, 73, 8064, 8064, 11, 8064, 74, 8064, 69, 8064, 8064, 8064, 81, 8064, 73, 61, 58, 8064, 81, 8064, 8064, 76, 8064, 54, 8064, 65, 8064, 8064, 65, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 8064, 12, 8064, 81]]
        # wav2vec2-base downsamples input audio by a factor of 320
        # sampling rate for wav2vec2-base is 16_000
        time_offset_mctc = 480 / 16_000

        expected_char_time_stamps_text = [' ', 'W', 'h', 'y', ' ', 'd', 'o', 'e', 's', ' ', 'M', 'e', 'l', 'i', 's', 's', 'a', 'n', 'd', 'r', 'a', ' ', 'l', 'o', 'o', 'k', ' ', 'l', 'i', 'k', 'e', ' ', 's', 'h', 'e', ' ', 'w', 'a', 'n', 't', 's', ' ', 't', 'o', ' ', 'c', 'o', 'n', 's', 'u', 'm', 'e', ' ', 'J', 'o', 'h', 'n', ' ', 'S', 'n', 'o', 'w', ' ', 'o', 'n', ' ', 't', 'h', 'e', ' ', 'r', 'i', 'g', 'h', 't', '-', 'u', 'p', ' ', 't', 'h', 'e', ' ', 'w', 'a', 'l', 'l', '.', ' ']
        expected_char_time_stamps_start =  [0.0, 1.23, 1.29, 1.35, 1.5, 1.56, 1.65, 1.71, 1.77, 1.92, 2.1, 2.16, 2.28, 2.34, 2.37, 2.46, 2.55, 2.61, 2.7, 2.73, 2.79, 2.91, 2.94, 3.0, 3.06, 3.09, 3.21, 3.24, 3.27, 3.33, 3.36, 3.51, 3.57, 3.6, 3.63, 3.69, 3.75, 3.78, 3.84, 3.87, 3.93, 3.99, 4.05, 4.11, 4.14, 4.26, 4.32, 4.38, 4.44, 4.53, 4.59, 4.65, 4.71, 4.89, 4.95, 4.98, 5.04, 5.16, 5.25, 5.31, 5.37, 5.43, 5.52, 5.58, 5.61, 5.67, 5.73, 5.76, 5.82, 5.85, 5.94, 6.0, 6.03, 6.09, 6.12, 6.21, 6.27, 6.33, 6.45, 6.51, 6.54, 6.57, 6.63, 6.72, 6.78, 6.84, 6.93, 7.59, 7.65]
        expected_char_time_stamps_end = [0.03, 1.26, 1.32, 1.38, 1.53, 1.59, 1.68, 1.74, 1.8, 2.01, 2.13, 2.19, 2.31, 2.37, 2.4, 2.49, 2.58, 2.64, 2.73, 2.79, 2.82, 2.94, 2.97, 3.03, 3.09, 3.12, 3.24, 3.27, 3.3, 3.36, 3.39, 3.57, 3.6, 3.63, 3.69, 3.75, 3.78, 3.81, 3.87, 3.93, 3.99, 4.05, 4.08, 4.14, 4.2, 4.29, 4.35, 4.41, 4.47, 4.56, 4.62, 4.68, 4.8, 4.92, 4.98, 5.04, 5.1, 5.22, 5.28, 5.34, 5.4, 5.46, 5.55, 5.61, 5.64, 5.7, 5.76, 5.82, 5.85, 5.91, 5.97, 6.03, 6.06, 6.12, 6.15, 6.24, 6.3, 6.36, 6.48, 6.54, 6.57, 6.6, 6.66, 6.75, 6.81, 6.87, 6.96, 7.62, 7.68]


        expected_word_time_stamps_text = ['Why', 'does', 'Melissandra', 'look', 'like', 'she', 'wants', 'to', 'consume', 'John', 'Snow', 'on', 'the', 'right-up', 'the', 'wall.']
        expected_word_time_stamps_start = [1.23, 1.56, 2.1, 2.94, 3.24, 3.57, 3.75, 4.05, 4.26, 4.89, 5.25, 5.58, 5.73, 5.94, 6.51, 6.72]
        expected_word_time_stamps_end = [1.38, 1.8, 2.82, 3.12, 3.39, 3.69, 3.99, 4.14, 4.68, 5.1, 5.46, 5.64, 5.85, 6.36, 6.6, 7.62]

        output = tokenizer.batch_decode(pred_ids, output_char_offsets=True, output_word_offsets=True)

        char_offsets_text = self.get_from_offsets(output["char_offsets"][0], "char")
        char_offsets_start = self.get_from_offsets(output["char_offsets"][0], "start_offset")
        char_offsets_end = self.get_from_offsets(output["char_offsets"][0], "end_offset")

        word_offsets_text = self.get_from_offsets(output["word_offsets"][0], "word")
        word_offsets_start = self.get_from_offsets(output["word_offsets"][0], "start_offset")
        word_offsets_end = self.get_from_offsets(output["word_offsets"][0], "end_offset")

        # let's transform offsets to time stamps in seconds
        char_time_stamps_start = [round(c * time_offset_mctc, 2) for c in char_offsets_start]
        char_time_stamps_end = [round(c * time_offset_mctc, 2) for c in char_offsets_end]

        word_time_stamps_start = [round(w * time_offset_mctc, 2) for w in word_offsets_start]
        word_time_stamps_end = [round(w * time_offset_mctc, 2) for w in word_offsets_end]

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

    def test_pretrained_model_lists(self):
        # MCTCModel has no max model length => no testing
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