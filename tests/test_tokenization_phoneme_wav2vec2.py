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
"""Tests for the Wav2Vec2Phoneme tokenizer."""
import json
import os
import random
import tempfile
import unittest

from transformers import Wav2Vec2PhonemeCTCTokenizer
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_phonemizer

from .test_tokenization_common import TokenizerTesterMixin


@require_phonemizer
class Wav2Vec2PhonemeCTCTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Wav2Vec2PhonemeCTCTokenizer
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
        return Wav2Vec2PhonemeCTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)

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

        tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_file)

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

        tokenizer.save_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(os.path.join(self.tmpdirname, "special_tokenizer"))

        expected_sent = tokenizer.decode(tokenizer(sent).input_ids, spaces_between_special_tokens=True)
        self.assertEqual(sent, expected_sent)

    def test_pretrained_model_lists(self):
        # Wav2Vec2PhonemeModel has no max model length => no testing
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
