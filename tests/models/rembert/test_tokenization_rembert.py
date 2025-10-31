# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the RemBert tokenizer."""

import tempfile
import unittest

from tests.test_tokenization_common import AddedToken, TokenizerTesterMixin
from transformers import RemBertTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers
from transformers.tokenization_sentencepiece import SentencePieceExtractor


SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/rembert"
    tokenizer_class = RemBertTokenizer
    pre_trained_model_path = "google/rembert"

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁', '▁', '▁', '▁', '▁', '▁', '▁Hello', '▁', '<s>', '▁hi', '<s>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'coded', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']
    integration_expected_token_ids = [1357, 619, 577, 3515, 573, 119091, 623, 820, 18648, 586, 940, 7905, 571, 599, 902, 619, 98696, 780, 572, 573, 6334, 649, 3975, 244511, 1034, 3211, 24624, 3211, 24624, 573, 573, 573, 573, 573, 573, 24624, 573, 3, 1785, 3, 90608, 660, 6802, 15930, 2575, 689, 43272, 592, 185434, 581, 24624, 572, 2878, 1032, 620, 599, 9070, 1032, 620, 60827, 20490, 1865, 781, 734, 9711]
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁', '生活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁', '▁', '▁', '▁', '▁', '▁', '▁Hello', '▁', '<s>', '▁hi', '<s>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁en', 'coded', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']
    integration_expected_decoded_text = 'This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello       Hello <s> hi<s>there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing'

    # Copied from ReformerTokenizationTest.get_input_output_texts
    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

    def test_added_tokens_serialization(self):
        # Utility to test the added vocab
        def _test_added_vocab_and_eos(expected, tokenizer_class, expected_eos, temp_dir):
            tokenizer = tokenizer_class.from_pretrained(temp_dir)
            self.assertTrue(str(expected_eos) not in tokenizer.additional_special_tokens)
            self.assertIn(new_eos, tokenizer.added_tokens_decoder.values())
            self.assertEqual(tokenizer.added_tokens_decoder[tokenizer.eos_token_id], new_eos)
            self.assertTrue(all(item in tokenizer.added_tokens_decoder.items() for item in expected.items()))
            return tokenizer

        new_eos = AddedToken("[NEW_EOS]", rstrip=False, lstrip=True, normalized=False, special=True)
        new_masked_token = AddedToken("[MASK]", lstrip=True, rstrip=False, normalized=False)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                # Load a slow tokenizer from the hub, init with the new token for fast to also include it
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, eos_token=new_eos, mask_token=new_masked_token
                )
                EXPECTED_ADDED_TOKENS_DECODER = tokenizer.added_tokens_decoder
                with self.subTest("Hub -> Slow: Test loading a slow tokenizer from the hub)"):
                    self.assertEqual(tokenizer._special_tokens_map["eos_token"], new_eos)
                    self.assertIn(new_eos, list(tokenizer.added_tokens_decoder.values()))

                with tempfile.TemporaryDirectory() as tmp_dir_2:
                    tokenizer.save_pretrained(tmp_dir_2)
                    with self.subTest(
                        "Hub -> Slow -> Slow: Test saving this slow tokenizer and reloading it in the fast class"
                    ):
                        _test_added_vocab_and_eos(
                            EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_2
                        )

                    if self.rust_tokenizer_class is not None:
                        with self.subTest(
                            "Hub -> Slow -> Fast: Test saving this slow tokenizer and reloading it in the fast class"
                        ):
                            tokenizer_fast = _test_added_vocab_and_eos(
                                EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_2
                            )
                            with tempfile.TemporaryDirectory() as tmp_dir_3:
                                tokenizer_fast.save_pretrained(tmp_dir_3)
                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Fast: Test saving this fast tokenizer and reloading it in the fast class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Slow: Test saving this slow tokenizer and reloading it in the slow class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                with self.subTest("Hub -> Fast: Test loading a fast tokenizer from the hub)"):
                    if self.rust_tokenizer_class is not None:
                        tokenizer_fast = self.get_tokenizer(pretrained_name, eos_token=new_eos)
                        self.assertEqual(tokenizer_fast._special_tokens_map["eos_token"], new_eos)
                        self.assertIn(new_eos, list(tokenizer_fast.added_tokens_decoder.values()))
                        # We can't test the following because for BC we kept the default rstrip lstrip in slow not fast. Will comment once normalization is alright
                        with self.subTest("Hub -> Fast == Hub -> Slow: make sure slow and fast tokenizer match"):
                            self.assertTrue(
                                all(
                                    item in tokenizer.added_tokens_decoder.items()
                                    for item in EXPECTED_ADDED_TOKENS_DECODER.items()
                                )
                            )

                        EXPECTED_ADDED_TOKENS_DECODER = tokenizer_fast.added_tokens_decoder
                        with tempfile.TemporaryDirectory() as tmp_dir_4:
                            tokenizer_fast.save_pretrained(tmp_dir_4)
                            with self.subTest("Hub -> Fast -> Fast: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_4
                                )

                            with self.subTest("Hub -> Fast -> Slow: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_4
                                )
