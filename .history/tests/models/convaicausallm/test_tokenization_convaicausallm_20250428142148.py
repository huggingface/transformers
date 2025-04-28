# Copyright 2024 Convai Innovations Inc. and The HuggingFace Team. All rights reserved.
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

from transformers import (
    AddedToken,
    ConvaiCausalLMTokenizer,
    ConvaiCausalLMTokenizerFast,
)
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_jinja,
    require_read_token, # Add if model is private/gated
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin


# Fixture files are specific to the tokenizer, get your actual tokenizer.model
# SAMPLE_VOCAB = get_tests_dir("fixtures/your_tokenizer.model") # Replace with path to your test vocab if needed
# For integration tests, we'll load from the hub


@require_sentencepiece
@require_tokenizers
class ConvaiCausalLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # Point to your model on the Hub or a local path containing tokenizer.model
    from_pretrained_id = "convaiinnovations/hindi-causal-lm"
    tokenizer_class = ConvaiCausalLMTokenizer
    rust_tokenizer_class = ConvaiCausalLMTokenizerFast
    test_rust_tokenizer = True # Set to True to test fast tokenizer
    test_sentencepiece = True
    from_pretrained_kwargs = {}
    test_seq2seq = False # Causal LM is not seq2seq

    # Use a setup method to load the actual tokenizer for testing consistency
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Save tokenizer to temp dir for tests that load from local path
        cls.tokenizer = cls.tokenizer_class.from_pretrained(cls.from_pretrained_id, **cls.from_pretrained_kwargs)
        cls.tokenizer.save_pretrained(cls.tmpdirname)
        if cls.test_rust_tokenizer:
            try:
                cls.rust_tokenizer = cls.rust_tokenizer_class.from_pretrained(cls.from_pretrained_id, **cls.from_pretrained_kwargs)
                # Save fast tokenizer files as well if conversion needed or tested
                cls.rust_tokenizer.save_pretrained(cls.tmpdirname)
            except Exception as e:
                # Handle case where fast tokenizer loading might fail initially
                print(f"Could not load fast tokenizer: {e}")
                cls.test_rust_tokenizer = False # Disable fast tests if loading fails

    # Override tests that might rely on specific Gemma vocab/behavior if needed
    # For example, if your tokenizer doesn't have specific behaviors tested in Gemma's test_pickle...

    @unittest.skip(reason="Subword regularization tests might depend on specific vocab properties not present.")
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    @unittest.skip(reason="Subword regularization tests might depend on specific vocab properties not present.")
    def test_subword_regularization_tokenizer(self):
        pass

    @slow
    @require_read_token # Add if model is private/gated
    def test_tokenizer_integration(self):
        # ** REPLACE with actual expected encoding for your Hindi tokenizer **
        # Run tokenizer("sample hindi text", padding=False, return_tensors=None)["input_ids"] to get these
        EXPECTED_ENCODING_SAMPLE_HINDI = {
            'input_ids': [[1, 8081, 377, 15117, 266, 685, 321, 2]], # Placeholder for "<s> भारत एक विशाल देश है </s>"
            # Add more samples as needed
        }
        # Note: Token IDs will be different for your 16k vocab Hindi model.
        # The structure [[ID, ID, ...]] is for a single sentence.
        # If testing multiple sentences, it becomes [[IDs for sent1], [IDs for sent2], ...]

        self.tokenizer_integration_test_util(
            expected_encoding=EXPECTED_ENCODING_SAMPLE_HINDI, # ** REPLACE THIS **
            model_name=self.from_pretrained_id,
            padding=False, # Adjust padding strategy if needed for test
        )

    # --- Specific Tests for ConvaiCausalLM Tokenizer ---

    def test_simple_encode_decode_hindi(self):
        # Test basic encoding/decoding with Hindi text
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer() if self.test_rust_tokenizer else None

        text = "भारत एक विशाल देश है"
        # ** REPLACE with expected IDs from your tokenizer **
        expected_ids = [1, 8081, 377, 15117, 266, 685, 321] # Assuming BOS=1, no EOS added by default
        expected_tokens = ["<s>", " भारत", " एक", " विशाल", " देश", " है"] # Example, check actual tokens

        # --- Slow Tokenizer ---
        tokens = tokenizer.tokenize(text)
        print("Actual Slow Tokens:", tokens)
        # self.assertEqual(tokens, expected_tokens) # Uncomment and assert after replacing

        encoded_ids = tokenizer.encode(text, add_special_tokens=True) # Assuming add_bos_token=True default
        print("Actual Slow IDs:", encoded_ids)
        # self.assertEqual(encoded_ids, expected_ids) # Uncomment and assert after replacing

        decoded_text = tokenizer.decode(expected_ids, skip_special_tokens=True)
        print("Actual Slow Decoded:", decoded_text)
        # self.assertEqual(decoded_text, text) # Uncomment and assert after replacing

        # --- Fast Tokenizer (if available) ---
        if rust_tokenizer:
            rust_tokens = rust_tokenizer.tokenize(text)
            print("Actual Fast Tokens:", rust_tokens)
            # self.assertEqual(rust_tokens, expected_tokens) # Uncomment and assert after replacing

            rust_encoded_ids = rust_tokenizer.encode(text, add_special_tokens=True)
            print("Actual Fast IDs:", rust_encoded_ids)
            # self.assertEqual(rust_encoded_ids, expected_ids) # Uncomment and assert after replacing

            rust_decoded_text = rust_tokenizer.decode(expected_ids, skip_special_tokens=True)
            print("Actual Fast Decoded:", rust_decoded_text)
            # self.assertEqual(rust_decoded_text, text) # Uncomment and assert after replacing

    def test_special_tokens_mapping(self):
        # Verify that special tokens (BOS, EOS, PAD, UNK) map to the correct IDs from your config
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 0)
        # Check UNK if it's defined differently from default <unk> ID
        # self.assertEqual(tokenizer.unk_token_id, EXPECTED_UNK_ID)

    def test_save_fast_load_slow(self):
        # Ensure that we can save a fast tokenizer and load it as a slow tokenizer
        if not self.test_rust_tokenizer:
            self.skipTest(reason="Fast tokenizer not available or test disabled.")

        slow_tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname)
        fast_tokenizer = self.rust_tokenizer_class.from_pretrained(self.tmpdirname)

        text = "कुछ नमूना पाठ" # Sample Hindi text
        # ** REPLACE with expected IDs **
        target_encoded = [1, 100, 200, 300] # Example

        slow_encoded = slow_tokenizer.encode(text, add_special_tokens=True)
        # self.assertEqual(slow_encoded, target_encoded) # Uncomment and assert after replacing

        slow_decoded = slow_tokenizer.decode(target_encoded, skip_special_tokens=True)
        # self.assertEqual(slow_decoded, text) # Uncomment and assert after replacing

        with tempfile.TemporaryDirectory() as dirname:
            fast_tokenizer.save_pretrained(dirname)
            slow_tokenizer_from_fast = self.tokenizer_class.from_pretrained(dirname)

        slow_from_fast_encoded = slow_tokenizer_from_fast.encode(text, add_special_tokens=True)
        # self.assertEqual(slow_from_fast_encoded, target_encoded) # Uncomment and assert after replacing

        slow_from_fast_decoded = slow_tokenizer_from_fast.decode(target_encoded, skip_special_tokens=True)
        # self.assertEqual(slow_from_fast_decoded, text) # Uncomment and assert after replacing

    # Add more tests specific to Hindi or SentencePiece behavior if needed