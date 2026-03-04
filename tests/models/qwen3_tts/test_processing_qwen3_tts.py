# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Qwen3TTSProcessor."""

import shutil
import tempfile
import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, slow

if is_torch_available():
    import torch
    from transformers import Qwen3TTSProcessor


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


@require_torch
class Qwen3TTSProcessorTest(unittest.TestCase):
    """Unit tests for Qwen3TTSProcessor."""

    @classmethod
    def setUpClass(cls):
        # Download the tokenizer once for all tests in this class
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = Qwen3TTSProcessor(tokenizer=self.tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_processor(self):
        return Qwen3TTSProcessor.from_pretrained(self.tmpdirname)

    # ── Class attributes ──────────────────────────────────────────────────────

    def test_processor_class_attributes(self):
        """Processor exposes the expected class-level attributes."""
        self.assertIn("tokenizer", Qwen3TTSProcessor.attributes)
        self.assertIn("Qwen2Tokenizer", Qwen3TTSProcessor.tokenizer_class)
        self.assertIn("Qwen2TokenizerFast", Qwen3TTSProcessor.tokenizer_class)

    # ── Instantiation & persistence ───────────────────────────────────────────

    def test_processor_save_load_roundtrip(self):
        """Processor saves and reloads with identical configuration."""
        processor = self.get_processor()
        processor2 = Qwen3TTSProcessor.from_pretrained(self.tmpdirname)
        self.assertEqual(processor.tokenizer.__class__, processor2.tokenizer.__class__)
        self.assertEqual(processor.tokenizer.vocab_size, processor2.tokenizer.vocab_size)

    def test_processor_instantiation(self):
        """Processor can be created directly from a tokenizer object."""
        processor = Qwen3TTSProcessor(tokenizer=self.tokenizer)
        self.assertIsNotNone(processor)
        self.assertIs(processor.tokenizer, self.tokenizer)

    # ── Text encoding ─────────────────────────────────────────────────────────

    def test_single_text_encoding(self):
        """Processor encodes a single string and returns input_ids + attention_mask."""
        processor = self.get_processor()
        output = processor(text="Hello world")
        self.assertIn("input_ids", output)
        self.assertIn("attention_mask", output)
        self.assertEqual(len(output["input_ids"]), 1)
        self.assertEqual(len(output["input_ids"][0]), len(output["attention_mask"][0]))

    def test_batch_text_encoding(self):
        """Processor encodes a list of strings as a batch."""
        processor = self.get_processor()
        texts = ["Hello world", "How are you?", "This is a test."]
        output = processor(text=texts)
        self.assertIn("input_ids", output)
        self.assertEqual(len(output["input_ids"]), len(texts))

    def test_single_string_wrapped_as_list(self):
        """A bare string and a one-element list produce identical token sequences."""
        processor = self.get_processor()
        out_str = processor(text="Hello")
        out_list = processor(text=["Hello"])
        self.assertEqual(out_str["input_ids"], out_list["input_ids"])

    def test_return_tensors_pt(self):
        """return_tensors='pt' with padding returns a 2D torch.Tensor."""
        processor = self.get_processor()
        output = processor(
            text=["Hello world", "Hi"],
            return_tensors="pt",
            padding=True,
        )
        self.assertIsInstance(output["input_ids"], torch.Tensor)
        self.assertEqual(output["input_ids"].dim(), 2)
        self.assertEqual(output["input_ids"].shape[0], 2)

    def test_missing_text_raises_value_error(self):
        """Calling processor without the text argument raises ValueError."""
        processor = self.get_processor()
        with self.assertRaises((ValueError, TypeError)):
            processor()

    def test_none_text_raises_value_error(self):
        """Calling processor with text=None raises ValueError."""
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor(text=None)

    def test_padding_produces_equal_length_sequences(self):
        """Padding a variable-length batch makes all sequences the same length."""
        processor = self.get_processor()
        texts = ["Short", "This is a much longer text with more tokens"]
        output = processor(text=texts, padding=True)
        self.assertEqual(len(output["input_ids"][0]), len(output["input_ids"][1]))

    def test_truncation_kwarg_respected(self):
        """max_length / truncation kwargs are forwarded to the tokenizer."""
        processor = self.get_processor()
        long_text = "word " * 100
        output = processor(text=long_text, return_tensors="pt", truncation=True, max_length=10)
        self.assertLessEqual(output["input_ids"].shape[1], 10)

    # ── Default padding side ──────────────────────────────────────────────────

    def test_default_padding_side_is_left(self):
        """Default padding side is left (correct for auto-regressive generation)."""
        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            self.skipTest("Tokenizer has no pad token.")
        output = processor(
            text=["Hello world today", "Hi"],
            return_tensors="pt",
            padding=True,
        )
        pad_id = processor.tokenizer.pad_token_id
        # With left-padding the last column must not be a pad token for either row
        self.assertNotEqual(output["input_ids"][0, -1].item(), pad_id)
        self.assertNotEqual(output["input_ids"][1, -1].item(), pad_id)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def test_batch_decode(self):
        """batch_decode returns a list of strings of the same length as the batch."""
        processor = self.get_processor()
        texts = ["Hello world", "Test text"]
        encoded = processor(text=texts)
        decoded = processor.batch_decode(encoded["input_ids"])
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), len(texts))
        self.assertIsInstance(decoded[0], str)

    def test_decode(self):
        """decode returns a single string for a single token sequence."""
        processor = self.get_processor()
        encoded = processor(text="Hello world")
        decoded = processor.decode(encoded["input_ids"][0])
        self.assertIsInstance(decoded, str)

    # ── model_input_names ─────────────────────────────────────────────────────

    def test_model_input_names_contains_input_ids(self):
        """model_input_names includes 'input_ids'."""
        processor = self.get_processor()
        self.assertIn("input_ids", processor.model_input_names)

    def test_model_input_names_no_duplicates(self):
        """model_input_names has no duplicate entries."""
        processor = self.get_processor()
        names = processor.model_input_names
        self.assertEqual(len(names), len(set(names)))

    # ── Consistency ───────────────────────────────────────────────────────────

    def test_encoding_is_deterministic(self):
        """The same text input always produces the same token IDs."""
        processor = self.get_processor()
        text = "Consistent test text"
        out1 = processor(text=text)
        out2 = processor(text=text)
        self.assertEqual(out1["input_ids"], out2["input_ids"])


@require_torch
class Qwen3TTSProcessorIntegrationTest(unittest.TestCase):
    """Integration tests for Qwen3TTSProcessor (marked @slow; require real model weights)."""

    model_id = MODEL_ID

    @slow
    def test_can_load_processor_from_pretrained(self):
        """Processor loads from the real pretrained model."""
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        self.assertIsNotNone(processor.tokenizer)

    @slow
    def test_tokenizer_integration(self):
        """Tokenizer correctly tokenises a standard TTS prompt."""
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        text = "Hello, how are you doing today?"
        output = processor(text=text, return_tensors="pt")
        self.assertIn("input_ids", output)
        self.assertGreater(output["input_ids"].shape[1], 0)

    @slow
    def test_processor_encode_decode_roundtrip(self):
        """Encoding then decoding a known string recovers the original text."""
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        text = "Hello world"
        encoded = processor(text=text, return_tensors="pt")
        decoded = processor.decode(encoded["input_ids"][0], skip_special_tokens=True)
        self.assertIn("Hello", decoded)

    @slow
    def test_apply_chat_template(self):
        """apply_chat_template formats a conversation into a non-empty string."""
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template.")
        conversation = [{"role": "user", "content": "Say hello."}]
        result = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @slow
    def test_batch_encoding_shapes(self):
        """Batch encoding with padding returns tensors with matching sequence lengths."""
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        texts = ["Hello.", "The weather is nice today."]
        output = processor(text=texts, return_tensors="pt", padding=True)
        self.assertEqual(output["input_ids"].shape[0], 2)
        self.assertEqual(output["input_ids"].shape, output["attention_mask"].shape)
