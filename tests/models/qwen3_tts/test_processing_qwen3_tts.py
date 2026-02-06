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

import shutil
import tempfile
import unittest

from transformers import Qwen2Tokenizer
from transformers.models.qwen3_tts import Qwen3TTSProcessor
from transformers.testing_utils import require_torch


@require_torch
class Qwen3TTSProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_tokenizer(self):
        """Get a Qwen2 tokenizer for testing."""
        try:
            return Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        except Exception:
            # If remote checkpoint not available, skip
            self.skipTest("Could not load Qwen2 tokenizer")

    def test_save_load_pretrained_default(self):
        """Test that processor can be saved and loaded."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor_loaded = Qwen3TTSProcessor.from_pretrained(self.tmpdirname)

        self.assertIsNotNone(processor_loaded)
        self.assertIsNotNone(processor_loaded.tokenizer)

    def test_processor_instantiation(self):
        """Test that processor can be instantiated with a tokenizer."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.tokenizer)
        self.assertEqual(processor.tokenizer, tokenizer)

    def test_single_text_processing(self):
        """Test processing a single text input."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        text = "Hello, this is a test."
        outputs = processor(text=text)

        self.assertIsNotNone(outputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertEqual(len(outputs["input_ids"]), 1)  # Batch size 1

    def test_batch_text_processing(self):
        """Test processing multiple text inputs."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        texts = [
            "This is the first text.",
            "This is the second text.",
        ]
        outputs = processor(text=texts)

        self.assertIsNotNone(outputs)
        self.assertIn("input_ids", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertEqual(len(outputs["input_ids"]), 2)  # Batch size 2

    def test_processor_with_padding(self):
        """Test processor with padding enabled."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        texts = [
            "Short",
            "This is a much longer text with more tokens",
        ]

        outputs = processor(text=texts, padding=True)

        # With padding, sequences should have same length
        self.assertEqual(
            len(outputs["input_ids"][0]),
            len(outputs["input_ids"][1])
        )

    def test_model_input_names(self):
        """Test that model_input_names property works."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        input_names = processor.model_input_names
        self.assertIsNotNone(input_names)
        self.assertIsInstance(input_names, list)
        self.assertIn("input_ids", input_names)

    def test_batch_decode(self):
        """Test batch_decode method."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        texts = ["Hello world", "Test text"]
        outputs = processor(text=texts)

        # Decode token ids back to text
        decoded = processor.batch_decode(outputs["input_ids"])
        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded), 2)
        self.assertIsInstance(decoded[0], str)

    def test_decode(self):
        """Test decode method."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        text = "Hello world"
        outputs = processor(text=text)

        # Decode single sequence
        decoded = processor.decode(outputs["input_ids"][0])
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, str)

    def test_processor_consistency(self):
        """Test that processor produces consistent results."""
        tokenizer = self.get_tokenizer()
        processor = Qwen3TTSProcessor(tokenizer=tokenizer)

        text = "Consistent test text"
        outputs1 = processor(text=text)
        outputs2 = processor(text=text)

        # Same input should produce same output
        self.assertListEqual(
            outputs1["input_ids"][0],
            outputs2["input_ids"][0]
        )


if __name__ == "__main__":
    unittest.main()
