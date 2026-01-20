# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the VibeVoice processor."""

import tempfile
import unittest

import numpy as np

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, VibeVoiceProcessor


@require_torch
class VibeVoiceProcessorTest(unittest.TestCase):
    """Tests for VibeVoiceProcessor."""

    def setUp(self):
        # Use a small Qwen2 tokenizer for testing
        # In practice, we'd use the actual tokenizer from the model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        self.processor = VibeVoiceProcessor(tokenizer=self.tokenizer)

    def test_processor_init(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.sampling_rate, 24000)

    def test_processor_init_requires_tokenizer(self):
        """Test that processor requires a tokenizer."""
        with self.assertRaises(ValueError):
            VibeVoiceProcessor(tokenizer=None)

    def test_process_text(self):
        """Test processing text input."""
        text = "Hello, this is a test."
        outputs = self.processor(text=text)

        self.assertIn("input_ids", outputs)
        self.assertIn("attention_mask", outputs)

    def test_process_text_batch(self):
        """Test processing batch of text inputs."""
        texts = ["Hello, this is a test.", "Another test sentence."]
        outputs = self.processor(text=texts, padding=True)

        self.assertIn("input_ids", outputs)
        self.assertEqual(len(outputs["input_ids"]), 2)

    def test_process_text_with_padding(self):
        """Test processing text with padding."""
        text = "Hello"
        outputs = self.processor(text=text, padding="max_length", max_length=20)

        self.assertEqual(outputs["input_ids"].shape[-1], 20)

    def test_process_audio(self):
        """Test processing audio input."""
        # Create dummy audio
        audio = np.random.randn(24000).astype(np.float32)  # 1 second of audio
        outputs = self.processor(text="Test", audio=audio)

        self.assertIn("input_ids", outputs)
        self.assertIn("audio_values", outputs)

    def test_process_audio_normalization(self):
        """Test that audio is normalized properly."""
        # Create audio that needs normalization (int16 range)
        audio = np.random.randint(-32768, 32767, size=24000).astype(np.float32)
        outputs = self.processor(text="Test", audio=audio)

        self.assertIn("audio_values", outputs)

    def test_process_audio_channel_handling(self):
        """Test audio channel dimension handling."""
        # 1D audio
        audio_1d = np.random.randn(24000).astype(np.float32)
        outputs_1d = self.processor(text="Test", audio=audio_1d)
        self.assertIn("audio_values", outputs_1d)

    def test_process_speaker_string(self):
        """Test processing with speaker name (requires speaker embeddings)."""
        # Without speaker embeddings loaded, this should raise an error
        with self.assertRaises(ValueError):
            self.processor(text="Test", speaker="speaker_1")

    def test_process_speaker_dict(self):
        """Test processing with speaker dict."""
        speaker_data = {"embedding": np.random.randn(256).astype(np.float32)}
        outputs = self.processor(text="Test", speaker=speaker_data)

        self.assertIn("speaker_embedding", outputs)

    def test_process_speaker_array(self):
        """Test processing with speaker array."""
        speaker_array = np.random.randn(256).astype(np.float32)
        outputs = self.processor(text="Test", speaker=speaker_array)

        self.assertIn("speaker_embedding", outputs)

    def test_process_requires_text_or_audio(self):
        """Test that processor requires at least text or audio."""
        with self.assertRaises(ValueError):
            self.processor()

    def test_available_speakers_empty(self):
        """Test available_speakers property when no speakers loaded."""
        self.assertEqual(self.processor.available_speakers, [])

    def test_model_input_names(self):
        """Test model_input_names property."""
        input_names = self.processor.model_input_names
        self.assertIn("input_ids", input_names)
        self.assertIn("attention_mask", input_names)

    def test_batch_decode(self):
        """Test batch_decode method."""
        text = "Hello, this is a test."
        outputs = self.processor(text=text)
        decoded = self.processor.batch_decode(outputs["input_ids"])

        self.assertIsInstance(decoded, list)

    def test_decode(self):
        """Test decode method."""
        text = "Hello"
        outputs = self.processor(text=text)
        decoded = self.processor.decode(outputs["input_ids"][0])

        self.assertIsInstance(decoded, str)

    def test_processor_save_load(self):
        """Test saving and loading processor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.processor.save_pretrained(tmp_dir)

            # Load processor
            loaded_processor = VibeVoiceProcessor.from_pretrained(tmp_dir)

        self.assertEqual(self.processor.sampling_rate, loaded_processor.sampling_rate)

    def test_return_tensors_pt(self):
        """Test returning PyTorch tensors."""
        text = "Hello"
        outputs = self.processor(text=text, return_tensors="pt")

        self.assertIsInstance(outputs["input_ids"], torch.Tensor)

    def test_return_tensors_np(self):
        """Test returning NumPy arrays."""
        text = "Hello"
        outputs = self.processor(text=text, return_tensors="np")

        self.assertIsInstance(outputs["input_ids"], np.ndarray)


@require_torch
class VibeVoiceProcessorWithSpeakerEmbeddingsTest(unittest.TestCase):
    """Tests for VibeVoiceProcessor with speaker embeddings."""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        # Create mock speaker embeddings
        self.speaker_embeddings = {
            "repo_or_path": "/tmp",
            "speaker_1": {"embedding": "speaker_1_embedding.npy"},
            "speaker_2": {"embedding": "speaker_2_embedding.npy"},
        }
        self.processor = VibeVoiceProcessor(
            tokenizer=self.tokenizer,
            speaker_embeddings=self.speaker_embeddings,
        )

    def test_available_speakers(self):
        """Test available_speakers property."""
        speakers = self.processor.available_speakers
        self.assertIn("speaker_1", speakers)
        self.assertIn("speaker_2", speakers)
        self.assertNotIn("repo_or_path", speakers)


@require_torch
@slow
class VibeVoiceProcessorIntegrationTest(unittest.TestCase):
    """Integration tests for VibeVoiceProcessor."""

    @unittest.skip("Model not yet available on Hub")
    def test_processor_from_pretrained(self):
        """Test loading processor from HuggingFace Hub."""
        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        self.assertIsNotNone(processor)

    @unittest.skip("Model not yet available on Hub")
    def test_full_pipeline(self):
        """Test full processing pipeline."""
        from transformers import VibeVoiceForConditionalGeneration

        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        model = VibeVoiceForConditionalGeneration.from_pretrained("microsoft/VibeVoice-1.5B")

        text = "Hello, this is a test of the VibeVoice system."
        inputs = processor(text=text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs)


if __name__ == "__main__":
    unittest.main()
