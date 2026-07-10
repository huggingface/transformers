# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MusicFlamingoProcessor,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_librosa, require_torch, slow

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


class MusicFlamingoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MusicFlamingoProcessor
    # Tiny processor created with make_tiny_processor.py from "nvidia/music-flamingo-2601-hf"
    tiny_model_id = "hf-internal-testing/tiny-processor-musicflamingo"
    checkpoint = "nvidia/music-flamingo-2601-hf"

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = MusicFlamingoProcessor.from_pretrained(cls.tiny_model_id)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = MusicFlamingoProcessor.from_pretrained(self.tiny_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id)
        processor = MusicFlamingoProcessor.from_pretrained(self.tiny_model_id)
        feature_extractor = processor.feature_extractor

        processor = MusicFlamingoProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = MusicFlamingoProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)

    @require_torch
    def test_tokenizer_integration(self):
        slow_tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id, use_fast=False)
        fast_tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_id, from_slow=True, legacy=False)

        prompt = (
            "<|im_start|>system\nAnswer the questions.<|im_end|>"
            "<|im_start|>user\n<sound>What is it?<|im_end|>"
            "<|im_start|>assistant\n"
        )

        # Verify slow and fast tokenizers produce the same output (parity test)
        self.assertEqual(slow_tokenizer.tokenize(prompt), fast_tokenizer.tokenize(prompt))

    @slow
    @require_torch
    def test_tokenizer_full_integration(self):
        slow_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        fast_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, from_slow=True, legacy=False)

        prompt = (
            "<|im_start|>system\nAnswer the questions.<|im_end|>"
            "<|im_start|>user\n<sound>What is it?<|im_end|>"
            "<|im_start|>assistant\n"
        )
        EXPECTED_OUTPUT = [
            "<|im_start|>",
            "system",
            "Ċ",
            "Answer",
            "Ġthe",
            "Ġquestions",
            ".",
            "<|im_end|>",
            "<|im_start|>",
            "user",
            "Ċ",
            "<sound>",
            "What",
            "Ġis",
            "Ġit",
            "?",
            "<|im_end|>",
            "<|im_start|>",
            "assistant",
            "Ċ",
        ]

        self.assertEqual(slow_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)
        self.assertEqual(fast_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)

    @require_torch
    def test_chat_template(self):
        processor = self.get_processor()
        expected_prompt = (
            "<|im_start|>system\nYou are Music Flamingo, a multimodal assistant for language and music. "
            "On each turn you receive an audio clip which contains music and optional text, "
            "you will receive at least one or both; use your world knowledge and reasoning "
            "to help the user with any task. Interpret the entirety of the content any input music"
            "--regardlenss of whether the user calls it audio, music, or sound.<|im_end|>\n"
            "<|im_start|>user\n<sound>What is surprising about the relationship between the barking and the music?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        conversations = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is surprising about the relationship between the barking and the music?",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
                    },
                ],
            }
        ]

        formatted = processor.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted)

    @require_torch
    def test_transcription_helpers_not_supported(self):
        processor = self.get_processor()
        self.assertFalse(hasattr(processor, "apply_transcription_request"))
        self.assertFalse(hasattr(processor, "_strip_assistant_prefix_and_quotes"))

    # Overwrite to remove skip numpy inputs (still need to keep as many cases as parent)
    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("MusicFlamingo only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    @require_torch
    def test_output_labels_with_audio(self):
        processor = self.get_processor()
        pad_token_id = processor.tokenizer.pad_token_id

        # Different text lengths so that padding is applied
        text = [
            f"{processor.audio_token} Describe the music.",
            f"{processor.audio_token} What instruments can you hear in this piece?",
        ]
        audio = self.prepare_audio_inputs(batch_size=2)

        inputs = processor(text=text, audio=audio, output_labels=True)

        self.assertIn("labels", inputs)
        self.assertNotIn("mm_token_type_ids", inputs)
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # audio token positions (including audio bos/eos) are masked
        audio_positions = (
            (input_ids == processor.audio_token_id)
            | (input_ids == processor.audio_bos_token_id)
            | (input_ids == processor.audio_eos_token_id)
        )
        self.assertTrue(audio_positions.any())
        self.assertTrue((labels[audio_positions] == -100).all())

        # padding positions are masked
        pad_positions = input_ids == pad_token_id
        self.assertTrue(pad_positions.any())
        self.assertTrue((labels[pad_positions] == -100).all())

        # all other positions match input_ids
        kept_positions = ~(audio_positions | pad_positions)
        self.assertTrue(kept_positions.any())
        self.assertTrue((labels[kept_positions] == input_ids[kept_positions]).all())

    @require_torch
    def test_output_labels_without_audio(self):
        processor = self.get_processor()
        pad_token_id = processor.tokenizer.pad_token_id

        # Different text lengths so that padding is applied
        text = ["Describe the music in detail.", "Hello!"]
        inputs = processor(text=text, output_labels=True)

        self.assertIn("labels", inputs)
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        self.assertEqual(labels.shape, input_ids.shape)

        # without audio, only padding positions are masked
        pad_positions = input_ids == pad_token_id
        self.assertTrue(pad_positions.any())
        self.assertTrue((labels[pad_positions] == -100).all())
        kept_positions = ~pad_positions
        self.assertTrue((labels[kept_positions] == input_ids[kept_positions]).all())
