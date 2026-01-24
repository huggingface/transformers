# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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
    AudioFlamingo3Processor,
    AutoProcessor,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_librosa, require_torch, require_torchaudio

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


class AudioFlamingo3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AudioFlamingo3Processor

    @classmethod
    @require_torch
    @require_torchaudio
    def setUpClass(cls):
        cls.checkpoint = "nvidia/audio-flamingo-3-hf"
        cls.tmpdirname = tempfile.mkdtemp()

        processor = AudioFlamingo3Processor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    @require_torchaudio
    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    @require_torch
    @require_torchaudio
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    @require_torchaudio
    def test_can_load_various_tokenizers(self):
        processor = AudioFlamingo3Processor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    @require_torchaudio
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = AudioFlamingo3Processor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = AudioFlamingo3Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = AudioFlamingo3Processor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)

    @require_torch
    @require_torchaudio
    def test_tokenizer_integration(self):
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
    @require_torchaudio
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
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
    @require_torchaudio
    def test_apply_transcription_request_single(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)

        audio_url = "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav"
        helper_outputs = processor.apply_transcription_request(audio=audio_url)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the input speech."},
                    {"type": "audio", "audio": audio_url},
                ],
            }
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    # Overwrite to remove skip numpy inputs (still need to keep as many cases as parent)
    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("AudioFlamingo3 only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )


class AudioFlamingo3MusicProcessingTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AudioFlamingo3Processor

    @classmethod
    @require_torch
    @require_torchaudio
    def setUpClass(cls):
        cls.checkpoint = "nvidia/music-flamingo-2601-hf"
        cls.tmpdirname = tempfile.mkdtemp()

        processor = AudioFlamingo3Processor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    @require_torchaudio
    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    @require_torch
    @require_torchaudio
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    @require_torchaudio
    def test_music_chat_template_and_boundaries(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_system_prompt = (
            "<|im_start|>system\nYou are Music Flamingo, a multimodal assistant for language and music. "
            "On each turn you receive an audio clip which contains music and optional text, "
            "you will receive at least one or both; use your world knowledge and reasoning "
            "to help the user with any task. Interpret the entirety of the content any input music"
            "--regardlenss of whether the user calls it audio, music, or sound.<|im_end|>\n"
        )

        # Verify that the music-specific system prompt is preserved
        self.assertIn(expected_system_prompt, processor.tokenizer.chat_template)

        # Basic integration test with dummy audio
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this track."},
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, add_generation_prompt=True
        )

        decoded = processor.decode(inputs["input_ids"][0])

        if processor.audio_bos_token is not None:
            self.assertIn(processor.audio_bos_token, decoded)
        if processor.audio_eos_token is not None:
            self.assertIn(processor.audio_eos_token, decoded)

        self.assertIn("<|im_start|>user", decoded)
        self.assertIn("Analyze this track", decoded)
        self.assertIn("<|im_start|>assistant", decoded)

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("AudioFlamingo3 only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    def prepare_processor_dict(self):
        return {
            "audio_bos_token": "<|sound_bos|>",
            "audio_eos_token": "<|sound_eos|>",
            "max_audio_len": 1200,
        }
