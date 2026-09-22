# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import EmbeddingGemma2Processor, EmbeddingGemma2VideoProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class EmbeddingGemma2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EmbeddingGemma2Processor
    videos_unstructured_max_length = 570
    videos_text_kwargs_max_length = 570
    videos_text_kwargs_override_max_length = 570

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        video_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
            "fps": 1,
        }
        return video_processor_class(**video_processor_kwargs)

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")
        return feature_extractor_class()

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
        }
        return image_processor_class(**image_processor_kwargs)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        extra_special_tokens = {
            "image_token": "<|image|>",
            "video_token": "<|video|>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "audio_token": "<audio_soft_token>",
            "boa_token": "<start_of_audio>",
            "eoa_token": "<end_of_audio>",
        }
        tokenizer = tokenizer_class.from_pretrained(
            SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 3}

    # Override as EmbeddingGemma 2 needs images to be an explicitly nested batch
    def prepare_images_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_images_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]

        # Overwritten because EmbeddingGemma 2 (like Gemma 3/4) needs nested image inputs
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append([np.random.randint(255, size=(h, w, 3), dtype=np.uint8)])

        text = [f"This is an image {getattr(self, 'image_token', '')}"] * len(image_inputs)
        inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt")

        # EmbeddingGemma 2 does not return `mm_token_type_ids`, so count the image placeholders
        # directly in `input_ids`.
        num_image_tokens_from_call = (inputs.input_ids == processor.image_token_id).sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    def test_get_num_multimodal_tokens_matches_processor_call_audio(self):
        """Tests the audio branch of the helper used internally in vLLM.

        `_compute_audio_num_tokens` derives the count analytically from the mel-frame
        and SSCP subsampling arithmetic, while the processor derives it from the audio
        tower's mask. Those are two independent implementations of the same quantity,
        and `_compute_audio_num_tokens` hardcodes the subsampling stack (2 layers,
        kernel 3 / stride 2 / padding 1), so this guards them against drifting apart.
        """

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        sampling_rate = processor.feature_extractor.sampling_rate
        # Sub-second through multi-second, so the mel/subsampling arithmetic is exercised
        # at several lengths rather than a single round number.
        audio_lengths = [sampling_rate // 4, sampling_rate // 2, sampling_rate, 2 * sampling_rate]
        audio_inputs = [np.zeros(length, dtype=np.float32) for length in audio_lengths]

        text = [f"This is audio {processor.audio_token}"] * len(audio_inputs)
        inputs = processor(text=text, audio=audio_inputs, padding=True, return_tensors="pt")

        # EmbeddingGemma 2 does not return `mm_token_type_ids`, so count the audio
        # placeholders directly in `input_ids`.
        num_audio_tokens_from_call = (inputs.input_ids == processor.audio_token_id).sum(-1).tolist()
        num_audio_tokens_from_helper = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)
        self.assertListEqual(num_audio_tokens_from_call, num_audio_tokens_from_helper["num_audio_tokens"])

    @require_torch
    def test_video_exclude_timestamps(self):
        """`exclude_timestamps=True` omits the timestamps and concatenates one block per frame."""
        processor = self.get_processor()
        video_inputs = [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)]
        text = f"{processor.video_token} What is this video?"

        out = processor(text=text, videos=[video_inputs], do_sample_frames=False, return_tensors="pt")
        decoded = processor.decode(out["input_ids"][0])
        self.assertNotIn("00:00", decoded)

        num_soft_tokens = processor.video_processor(
            video_inputs, overflow_strategy="uniform", max_frames=2, fps=None, return_tensors="pt"
        )["num_soft_tokens_per_video"][0]
        expected_frame = f"{processor.boi_token}{processor.video_token * num_soft_tokens}{processor.eoi_token}"
        expected_video_str = expected_frame * 2
        self.assertIn(expected_video_str, decoded)

        # Opting back in restores the `mm:ss` timestamps
        out_with_ts = processor(
            text=text,
            videos=[video_inputs],
            do_sample_frames=True,
            videos_kwargs={"exclude_timestamps": False, "overflow_strategy": "uniform", "max_frames": 2, "fps": None},
            return_tensors="pt",
        )
        decoded_with_ts = processor.decode(out_with_ts["input_ids"][0])
        self.assertIn("00:00", decoded_with_ts)

    def test_processor_and_video_processor_serialization(self):
        """The EmbeddingGemma 2 video flags survive a `save_pretrained` / `from_pretrained` round-trip."""
        processor = self.get_processor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            loaded_processor = self.processor_class.from_pretrained(tmp_dir)
            self.assertIsInstance(loaded_processor.video_processor, EmbeddingGemma2VideoProcessor)
            self.assertTrue(loaded_processor.video_processor.max_frames)
            self.assertTrue(loaded_processor.video_processor.exclude_timestamps)

    def test_single_modality_inputs_need_no_text(self):
        """Unlike Gemma 4, any single modality on its own is a valid embedding input."""
        processor = self.get_processor()

        image_input = self.prepare_images_inputs(batch_size=2)
        out = processor(images=image_input, return_tensors="np")
        self.assertIn("input_ids", out)
        self.assertIn(self.images_input_name, out)

        with self.assertRaises(ValueError):
            processor()
