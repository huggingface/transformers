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
from transformers.video_processing_utils import VideoMetadata

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
            "num_frames": 2,
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
        inputs = processor(
            text=text, images=image_inputs, padding=True, return_mm_token_type_ids=True, return_tensors="pt"
        )

        if "mm_token_type_ids" not in inputs:
            self.skipTest("Processor doesn't support `mm_token_type_ids`")

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    def test_video_processor_flag_defaults(self):
        """EmbeddingGemma 2 was trained on visual-only, 1-FPS-sampled video, so both flags default to True."""
        video_processor = EmbeddingGemma2VideoProcessor()
        self.assertTrue(video_processor.use_1fps_linear_sampling)
        self.assertTrue(video_processor.exclude_timestamps)

        # ... and the defaults survive the component setup used by the processor tests
        component = self.get_component("video_processor")
        self.assertTrue(component.use_1fps_linear_sampling)
        self.assertTrue(component.exclude_timestamps)

    def test_video_1fps_linear_sampling(self):
        """Tests that `use_1fps_linear_sampling` implements 1-FPS linspace sequence sampling."""
        video_processor = self.get_component("video_processor")

        # Short video: 10 seconds at 25 fps = 250 frames. One frame per second, all kept.
        meta_short = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)
        sampled_short = video_processor.sample_frames(meta_short, num_frames=32)
        expected_short = np.array([int(s * 25) for s in range(10)])
        np.testing.assert_array_equal(sampled_short, expected_short)

        # Long video: 100 seconds at 25 fps = 2500 frames. The per-second indices are subsampled
        # with a linspace down to `num_frames`.
        meta_long = VideoMetadata(fps=25.0, total_num_frames=2500, duration=100.0)
        sampled_long = video_processor.sample_frames(meta_long, num_frames=32)
        self.assertEqual(len(sampled_long), 32)
        sec_indices_long = [int(s * 25) for s in range(100)]
        expected_linspace_idx = np.linspace(0, 99, 32, dtype=int)
        expected_long = np.array([sec_indices_long[i] for i in expected_linspace_idx])
        np.testing.assert_array_equal(sampled_long, expected_long)

        # Fallback when `total_num_frames` is missing but `duration` and `fps` are known
        meta_duration_only = VideoMetadata(fps=25.0, total_num_frames=None, duration=10.0)
        sampled_duration = video_processor.sample_frames(meta_duration_only, num_frames=32)
        np.testing.assert_array_equal(sampled_duration, expected_short)

        # Error when neither `total_num_frames` nor `duration` is available
        meta_missing = VideoMetadata(fps=25.0, total_num_frames=None, duration=None)
        with self.assertRaises(ValueError):
            video_processor.sample_frames(meta_missing, num_frames=32)

        # Explicitly opting out falls back to the base uniform sampling
        sampled_default = video_processor.sample_frames(meta_short, num_frames=2, use_1fps_linear_sampling=False)
        self.assertEqual(len(sampled_default), 2)

    @require_torch
    def test_video_exclude_timestamps(self):
        """`exclude_timestamps=True` omits the timestamps and concatenates one block per frame."""
        processor = self.get_processor()
        video_inputs = [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)]
        text = f"{processor.video_token} What is this video?"

        out = processor(text=text, videos=[video_inputs], do_sample_frames=False, return_tensors="pt")
        decoded = processor.decode(out["input_ids"][0])
        self.assertNotIn("00:00", decoded)

        num_soft_tokens = processor.video_processor(video_inputs, return_tensors="pt")["num_soft_tokens_per_video"][0]
        expected_frame = f"{processor.boi_token}{processor.video_token * num_soft_tokens}{processor.eoi_token}"
        expected_video_str = expected_frame * 2
        self.assertIn(expected_video_str, decoded)

        # Opting back in restores the `mm:ss` timestamps
        out_with_ts = processor(
            text=text,
            videos=[video_inputs],
            do_sample_frames=False,
            videos_kwargs={"exclude_timestamps": False},
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
            self.assertTrue(loaded_processor.video_processor.use_1fps_linear_sampling)
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
