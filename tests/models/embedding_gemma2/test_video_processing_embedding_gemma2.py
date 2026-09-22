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

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available() and is_torchvision_available():
    from transformers import EmbeddingGemma2VideoProcessor


class EmbeddingGemma2VideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_frames=8,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        patch_size=6,
        max_soft_tokens=70,
        pooling_kernel_size=1,
    ):
        image_mean = image_mean if image_mean is not None else [0.0, 0.0, 0.0]
        image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "patch_size": self.patch_size,
            "max_soft_tokens": self.max_soft_tokens,
            "pooling_kernel_size": self.pooling_kernel_size,
            "do_sample_frames": False,
        }

    def expected_output_video_shape(self, videos=None):
        """Encoder-free output is padded to max_soft_tokens: shape does not depend on input resolution."""
        model_patch_size = self.patch_size * self.pooling_kernel_size
        return [self.num_frames, self.max_soft_tokens, model_patch_size**2 * 3]

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        videos = prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )
        return videos


@require_torch
@require_vision
class EmbeddingGemma2VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = EmbeddingGemma2VideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = EmbeddingGemma2VideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    @unittest.skip(
        "EmbeddingGemma 2 patchification requires RGB (3-channel) videos; 4-channel inputs are unsupported."
    )
    def test_call_numpy_4_channels(self):
        pass

    def test_call_sample_frames(self):
        """`max_frames` sampling only: EmbeddingGemma 2 defaults to 1-FPS sampling, which needs metadata,
        and a class-level `max_frames` default means `fps`-only never reaches the metadata-required path."""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict, fps=1)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )

            video_processing.do_sample_frames = False
            encoded = video_processing(video_inputs[0], return_tensors="pt", fps=None, max_frames=3)[self.input_name]
            self.assertEqual(encoded.shape[1], self.video_processor_tester.num_frames)

            video_processing.do_sample_frames = True
            encoded = video_processing(video_inputs[0], return_tensors="pt", fps=None, max_frames=3)[self.input_name]
            encoded_batched = video_processing(video_inputs, return_tensors="pt", fps=None, max_frames=3)[
                self.input_name
            ]
            self.assertEqual(encoded.shape[1], 3)
            self.assertEqual(encoded_batched.shape[1], 3)

    def test_video_processor_from_dict_with_kwargs(self):
        """EmbeddingGemma 2 has no `size`/`crop_size`; override with patch budget kwargs instead."""
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.patch_size, self.video_processor_tester.patch_size)
        self.assertEqual(video_processor.max_soft_tokens, self.video_processor_tester.max_soft_tokens)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=18)
        self.assertEqual(video_processor.patch_size, 18)

    def test_video_processor_defaults(self):
        processor = self.fast_video_processing_class()
        self.assertEqual(processor.patch_size, 16)
        self.assertEqual(processor.max_soft_tokens, 70)
        self.assertEqual(processor.pooling_kernel_size, 3)
        self.assertEqual(processor.fps, 1)

    def test_embedding_gemma2_video_flag_defaults(self):
        """EmbeddingGemma 2 was trained on visual-only, 1-FPS-sampled video, so both flags default to
        `True` -- unlike Gemma 4, which defaults both to `False`."""
        processor = self.fast_video_processing_class()
        self.assertTrue(processor.max_frames)
        self.assertTrue(processor.exclude_timestamps)

    def test_video_flag_defaults_survive_roundtrip(self):
        import tempfile

        processor = self.fast_video_processing_class()
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = self.fast_video_processing_class.from_pretrained(tmpdir)
        self.assertTrue(reloaded.exclude_timestamps)

    def test_sample_frames_1fps_linear(self):
        """`use_1fps_linear_sampling` takes one frame per second, then linspace-subsamples to `num_frames`."""
        processor = self.fast_video_processing_class()

        # 10 seconds at 25 fps: one frame per second, all kept.
        meta_short = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)
        sampled_short = processor.sample_frames(meta_short, fps=1, max_frames=32)
        expected_short = np.array([int(s * 25) for s in range(10)])
        np.testing.assert_array_equal(sampled_short, expected_short)

        # 100 seconds at 25 fps: per-second indices subsampled down to `max_frames`.
        meta_long = VideoMetadata(fps=25.0, total_num_frames=2500, duration=100.0)
        sampled_long = processor.sample_frames(meta_long, fps=1, max_frames=32, overflow_strategy="uniform")
        self.assertEqual(len(sampled_long), 32)
        sec_indices_long = [int(s * 25) for s in range(100)]
        expected_long = np.array([sec_indices_long[i] for i in np.linspace(0, 99, 32, dtype=int)])
        np.testing.assert_array_equal(sampled_long, expected_long)

        # `metadata.fps` is not recoverable.
        meta_missing = VideoMetadata(fps=None, total_num_frames=25)
        with self.assertRaises(ValueError):
            processor.sample_frames(meta_missing, fps=1)

        # Opting out falls back to the base uniform sampling.
        self.assertEqual(len(processor.sample_frames(meta_short, overflow_strategy="uniform", max_frames=2)), 2)

    def test_unsupported_max_soft_tokens_raises(self):
        with self.assertRaises(ValueError):
            self.fast_video_processing_class(max_soft_tokens=71)

    def test_output_keys(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs(return_tensors="torch")
        result = processor(videos[0], return_tensors="pt")
        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_position_ids", result)
        self.assertIn("num_soft_tokens_per_video", result)

    def test_position_ids_structure(self):
        """Per frame: real positions are non-negative and contiguous, padding positions are (-1, -1)."""
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs(return_tensors="torch")
        result = processor(videos[0], return_tensors="pt")

        position_ids = result.video_position_ids[0]
        self.assertEqual(position_ids.shape[-1], 2)
        for frame_positions in position_ids:
            real_mask = frame_positions[:, 0] >= 0
            self.assertGreater(real_mask.sum().item(), 0)
            pad_mask = ~real_mask
            if pad_mask.any():
                self.assertTrue((frame_positions[pad_mask] == -1).all())
                last_real_idx = torch.where(real_mask)[0][-1].item()
                first_pad_idx = torch.where(pad_mask)[0][0].item()
                self.assertEqual(last_real_idx + 1, first_pad_idx)

    def test_padding_patches_are_zero(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        video = torch.randint(1, 255, (self.video_processor_tester.num_frames, 3, 50, 50), dtype=torch.uint8)
        result = processor(video, return_tensors="pt")

        position_ids = result.video_position_ids[0]
        pixel_values = result.pixel_values_videos[0]
        for frame_index in range(position_ids.shape[0]):
            pad_mask = position_ids[frame_index, :, 0] < 0
            if pad_mask.any():
                self.assertTrue((pixel_values[frame_index, pad_mask] == 0).all())

    def test_num_soft_tokens_per_video(self):
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = self.video_processor_tester.prepare_video_inputs(return_tensors="torch")
        result = processor(videos, return_tensors="pt")

        num_soft_tokens = np.asarray(result.num_soft_tokens_per_video)
        self.assertEqual(num_soft_tokens.shape[0], self.video_processor_tester.batch_size)
        self.assertTrue((num_soft_tokens > 0).all())
        self.assertTrue((num_soft_tokens <= self.video_processor_tester.max_soft_tokens).all())
