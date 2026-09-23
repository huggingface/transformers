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
            self.assertEqual(encoded.shape[0], self.video_processor_tester.num_frames)

            video_processing.do_sample_frames = True
            encoded = video_processing(video_inputs[0], return_tensors="pt", fps=None, max_frames=3)[self.input_name]
            encoded_batched = video_processing(video_inputs, return_tensors="pt", fps=None, max_frames=3)[
                self.input_name
            ]
            self.assertEqual(encoded.shape[0], 3)
            self.assertEqual(encoded_batched.shape[0], len(video_inputs) * 3)

    def test_video_processor_from_dict_with_kwargs(self):
        """EmbeddingGemma 2 has no `size`/`crop_size`; override with patch budget kwargs instead."""
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.patch_size, self.video_processor_tester.patch_size)
        self.assertEqual(video_processor.max_soft_tokens, self.video_processor_tester.max_soft_tokens)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=18)
        self.assertEqual(video_processor.patch_size, 18)

    def test_sample_frames_1fps_linear(self):
        """`fps=1` takes one frame per second; `overflow_strategy="uniform"` then linspace-subsamples
        the per-second indices down to `max_frames`."""
        processor = self.fast_video_processing_class()

        # 10 seconds at 25 fps: one frame per second, all kept (under budget, so the cap is a no-op).
        meta_short = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)
        sampled_short = processor.sample_frames(meta_short, fps=1, max_frames=32, overflow_strategy="uniform")
        expected_short = np.array([int(s * 25) for s in range(10)])
        np.testing.assert_array_equal(sampled_short, expected_short)

        # 100 seconds at 25 fps: per-second indices subsampled down to `max_frames`.
        meta_long = VideoMetadata(fps=25.0, total_num_frames=2500, duration=100.0)
        sampled_long = processor.sample_frames(meta_long, fps=1, max_frames=32, overflow_strategy="uniform")
        self.assertEqual(len(sampled_long), 32)
        sec_indices_long = [int(s * 25) for s in range(100)]
        expected_long = np.array([sec_indices_long[i] for i in np.linspace(0, 99, 32, dtype=int)])
        np.testing.assert_array_equal(sampled_long, expected_long)

        # `metadata.fps` is not recoverable, so rate-based sampling is skipped and every frame is kept.
        meta_missing = VideoMetadata(fps=None, total_num_frames=25)
        np.testing.assert_array_equal(processor.sample_frames(meta_missing, fps=1), np.arange(25))

        # Opting out falls back to the base uniform sampling.
        self.assertEqual(len(processor.sample_frames(meta_short, overflow_strategy="uniform", max_frames=2)), 2)

    def test_sample_frames_under_budget_is_not_upsampled(self):
        """Regression: `uniform` must not pad a short video up to `max_frames` by repeating indices."""
        processor = self.fast_video_processing_class()
        meta = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)

        sampled = processor.sample_frames(meta, fps=1, max_frames=32, overflow_strategy="uniform")
        self.assertEqual(len(sampled), 10)
        self.assertEqual(len(np.unique(sampled)), 10)

        # Same guarantee through `preprocess`, where the class defaults supply `overflow_strategy`.
        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_processing.do_sample_frames = True
        video = torch.randint(0, 255, (10, 3, 32, 32), dtype=torch.uint8)
        encoded = video_processing(
            video,
            video_metadata=[VideoMetadata(fps=1.0, total_num_frames=10, duration=10.0)],
            return_tensors="pt",
        )[self.input_name]
        self.assertEqual(encoded.shape[0], 10)

    def test_sample_frames_truncate_keeps_the_first_frames(self):
        processor = self.fast_video_processing_class()
        meta = VideoMetadata(fps=25.0, total_num_frames=2500, duration=100.0)

        sampled = processor.sample_frames(meta, fps=1, max_frames=32, overflow_strategy="truncate")
        np.testing.assert_array_equal(sampled, np.array([int(s * 25) for s in range(32)]))

        # Under budget, truncation is a no-op rather than a pad.
        meta_short = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)
        self.assertEqual(
            len(processor.sample_frames(meta_short, fps=1, max_frames=32, overflow_strategy="truncate")), 10
        )

    def test_sample_frames_rejects_num_frames(self):
        """EmbeddingGemma 2 does not implement the `num_frames` contract: it must fail loudly rather than
        silently sample by `fps` instead (a stale `num_frames` in an exported config lands here too)."""
        processor = self.fast_video_processing_class()
        meta = VideoMetadata(fps=25.0, total_num_frames=250, duration=10.0)
        with self.assertRaises(ValueError):
            processor.sample_frames(meta, num_frames=8)

        video_processing = self.fast_video_processing_class(**self.video_processor_dict)
        video_processing.do_sample_frames = True
        video = torch.randint(0, 255, (10, 3, 32, 32), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            video_processing(
                video,
                num_frames=8,
                video_metadata=[VideoMetadata(fps=1.0, total_num_frames=10, duration=10.0)],
                return_tensors="pt",
            )

    def test_sample_frames_invalid_arguments_raise(self):
        processor = self.fast_video_processing_class()
        meta = VideoMetadata(fps=25.0, total_num_frames=2500, duration=100.0)

        # Unknown strategy: must not silently skip the cap.
        with self.assertRaises(ValueError):
            processor.sample_frames(meta, fps=1, max_frames=32, overflow_strategy="unifrom")

        # A strategy without a budget is meaningless.
        with self.assertRaises(ValueError):
            processor.sample_frames(meta, fps=1, max_frames=None, overflow_strategy="uniform")

    def test_sample_frames_incomplete_metadata_falls_back_to_cap_only(self):
        """A decoded array has no frame rate, so FPS sampling cannot apply. Rather than raise or guess a
        source rate, the request degrades to the `max_frames` budget alone and every frame is kept."""
        processor = self.fast_video_processing_class()

        # `fps` present but `duration` missing, and vice versa: neither is enough on its own.
        for meta in (
            VideoMetadata(fps=25.0, total_num_frames=20),
            VideoMetadata(total_num_frames=20, duration=100.0),
            VideoMetadata(total_num_frames=20),
        ):
            indices = processor.sample_frames(meta, fps=1, max_frames=32, overflow_strategy="uniform")
            np.testing.assert_array_equal(indices, np.arange(20))

        # The budget is still enforced on the kept frames.
        meta = VideoMetadata(total_num_frames=100)
        indices = processor.sample_frames(meta, fps=1, max_frames=4, overflow_strategy="uniform")
        np.testing.assert_array_equal(indices, np.array([0, 33, 66, 99]))

    def test_sample_frames_without_fps_keeps_all_frames_then_caps(self):
        """`fps=None` is the documented recipe for decoded arrays with no metadata: no FPS sampling, but
        the `max_frames` budget still applies."""
        processor = self.fast_video_processing_class()
        meta = VideoMetadata(total_num_frames=100)

        np.testing.assert_array_equal(processor.sample_frames(meta, fps=None), np.arange(100))
        np.testing.assert_array_equal(
            processor.sample_frames(meta, fps=None, max_frames=4, overflow_strategy="uniform"),
            np.array([0, 33, 66, 99]),
        )
        np.testing.assert_array_equal(
            processor.sample_frames(meta, fps=None, max_frames=4, overflow_strategy="truncate"),
            np.array([0, 1, 2, 3]),
        )

    def test_batch_ragged_slices_match_single_video(self):
        """Each video's slice of a ragged batch is identical to processing that video alone."""
        processor = self.fast_video_processing_class(**self.video_processor_dict)
        videos = [
            torch.randint(0, 255, (2, 3, 40, 40), dtype=torch.uint8),
            torch.randint(0, 255, (5, 3, 60, 40), dtype=torch.uint8),
        ]
        batched = processor(videos, return_tensors="pt", do_sample_frames=False)

        offset = 0
        for video, num_frames in zip(videos, batched.num_frames_per_video):
            alone = processor(video, return_tensors="pt", do_sample_frames=False)
            torch.testing.assert_close(
                batched.pixel_values_videos[offset : offset + num_frames], alone.pixel_values_videos
            )
            torch.testing.assert_close(
                batched.video_position_ids[offset : offset + num_frames], alone.video_position_ids
            )
            offset += num_frames

    def assert_expected_videos_shape(self, encoded_videos, expected_output_video_shape, num_videos):
        """The frames of every video are concatenated along one axis instead of stacked on a `num_videos` axis."""
        num_frames, *rest = expected_output_video_shape
        self.assertEqual(tuple(encoded_videos.shape), (num_videos * num_frames, *rest))
