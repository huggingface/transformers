# Copyright 2026 HuggingFace Inc.
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

import math
import unittest

import numpy as np

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, get_image_size
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    from transformers import MiniCPMOcrVideoProcessor


class MiniCPMOcrVideoProcessingTester:
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
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_slice_nums=5,
        scale_resolution=448,
        patch_size=14,
        slice_mode=True,
        downsample_mode="4x",
    ):
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
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.slice_mode = slice_mode
        self.downsample_mode = downsample_mode

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": False,
            "max_slice_nums": self.max_slice_nums,
            "scale_resolution": self.scale_resolution,
            "patch_size": self.patch_size,
            "slice_mode": self.slice_mode,
            "downsample_mode": self.downsample_mode,
        }

    def expected_output_video_shape(self, video_inputs):
        """Return the expected NaViT-packed shape [C, P, total_L] for encoded_videos[0]."""
        total_L = 0
        for video in video_inputs:
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                frames = np.stack([np.array(frame) for frame in video])
            elif hasattr(video, "shape"):
                frames = video
            else:
                frames = np.array(video)

            height, width = get_image_size(frames[0])
            num_frames = len(frames)

            aspect_ratio = width / height
            new_height = int(self.scale_resolution / math.sqrt(aspect_ratio))
            new_width = int(new_height * aspect_ratio)

            # MiniCPM-OCR aligns the resize grid to `patch_size * 2` in `"4x"` mode and
            # `patch_size * 4` in `"16x"` mode, matching the number of 2x2 spatial merges.
            divisor = self.patch_size * (2 if self.downsample_mode == "4x" else 4)
            best_height = max(round(new_height / divisor) * divisor, divisor)
            best_width = max(round(new_width / divisor) * divisor, divisor)

            total_L += num_frames * (best_height * best_width // self.patch_size)

        return [self.num_channels, self.patch_size, total_L]

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
class MiniCPMOcrVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MiniCPMOcrVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = MiniCPMOcrVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.patch_size, 14)

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, patch_size=36)
        self.assertEqual(video_processor.patch_size, 36)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False)

            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )

            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_output_video_shape)

    def test_nested_input(self):
        """NaViT packing: dim 0 is always 1 regardless of batch size."""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            video_inputs = [list(video) for video in video_inputs]

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

    @unittest.skip("NaViT expected_output_video_shape cannot infer channel dim for 4-channel images")
    def test_call_numpy_4_channels(self):
        pass

    def test_resize_alignment_follows_downsample_mode(self):
        """`"4x"` aligns the resize grid to `patch_size * 2`, `"16x"` to `patch_size * 4`."""
        for video_processing_class in self.video_processor_list:
            patch_size = 14
            # 2 * patch_size = 28 is a valid "4x" grid but not a valid "16x" one (needs 56).
            video = torch.zeros(2, 3, 28 * 3, 28 * 3, dtype=torch.uint8)

            video_processing = video_processing_class(
                slice_mode=False,
                scale_resolution=84,
                patch_size=patch_size,
                downsample_mode="4x",
                do_sample_frames=False,
            )
            target_sizes_4x = video_processing(video, return_tensors="pt")["target_sizes_videos"]
            for height, width in target_sizes_4x.tolist():
                self.assertEqual(height % 2, 0)
                self.assertEqual(width % 2, 0)

            # A per-call `downsample_mode` must win over the instance attribute.
            target_sizes_16x = video_processing(video, downsample_mode="16x", return_tensors="pt")[
                "target_sizes_videos"
            ]
            for height, width in target_sizes_16x.tolist():
                self.assertEqual(height % 4, 0)
                self.assertEqual(width % 4, 0)

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )
            video_metadata = [
                VideoMetadata(total_num_frames=len(video), duration=4.0, fps=2.0) for video in video_inputs
            ]
            sampled_video_processor_dict = {**video_processor_dict, "do_sample_frames": True}

            # stack_frames=1: sample one main frame per second from complete metadata.
            video_processor = video_processing_class(**sampled_video_processor_dict, max_num_frames=20, stack_frames=1)

            encoded_videos = video_processor(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            expected_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0][[0, 2, 4, 6]]])
            self.assertEqual(len(encoded_videos), 1)
            self.assertListEqual(list(encoded_videos[0].shape), expected_shape)

            encoded_videos_batched = video_processor(video_inputs, video_metadata=video_metadata, return_tensors="pt")[
                self.input_name
            ]
            expected_shape_batched = self.video_processor_tester.expected_output_video_shape(
                [video[[0, 2, 4, 6]] for video in video_inputs]
            )
            self.assertEqual(len(encoded_videos_batched), 1)
            self.assertListEqual(list(encoded_videos_batched[0].shape), expected_shape_batched)

            # stack_frames=2, duration=4.0: tensor layout = [4 main | 4 sub]
            # Each second gets 1 sub-frame composited (single frame → same size as main)
            # → 8 visual units interleaved: [main_0, comp_0, ..., main_3, comp_3]
            video_processor_sf2 = video_processing_class(
                **sampled_video_processor_dict, max_num_frames=20, stack_frames=2
            )

            encoded_videos_sf2 = video_processor_sf2(
                video_inputs[0],
                video_metadata=[video_metadata[0]],
                return_tensors="pt",
            )[self.input_name]
            self.assertEqual(len(encoded_videos_sf2), 1)
            self.assertListEqual(
                list(encoded_videos_sf2[0].shape),
                self.video_processor_tester.expected_output_video_shape([video_inputs[0]]),
            )

    def test_call_sample_frames_stacked_composites(self):
        # Stage 2 must consume `visual_units` from Stage 1, not the raw sampled frames.
        for video_processing_class in self.video_processor_list:
            video_processor_dict = self.video_processor_dict.copy()
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=True, return_tensors="torch"
            )
            metadata = VideoMetadata(total_num_frames=len(video_inputs[0]), duration=4.0, fps=2.0)
            sampled_dict = {**video_processor_dict, "do_sample_frames": True}

            baseline = video_processing_class(**sampled_dict, max_num_frames=20, stack_frames=1)(
                video_inputs[0], video_metadata=[metadata], return_tensors="pt"
            )
            main_grids = {tuple(size) for size in baseline["target_sizes_videos"].tolist()}
            self.assertEqual(len(main_grids), 1, "all main frames should share one patch grid")
            main_grid = main_grids.pop()

            processor = video_processing_class(**sampled_dict, max_num_frames=20, stack_frames=3)
            out = processor(video_inputs[0], video_metadata=[metadata], return_tensors="pt")

            self.assertEqual(sum(out["num_frames_per_video"]), 8)

            # 8 visual units vs 12 raw frames: a mismatch means Stage 2 read `videos`.
            self.assertEqual(len(out["target_sizes_videos"]), sum(out["num_frames_per_video"]))
            self.assertEqual(len(out["num_patches_per_frame"]), sum(out["num_frames_per_video"]))

            grids = [tuple(size) for size in out["target_sizes_videos"].tolist()]
            self.assertEqual(len(grids), 8)
            for i, grid in enumerate(grids):
                if i % 2 == 0:
                    self.assertEqual(grid, main_grid, f"unit {i} should be a main frame")
                else:
                    self.assertNotEqual(grid, main_grid, f"unit {i} should be a composite canvas")
            self.assertEqual(len(set(grids[1::2])), 1)
