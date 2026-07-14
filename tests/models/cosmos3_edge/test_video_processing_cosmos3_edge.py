# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the Cosmos3 Edge video processor."""

import unittest

import numpy as np

from transformers import Cosmos3EdgeVideoProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_vision_available():
    from PIL import Image


class Cosmos3EdgeVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_frames=4,
        num_channels=3,
        min_resolution=32,
        max_resolution=64,
        size=None,
        do_resize=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        temporal_patch_size=1,
        patch_size=16,
        merge_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = (
            size
            if size is not None
            else {
                "shortest_edge": 32 * 32,
                "longest_edge": num_frames * 64 * 64,
            }
        )
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": False,
            "temporal_patch_size": self.temporal_patch_size,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
        }

    def prepare_video_inputs(self, equal_resolution=False, return_tensors="pil"):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            return_tensors=return_tensors,
        )

    def prepare_video_metadata(self, videos):
        return [{"fps": 2, "total_num_frames": len(video), "duration": len(video) / 2} for video in videos]


@require_torch
@require_torchvision
@require_vision
class Cosmos3EdgeVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Cosmos3EdgeVideoProcessor

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Cosmos3EdgeVideoProcessingTester(self)

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def assert_packed_output(self, output, batch_size):
        expected_num_patches = int(output.video_grid_thw.prod(dim=-1).sum())
        expected_patch_width = self.video_processor_tester.num_channels * self.video_processor_tester.patch_size**2

        self.assertEqual(output.video_grid_thw.shape[0], batch_size)
        self.assertEqual(tuple(output.pixel_values_videos.shape), (expected_num_patches, expected_patch_width))

    def test_video_processor_properties(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        for attribute in (
            "do_resize",
            "size",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "temporal_patch_size",
            "patch_size",
            "merge_size",
        ):
            self.assertTrue(hasattr(video_processor, attribute))

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(
            video_processor.size,
            {"shortest_edge": 32 * 32, "longest_edge": self.video_processor_tester.num_frames * 64 * 64},
        )

        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict,
            size={"shortest_edge": 64 * 64, "longest_edge": 8 * 96 * 96},
        )
        self.assertEqual(video_processor.size, {"shortest_edge": 64 * 64, "longest_edge": 8 * 96 * 96})

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=True)
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            output = video_processor(video_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = video_processor(video_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.video_processor_tester.batch_size)

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=True, return_tensors="np")

            output = video_processor(video_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = video_processor(video_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.video_processor_tester.batch_size)

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=True, return_tensors="torch"
            )

            output = video_processor(video_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = video_processor(video_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.video_processor_tester.batch_size)

    def test_nested_input(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=True, return_tensors="np")
            nested_video_inputs = [list(video) for video in video_inputs]

            output = video_processor(nested_video_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = video_processor(nested_video_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.video_processor_tester.batch_size)

    def test_call_sample_frames(self):
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=True, return_tensors="torch")
        metadata = self.video_processor_tester.prepare_video_metadata(video_inputs)

        output = video_processor(
            video_inputs,
            video_metadata=metadata,
            do_sample_frames=True,
            num_frames=2,
            fps=None,
            return_tensors="pt",
        )

        self.assertEqual(output.video_grid_thw[:, 0].tolist(), [2] * self.video_processor_tester.batch_size)
        self.assert_packed_output(output, batch_size=self.video_processor_tester.batch_size)

    @unittest.skip(reason="Cosmos3EdgeVideoProcessor converts inputs to RGB")
    def test_call_numpy_4_channels(self):
        pass

    def test_temporal_patch_size_is_fixed_to_one(self):
        with self.assertRaisesRegex(ValueError, "temporal_patch_size=1"):
            Cosmos3EdgeVideoProcessor(temporal_patch_size=2)

    def test_samples_two_frames_per_second_by_default(self):
        processor = Cosmos3EdgeVideoProcessor()
        metadata = VideoMetadata(total_num_frames=8, fps=4, duration=2.0)

        indices = processor.sample_frames(metadata)

        self.assertEqual(indices.tolist(), [0, 2, 5, 7])

    def test_video_processor_preserves_one_grid_step_per_frame(self):
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.zeros((3, 4, 8, 3), dtype=np.uint8)

        output = processor(
            video,
            video_metadata=[{"fps": 2, "total_num_frames": 3, "duration": 1.5}],
            return_tensors="pt",
        )

        self.assertEqual(output.video_grid_thw.tolist(), [[3, 2, 4]])
        self.assertEqual(tuple(output.pixel_values_videos.shape), (24, 12))
