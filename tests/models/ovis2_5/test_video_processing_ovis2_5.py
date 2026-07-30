# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_video_processing_common import VideoProcessingTestMixin, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


if is_torchvision_available():
    from transformers import Ovis2_5VideoProcessor
    from transformers.models.ovis2_5.image_processing_ovis2_5 import smart_resize


class Ovis2_5VideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_frames=4,
        num_channels=3,
        min_resolution=32,
        max_resolution=96,
        do_resize=True,
        size=None,
        do_rescale=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        do_sample_frames=False,
        patch_size=16,
        temporal_patch_size=1,
        merge_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 32 * 32, "longest_edge": 96 * 96}
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.do_sample_frames = do_sample_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": self.do_sample_frames,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
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

    def expected_output(self, videos, num_frames=None, num_channels=3):
        grids = []
        for video in videos:
            frames = len(video) if num_frames is None else num_frames
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                width, height = video[0].size
            elif isinstance(video, list):
                height, width = video[0].shape[-3:-1]
            elif isinstance(video, torch.Tensor):
                height, width = video.shape[-2:]
            else:
                height, width = video.shape[-3:-1]
            height, width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            grids.append(
                [
                    frames // self.temporal_patch_size,
                    height // self.patch_size,
                    width // self.patch_size,
                ]
            )

        num_patches = sum(np.prod(grid) for grid in grids)
        patch_dim = num_channels * self.temporal_patch_size * self.patch_size**2
        return (num_patches, patch_dim), grids


@require_torch
@require_vision
@require_torchvision
class Ovis2_5VideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = Ovis2_5VideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    def setUp(self):
        super().setUp()
        self.video_processor_tester = Ovis2_5VideoProcessingTester(self)
        self.video_processor = self.fast_video_processing_class(
            size={"shortest_edge": 64 * 96, "longest_edge": 64 * 96}
        )

    @property
    def video_processor_dict(self):
        return self.video_processor_tester.prepare_video_processor_dict()

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, self.video_processor_tester.size)

        overridden_size = {"shortest_edge": 64 * 64, "longest_edge": 128 * 128}
        video_processor = self.fast_video_processing_class.from_dict(
            self.video_processor_dict,
            size=overridden_size,
        )
        self.assertEqual(video_processor.size, overridden_size)

    def _check_input_type(self, video_inputs, **kwargs):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)

            output = video_processor(video_inputs[0], return_tensors="pt", **kwargs)
            expected_shape, expected_grid = self.video_processor_tester.expected_output([video_inputs[0]])
            self.assertEqual(tuple(output[self.input_name].shape), expected_shape)
            self.assertEqual(output.video_grid_thw.tolist(), expected_grid)

            output = video_processor(video_inputs, return_tensors="pt", **kwargs)
            expected_shape, expected_grid = self.video_processor_tester.expected_output(video_inputs)
            self.assertEqual(tuple(output[self.input_name].shape), expected_shape)
            self.assertEqual(output.video_grid_thw.tolist(), expected_grid)

    def test_call_pil(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="pil",
        )
        self.assertTrue(all(isinstance(video[0], Image.Image) for video in video_inputs))
        self._check_input_type(video_inputs)

    def test_call_numpy(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="np",
        )
        self.assertTrue(all(isinstance(video, np.ndarray) for video in video_inputs))
        self._check_input_type(video_inputs)

    def test_call_pytorch(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="torch",
        )
        self.assertTrue(all(isinstance(video, torch.Tensor) for video in video_inputs))
        self._check_input_type(video_inputs)

    def test_nested_input(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="np",
        )
        self._check_input_type([list(video) for video in video_inputs])

    def test_call_numpy_4_channels(self):
        self.video_processor_tester.num_channels = 4
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="np",
        )
        # Ovis2.5 always converts visual inputs to the checkpoint's three-channel RGB representation.
        self._check_input_type(video_inputs, input_data_format="channels_last")

    def test_call_sample_frames(self):
        video_inputs = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False,
            return_tensors="torch",
        )
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)

            output = video_processor(video_inputs, num_frames=2, return_tensors="pt")
            expected_shape, expected_grid = self.video_processor_tester.expected_output(video_inputs)
            self.assertEqual(tuple(output[self.input_name].shape), expected_shape)
            self.assertEqual(output.video_grid_thw.tolist(), expected_grid)

            video_processor.do_sample_frames = True
            output = video_processor(video_inputs, num_frames=2, return_tensors="pt")
            expected_shape, expected_grid = self.video_processor_tester.expected_output(video_inputs, num_frames=2)
            self.assertEqual(tuple(output[self.input_name].shape), expected_shape)
            self.assertEqual(output.video_grid_thw.tolist(), expected_grid)

    def test_native_video_defaults(self):
        self.assertEqual(self.video_processor.patch_size, 16)
        self.assertEqual(self.video_processor.temporal_patch_size, 1)
        self.assertEqual(self.video_processor.merge_size, 2)
        self.assertFalse(self.video_processor.do_sample_frames)

    def test_exact_video_grid_and_packing(self):
        video = np.zeros((3, 64, 96, 3), dtype=np.uint8)
        output = self.video_processor(video, return_tensors="pt")

        self.assertEqual(tuple(output.pixel_values_videos.shape), (72, 768))
        self.assertEqual(output.video_grid_thw.tolist(), [[3, 4, 6]])
        self.assertEqual(
            self.video_processor.get_number_of_video_patches(3, 64, 96),
            72,
        )

    def test_rejects_incompatible_temporal_patches(self):
        video = np.zeros((3, 64, 96, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "temporal_patch_size=1"):
            self.video_processor(video, temporal_patch_size=2)
        with self.assertRaisesRegex(ValueError, "temporal_patch_size=1"):
            self.video_processor.get_number_of_video_patches(
                3,
                64,
                96,
                {"temporal_patch_size": 2},
            )

    def test_rejects_zero_frame_video(self):
        video = np.zeros((0, 64, 96, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "zero frames|at least one"):
            self.video_processor(video)


if __name__ == "__main__":
    unittest.main()
