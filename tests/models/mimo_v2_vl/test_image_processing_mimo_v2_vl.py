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

from transformers import MiMoV2ImageProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class MiMoV2ImageProcessorTest(unittest.TestCase):
    def test_grouping_preserves_mixed_size_batch_order(self):
        processor = MiMoV2ImageProcessor(patch_size=2, merge_size=2, size={"shortest_edge": 16, "longest_edge": 256})
        rng = np.random.default_rng(0)
        images = [
            rng.integers(0, 256, (3, height, width), dtype=np.uint8) for height, width in [(7, 11), (11, 7), (7, 11)]
        ]
        kwargs = {"input_data_format": "channels_first", "return_tensors": "np"}
        batched = processor(images, disable_grouping=False, **kwargs)
        individual = [processor(image, **kwargs) for image in images]
        np.testing.assert_array_equal(batched.pixel_values, np.concatenate([out.pixel_values for out in individual]))
        np.testing.assert_array_equal(
            batched.image_grid_thw, np.concatenate([out.image_grid_thw for out in individual])
        )

    def test_resize_uses_float_pixels_without_antialiasing(self):
        processor = MiMoV2ImageProcessor(
            patch_size=2,
            merge_size=2,
            size={"shortest_edge": 16, "longest_edge": 16},
            do_rescale=False,
            do_normalize=False,
        )
        image = np.zeros((3, 8, 8), dtype=np.uint8)
        image[:, 1::2] = 255
        output = processor(image, input_data_format="channels_first", return_tensors="np")
        # Bilinear 2x downsampling averages each pair of rows without uint8 rounding or boundary filtering.
        np.testing.assert_array_equal(output.pixel_values, np.full((4, 24), 127.5, dtype=np.float32))
        np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 2]])

    def test_small_image_is_scaled_to_a_merge_block(self):
        processor = MiMoV2ImageProcessor(patch_size=2, merge_size=2, size={"shortest_edge": 16, "longest_edge": 256})
        image = np.zeros((3, 1, 5), dtype=np.uint8)
        output = processor(image, input_data_format="channels_first", return_tensors="np")
        np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 10]])
        self.assertEqual(processor.get_number_of_image_patches(1, 5), 20)

    def test_float32_normalization_independent_of_default_dtype(self):
        import torch

        processor = MiMoV2ImageProcessor(patch_size=2, merge_size=2, do_resize=False)
        image = np.arange(3 * 4 * 8, dtype=np.uint8).reshape(3, 4, 8)
        kwargs = {"input_data_format": "channels_first", "return_tensors": "pt"}
        expected = processor(image, **kwargs).pixel_values
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            actual = processor(image, **kwargs).pixel_values
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spatial_merge_and_repeated_temporal_patch_order(self):
        processor = MiMoV2ImageProcessor(
            patch_size=2, merge_size=2, temporal_patch_size=2, do_resize=False, do_rescale=False, do_normalize=False
        )
        image = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
        output = processor(image, input_data_format="channels_first", return_tensors="np")
        coordinates = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 4), (0, 6), (2, 4), (2, 6)]
        expected = np.stack(
            [np.stack([image[:, row : row + 2, col : col + 2]] * 2, axis=1).reshape(-1) for row, col in coordinates]
        )
        np.testing.assert_array_equal(output.pixel_values, expected)
        np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 4]])
