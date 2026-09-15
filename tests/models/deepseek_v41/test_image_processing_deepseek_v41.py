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

import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image, ImageOps


IMAGE_PROCESSOR_DICT = {
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "size": {"shortest_edge": 56 * 56, "longest_edge": 64},
    "patch_size": 14,
    "downsample_ratio": 3,
    "max_wh_ratio": 4,
}


class DeepseekV41ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, batch_size=7, num_channels=3, min_resolution=30, max_resolution=450, **kwargs):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_channels = num_channels
        self.size = dict(IMAGE_PROCESSOR_DICT["size"])

    def prepare_image_processor_dict(self):
        return {**IMAGE_PROCESSOR_DICT, "size": dict(self.size)}


@require_torch
@require_vision
class DeepseekV41ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = DeepseekV41ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_plan_matches_reference_formulas(self):
        from transformers.models.deepseek_v41.image_processing_deepseek_v41 import (
            llm_grid,
            num_image_tokens,
            plan_image_grid,
        )

        self.assertEqual(plan_image_grid(40, 100, 14, 3, 3136, 64), (3, 1, 112, 42))
        self.assertEqual(llm_grid(112, 42, 14, 3), (3, 1))
        self.assertEqual(num_image_tokens(3, 1), 8)
        # Input is (width, height); a portrait stays portrait after upscaling.
        self.assertEqual(plan_image_grid(20, 40, 14, 3, 3136, 64), (2, 1, 84, 42))
        self.assertEqual(plan_image_grid(400, 28, 14, 3, 392, 64, 4), (1, 3, 28, 112))
        for width, height in [(800, 600), (10000, 14), (14, 10000)]:
            n_h, n_w, best_h, best_w = plan_image_grid(width, height, 14, 3, 3136, 64)
            self.assertLessEqual(num_image_tokens(n_h, n_w), 64)
            self.assertEqual((best_h % 14, best_w % 14), (0, 0))
            self.assertGreater(min(best_h, best_w), 0)

    def _check_input_batch(self, images, **kwargs):
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            singles = [processor(image, return_tensors="pt", **kwargs) for image in images]
            expected_pixels = torch.cat([output.pixel_values for output in singles])
            expected_grids = torch.cat([output.image_grid_thw for output in singles])
            for output in singles:
                self.assertEqual(output.pixel_values.shape[0], output.image_grid_thw.prod(-1).sum().item())
            # Interleaved shapes exercise grouping and restoration of complete image blocks.
            for batch in (images, [[images[0]], images[1:]]):
                output = processor(batch, return_tensors="pt", **kwargs)
                torch.testing.assert_close(output.pixel_values, expected_pixels)
                torch.testing.assert_close(output.image_grid_thw, expected_grids)

    def test_call_pil(self):
        self._check_input_batch(
            [
                Image.new("RGB", (28, 28), (20, 40, 60)),
                Image.new("RGB", (56, 28), (100, 110, 120)),
                Image.new("RGB", (28, 28), (220, 230, 240)),
            ]
        )

    def test_call_numpy(self):
        images = [
            np.full((h, w, 3), value, dtype=np.uint8) for h, w, value in [(28, 28, 20), (42, 28, 90), (28, 28, 180)]
        ]
        self._check_input_batch(images)
        self._check_input_batch([image.transpose(2, 0, 1) for image in images], input_data_format="channels_first")

    def test_call_pytorch(self):
        images = [
            torch.full((3, h, w), value, dtype=torch.uint8)
            for h, w, value in [(28, 28, 20), (42, 28, 90), (28, 28, 180)]
        ]
        self._check_input_batch(images)
        self._check_input_batch([image.permute(1, 2, 0) for image in images], input_data_format="channels_last")

    def test_call_numpy_4_channels(self):
        data = np.arange(28 * 28 * 4, dtype=np.uint8).reshape(28, 28, 4)
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            output = processor(
                [data, data],
                do_resize=False,
                do_convert_rgb=False,
                image_mean=[0.0] * 4,
                image_std=[1.0] * 4,
                input_data_format="channels_last",
                return_tensors="np",
            )
            expected = data.transpose(2, 0, 1).reshape(4, 2, 14, 2, 14).transpose(1, 3, 0, 2, 4).reshape(4, -1)
            np.testing.assert_allclose(output.pixel_values, np.tile(expected / 255.0, (2, 1)), atol=1e-6)
            np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 2], [1, 2, 2]])

    def test_patch_layout_is_row_major_channel_major(self):
        data = np.arange(28 * 28 * 3, dtype=np.uint8).reshape(28, 28, 3)
        expected = (
            np.stack(
                [data[h : h + 14, w : w + 14].transpose(2, 0, 1).reshape(-1) for h in (0, 14) for w in (0, 14)]
            ).astype(np.float32)
            / 127.5
            - 1
        )
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            for disable_grouping in (True, False):
                output = processor(
                    [data, data], do_resize=False, disable_grouping=disable_grouping, return_tensors="np"
                )
                np.testing.assert_allclose(output.pixel_values, np.tile(expected, (2, 1)), atol=1e-6)
                np.testing.assert_array_equal(output.image_grid_thw, [[1, 2, 2], [1, 2, 2]])

    def test_rgb_conversion_input_formats(self):
        for mode in ("L", "RGBA"):
            image = Image.new(mode, (28, 28), 75 if mode == "L" else (20, 60, 90, 0))
            array = np.array(image)
            for image_processing_class in self.image_processing_classes.values():
                processor = image_processing_class(**self.image_processor_dict)
                expected = processor(image.convert("RGB"), do_resize=False, return_tensors="pt")
                for source in (image, array, torch.from_numpy(array)):
                    output = processor(source, do_resize=False, return_tensors="pt")
                    torch.testing.assert_close(output.pixel_values, expected.pixel_values)
                    torch.testing.assert_close(output.image_grid_thw, expected.image_grid_thw)

    def test_reference_resize_and_gray_padding(self):
        for width, height, box, max_wh_ratio in [(50, 14, (56, 14), None), (400, 28, (112, 28), 4)]:
            data = np.zeros((height, width, 3), dtype=np.uint8)
            data[..., 0] = np.arange(width, dtype=np.uint16) % 256
            data[..., 1] = 100
            image = Image.fromarray(data)
            reference = image.resize(box) if max_wh_ratio else ImageOps.pad(image, box, color=(127, 127, 127))
            expected = np.array(reference, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
            for backend, image_processing_class in self.image_processing_classes.items():
                processor = image_processing_class(**self.image_processor_dict)
                output = processor(image, min_pixels=392, max_wh_ratio=max_wh_ratio, return_tensors="np")
                grid_h, grid_w = box[1] // 14, box[0] // 14
                pixels = (
                    output.pixel_values.reshape(grid_h, grid_w, 3, 14, 14)
                    .transpose(2, 0, 3, 1, 4)
                    .reshape(3, *expected.shape[1:])
                )
                np.testing.assert_array_equal(output.image_grid_thw, [[1, grid_h, grid_w]])
                np.testing.assert_allclose(pixels, expected, atol=1e-6 if backend == "pil" else 0.1)
                self.assertLessEqual(np.abs(pixels - expected).mean(), 5e-3)
                if max_wh_ratio is None:
                    np.testing.assert_allclose(pixels[:, :, -3:], 127 / 127.5 - 1, atol=1e-6)

    def test_resize_overrides_counts_and_roundtrip(self):
        image = Image.new("RGB", (400, 28), (20, 80, 120))
        overrides = {
            "size": {"shortest_edge": 784, "longest_edge": 32},
            "min_pixels": 392,
            "max_image_tokens": 16,
            "max_wh_ratio": None,
        }
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            expected = processor(image, **overrides, return_tensors="np")
            configured = image_processing_class(**{**self.image_processor_dict, **overrides})
            with tempfile.TemporaryDirectory() as directory:
                configured.save_pretrained(directory)
                reloaded = image_processing_class.from_pretrained(directory)
            output = reloaded(image, return_tensors="np")
            np.testing.assert_allclose(output.pixel_values, expected.pixel_values)
            np.testing.assert_array_equal(output.image_grid_thw, expected.image_grid_thw)
            self.assertEqual(processor.get_number_of_image_patches(28, 400, overrides), expected.pixel_values.shape[0])
            self.assertEqual(processor.get_number_of_image_patches(28, 42, {"do_resize": False}), 6)

    def test_non_patch_aligned_without_resize(self):
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            with self.assertRaises(ValueError):
                processor(Image.new("RGB", (29, 28)), do_resize=False)
            with self.assertRaises(ValueError):
                processor.get_number_of_image_patches(28, 29, {"do_resize": False})
