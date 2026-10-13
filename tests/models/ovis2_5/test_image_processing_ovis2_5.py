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

from transformers.models.ovis2_5.image_processing_pil_ovis2_5 import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Ovis2_5ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("min_resolution", 32)
        kwargs.setdefault("max_resolution", 96)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 32 * 32, "longest_edge": 96 * 96})
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_convert_rgb", True)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("merge_size", 2)
        super().__init__(**kwargs)

    def expected_output(self, images):
        grids = []
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
            else:
                height, width = image.shape[-3:-1]
            height, width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            grids.append([1, height // self.patch_size, width // self.patch_size])

        num_patches = sum(np.prod(grid) for grid in grids)
        patch_dim = self.num_channels * self.temporal_patch_size * self.patch_size**2
        return (num_patches, patch_dim), grids

    def expected_output_image_shape(self, images):
        return self.expected_output(images)[0]


@require_torch
@require_vision
class Ovis2_5ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Ovis2_5ImageProcessingTester

    def _check_input_type(self, image_inputs, **kwargs):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)

            output = image_processor(image_inputs[0], return_tensors="pt", **kwargs)
            expected_shape, expected_grid = self.image_processor_tester.expected_output([image_inputs[0]])
            self.assertEqual(tuple(output.pixel_values.shape), expected_shape)
            self.assertEqual(output.image_grid_thw.tolist(), expected_grid)

            output = image_processor(image_inputs, return_tensors="pt", **kwargs)
            expected_shape, expected_grid = self.image_processor_tester.expected_output(image_inputs)
            self.assertEqual(tuple(output.pixel_values.shape), expected_shape)
            self.assertEqual(output.image_grid_thw.tolist(), expected_grid)

    def test_call_pil(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        self.assertTrue(all(isinstance(image, Image.Image) for image in image_inputs))
        self._check_input_type(image_inputs)

    def test_call_numpy(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        self.assertTrue(all(isinstance(image, np.ndarray) for image in image_inputs))
        self._check_input_type(image_inputs)

    def test_call_pytorch(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        self.assertTrue(all(isinstance(image, torch.Tensor) for image in image_inputs))
        self._check_input_type(image_inputs)

    @unittest.skip("Ovis2.5 checkpoints require three-channel RGB patch vectors")
    def test_call_numpy_4_channels(self):
        pass

    def test_smart_resize_matches_native_ovis_policy(self):
        self.assertEqual(smart_resize(448, 448), (448, 448))
        self.assertEqual(smart_resize(333, 527), (384, 576))
        self.assertEqual(smart_resize(20, 100), (224, 1024))
        self.assertEqual(smart_resize(100, 25_000), (96, 20_000))
        self.assertEqual(smart_resize(4_000, 4_000), (1536, 1536))
        self.assertEqual(
            smart_resize(4_000, 4_000, max_pixels=1792 * 1792),
            (1792, 1792),
        )

    def test_patch_order_matches_native_ovis(self):
        rows, columns = np.indices((32, 32))
        image = np.stack((rows * 7 % 256, columns * 5 % 256, (rows * 11 + columns * 13) % 256), axis=-1).astype(
            np.uint8
        )
        normalized = image.astype(np.float32) / 127.5 - 1.0
        expected = np.stack(
            (
                normalized[:16, :16],
                normalized[:16, 16:],
                normalized[16:, :16],
                normalized[16:, 16:],
            )
        )
        expected = expected.transpose(0, 3, 1, 2).reshape(4, 768)

        for image_processing_class in self.image_processing_classes.values():
            with self.subTest(image_processing_class=image_processing_class.__name__):
                processor = image_processing_class(size={"shortest_edge": 32 * 32, "longest_edge": 32 * 32})
                output = processor(image, return_tensors="np")
                np.testing.assert_array_equal(output.image_grid_thw, np.array([[1, 2, 2]]))
                np.testing.assert_allclose(output.pixel_values, expected, atol=1e-6, rtol=0)
