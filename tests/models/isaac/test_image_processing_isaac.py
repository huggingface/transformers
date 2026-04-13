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

from transformers.models.isaac.image_processing_isaac import get_image_size_for_max_num_patches
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    return Image.new("RGB", size, color=color)


class IsaacImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        patch_size=16,
        max_num_patches=16,
        min_num_patches=4,
        pixel_shuffle_scale=1,
        do_convert_rgb=True,
    ):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.do_convert_rgb = do_convert_rgb

    @property
    def patch_dim(self):
        return self.num_channels * self.patch_size * self.patch_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
            "max_num_patches": self.max_num_patches,
            "min_num_patches": self.min_num_patches,
            "pixel_shuffle_scale": self.pixel_shuffle_scale,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            num_channels=self.num_channels,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]

    def expected_output_image_shape(self, images):
        max_images = 0
        max_patches = 0
        for sample_images in images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]

            max_images = max(max_images, len(sample_images))
            for image in sample_images:
                if isinstance(image, Image.Image):
                    width, height = image.size
                elif isinstance(image, np.ndarray):
                    height, width = image.shape[:2]
                else:
                    height, width = image.shape[-2:]

                target_height, target_width = get_image_size_for_max_num_patches(
                    image_height=height,
                    image_width=width,
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    min_num_patches=self.min_num_patches,
                    pixel_shuffle_scale=self.pixel_shuffle_scale,
                )
                max_patches = max(max_patches, (target_height // self.patch_size) * (target_width // self.patch_size))

        return (max_images, max_patches, self.patch_dim)


@require_torch
@require_vision
class IsaacImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = IsaacImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for sample_images in image_inputs:
                self.assertEqual(len(sample_images), 1)
                self.assertIsInstance(sample_images[0], Image.Image)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for sample_images in image_inputs:
                self.assertEqual(len(sample_images), 1)
                self.assertIsInstance(sample_images[0], np.ndarray)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for sample_images in image_inputs:
                self.assertEqual(len(sample_images), 1)
                self.assertIsInstance(sample_images[0], torch.Tensor)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    @unittest.skip(reason="Isaac image processor 4-channel coverage is not defined")
    def test_call_numpy_4_channels(self):
        pass

    def test_flat_list_is_single_multi_image_sample(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                    "max_num_patches": 64,
                    "min_num_patches": 1,
                    "pixel_shuffle_scale": 1,
                }
            )
            image_inputs = [
                _make_dummy_image(size=(32, 32), color=(255, 0, 0)),
                _make_dummy_image(size=(32, 32), color=(0, 255, 0)),
            ]

            encoding = image_processor(image_inputs, return_tensors="pt")
            self.assertEqual(tuple(encoding["pixel_values"].shape), (1, 2, 4, 768))

            expected_grids = torch.tensor([[[1, 2, 2], [1, 2, 2]]], dtype=torch.long)
            torch.testing.assert_close(encoding["image_grid_thw"], expected_grids)

    def test_nested_multi_image_batch_preserves_grids_and_padding(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                    "max_num_patches": 64,
                    "min_num_patches": 1,
                    "pixel_shuffle_scale": 1,
                }
            )
            image_inputs = [
                [_make_dummy_image(size=(32, 32), color=(255, 0, 0))],
                [
                    _make_dummy_image(size=(48, 32), color=(0, 255, 0)),
                    _make_dummy_image(size=(32, 48), color=(0, 0, 255)),
                ],
            ]

            encoding = image_processor(image_inputs, return_tensors="pt")
            self.assertEqual(tuple(encoding["pixel_values"].shape), (2, 2, 6, 768))

            expected_grids = torch.tensor(
                [
                    [[1, 2, 2], [0, 0, 0]],
                    [[1, 2, 3], [1, 3, 2]],
                ],
                dtype=torch.long,
            )

            torch.testing.assert_close(encoding["image_grid_thw"], expected_grids)
            self.assertTrue(torch.all(encoding["pixel_values"][0, 1] == 0))

    def test_pixel_shuffle_scale_requires_divisible_token_grid(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                    "pixel_shuffle_scale": 2,
                }
            )

            with self.assertRaisesRegex(ValueError, "must be divisible by pixel_shuffle_scale"):
                image_processor([[_make_dummy_image(size=(32, 16))]], return_tensors="pt")
