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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class LocateAnythingImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        min_resolution=28,
        max_resolution=84,
        patch_size=14,
        merge_kernel_size=(2, 2),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.patch_size = patch_size
        self.merge_kernel_size = merge_kernel_size
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "patch_size": self.patch_size,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

    def expected_num_patches(self, image):
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            height, width = image.shape[-2:]
        pad = self.merge_kernel_size[0] * self.patch_size
        target_h = -(-height // pad) * pad
        target_w = -(-width // pad) * pad
        return (target_h // self.patch_size) * (target_w // self.patch_size)


@require_torch
@require_vision
class LocateAnythingImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = LocateAnythingImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_kernel_size"))

    def _check_packed_output(self, image_inputs, image_processing):
        patch = self.image_processor_tester.patch_size

        process_out = image_processing(image_inputs[0], return_tensors="pt")
        expected = self.image_processor_tester.expected_num_patches(image_inputs[0])
        self.assertEqual(tuple(process_out.pixel_values.shape), (expected, 3, patch, patch))
        self.assertEqual(tuple(process_out.image_grid_thw.shape), (1, 3))

        process_out = image_processing(image_inputs, return_tensors="pt")
        expected = sum(self.image_processor_tester.expected_num_patches(image) for image in image_inputs)
        self.assertEqual(tuple(process_out.pixel_values.shape), (expected, 3, patch, patch))
        self.assertEqual(tuple(process_out.image_grid_thw.shape), (self.image_processor_tester.batch_size, 3))

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)
            self._check_packed_output(image_inputs, image_processing)

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)
            self._check_packed_output(image_inputs, image_processing)

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)
            self._check_packed_output(image_inputs, image_processing)

    def test_normalization_range(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            pixel_values = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertGreaterEqual(float(pixel_values.min()), -1.0 - 1e-4)
            self.assertLessEqual(float(pixel_values.max()), 1.0 + 1e-4)

    @unittest.skip(reason="LocateAnything packs patches, so the standard 4-channel call shape does not apply")
    def test_call_numpy_4_channels(self):
        pass
