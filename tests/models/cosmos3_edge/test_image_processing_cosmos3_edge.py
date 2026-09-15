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

"""Tests for the Cosmos3 Edge image processors."""

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin, prepare_image_inputs


class Cosmos3EdgeImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("min_resolution", 32)
        kwargs.setdefault("max_resolution", 64)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 32 * 32, "longest_edge": 64 * 64})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_convert_rgb", True)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("merge_size", 2)
        super().__init__(parent, **kwargs)

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        """Wrap one image per sample to exercise Edge's nested multimodal input form."""
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]


@require_torch
@require_torchvision
@require_vision
class Cosmos3EdgeImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Cosmos3EdgeImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def assert_packed_output(self, output, batch_size):
        """Check Edge's flattened patch matrix against its per-image THW grids."""
        expected_num_patches = int(output.image_grid_thw.prod(dim=-1).sum())
        expected_patch_width = self.image_processor_tester.num_channels * self.image_processor_tester.patch_size**2

        self.assertEqual(output.image_grid_thw.shape[0], batch_size)
        self.assertEqual(tuple(output.pixel_values.shape), (expected_num_patches, expected_patch_width))

    def test_image_processor_properties(self):
        """Cover the patch and merge settings added by the Edge image processor."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            for attribute in (
                "do_resize",
                "size",
                "do_normalize",
                "image_mean",
                "image_std",
                "do_convert_rgb",
                "patch_size",
                "merge_size",
            ):
                self.assertTrue(hasattr(image_processor, attribute))

    def test_image_processor_from_dict_with_kwargs(self):
        """Ensure Edge's pixel-budget size dictionary can be overridden on loading."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 32 * 32, "longest_edge": 64 * 64})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict,
                size={"shortest_edge": 64 * 64, "longest_edge": 96 * 96},
            )
            self.assertEqual(image_processor.size, {"shortest_edge": 64 * 64, "longest_edge": 96 * 96})

    def test_call_pil(self):
        """Adapt the shared PIL test to Edge's packed patch output layout."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            output = image_processor(image_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = image_processor(image_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.image_processor_tester.batch_size)

    def test_call_numpy(self):
        """Adapt the shared NumPy test to Edge's packed patch output layout."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            output = image_processor(image_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = image_processor(image_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.image_processor_tester.batch_size)

    def test_call_pytorch(self):
        """Adapt the shared PyTorch test to Edge's packed patch output layout."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            output = image_processor(image_inputs[0], return_tensors="pt")
            self.assert_packed_output(output, batch_size=1)

            output = image_processor(image_inputs, return_tensors="pt")
            self.assert_packed_output(output, batch_size=self.image_processor_tester.batch_size)

    @unittest.skip(reason="Cosmos3EdgeImageProcessor converts inputs to RGB")
    def test_call_numpy_4_channels(self):
        pass
