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

from transformers import is_vision_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    pass


class UnlimitedOcrImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=500,
        max_resolution=800,
        do_resize=True,
        size=None,
        tile_size=384,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"height": 512, "width": 512}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.tile_size = tile_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "tile_size": self.tile_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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


@require_torch
@require_vision
class UnlimitedOcrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = UnlimitedOcrImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass

    def test_preprocess_crop_to_patches(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            num_channels = self.image_processor_tester.num_channels
            size = self.image_processor_tester.size
            tile_size = self.image_processor_tester.tile_size

            image = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)[0]
            processed_image = image_processor.preprocess(images=image, crop_to_patches=True)

            num_local_patches = processed_image["num_local_patches"][0]
            num_columns, num_rows = processed_image["patches_grid"][0]
            self.assertEqual(num_local_patches, num_columns * num_rows)

            self.assertEqual(len(processed_image["pixel_values"]), 1)
            self.assertEqual(processed_image["pixel_values"][0].shape, (num_channels, size["height"], size["width"]))

            self.assertEqual(len(processed_image["pixel_values_local"]), num_local_patches)
            for local_patch in processed_image["pixel_values_local"]:
                self.assertEqual(local_patch.shape, (num_channels, tile_size, tile_size))

    def test_preprocess_no_crop_to_patches(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            num_channels = self.image_processor_tester.num_channels
            size = self.image_processor_tester.size

            image = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)[0]
            processed_image = image_processor.preprocess(images=image, crop_to_patches=False)

            self.assertEqual(processed_image["num_local_patches"][0], 0)
            self.assertNotIn("pixel_values_local", processed_image)

            self.assertEqual(len(processed_image["pixel_values"]), 1)
            self.assertEqual(processed_image["pixel_values"][0].shape, (num_channels, size["height"], size["width"]))
