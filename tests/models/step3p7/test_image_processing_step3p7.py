# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from PIL import Image

from transformers import AutoImageProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


class Step3VisionImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        size=None,
        patch_size=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = size if size is not None else {"height": 728, "width": 728}
        self.patch_size = patch_size if patch_size is not None else {"height": 504, "width": 504}

    def prepare_image_processor_dict(self):
        return {
            "size": self.size,
            "patch_size": self.patch_size,
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
class Step3VisionImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Step3VisionImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_outputs_main_and_patch_sizes(self):
        image = Image.new("RGB", (20, 10), "red")

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_outputs = image_processor(image, return_tensors="pt")
            patch_outputs = image_processor(image, is_patch=True, return_tensors="pt")

            self.assertEqual(tuple(image_outputs["pixel_values"].shape), (1, 3, 728, 728))
            self.assertEqual(tuple(patch_outputs["pixel_values"].shape), (1, 3, 504, 504))

    def test_image_processor_auto_save_load(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)

            with self.subTest(backend=backend_name), self.tmpdirname() as tmp_dir:
                image_processor.save_pretrained(tmp_dir)
                loaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir, backend=backend_name)

            self.assertIsInstance(loaded_image_processor, image_processing_class)
            self.assertEqual(loaded_image_processor.size, {"height": 728, "width": 728})
            self.assertEqual(loaded_image_processor.patch_size, {"height": 504, "width": 504})

    def tmpdirname(self):
        import tempfile

        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
