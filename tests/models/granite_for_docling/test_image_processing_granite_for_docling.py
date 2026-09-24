# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class GraniteForDoclingImageProcessingTester(ImageProcessingTester):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
        crop_to_patches=False,
        min_patches=1,
        max_patches=6,
    ):
        super().__init__()
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.crop_to_patches = crop_to_patches
        self.min_patches = min_patches
        self.max_patches = max_patches

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "crop_to_patches": self.crop_to_patches,
            "min_patches": self.min_patches,
            "max_patches": self.max_patches,
        }

    def prepare_image_inputs(self, **kwargs):
        # One image per sample: the processor expects a list of images per sample
        return [[image] for image in super().prepare_image_inputs(**kwargs)]

    def expected_output_image_shape(self, images):
        # One tile per image when `crop_to_patches` is off, stacked on the tile dimension
        return 1, self.num_channels, self.size["height"], self.size["width"]


@require_torch
@require_vision
class GraniteForDoclingImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = GraniteForDoclingImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processor, "crop_to_patches"))
            self.assertTrue(hasattr(image_processor, "min_patches"))
            self.assertTrue(hasattr(image_processor, "max_patches"))

    def _test_call(self, image_inputs, image_type):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            for sample_images in image_inputs:
                for image in sample_images:
                    self.assertIsInstance(image, image_type)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        self._test_call(
            self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True), np.ndarray
        )

    def test_call_numpy_4_channels(self):
        # Images are always converted to RGB, so the output always has 3 channels
        self._test_call(
            self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True), np.ndarray
        )

    def test_call_pil(self):
        self._test_call(self.image_processor_tester.prepare_image_inputs(equal_resolution=False), Image.Image)

    def test_call_pytorch(self):
        self._test_call(
            self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True), torch.Tensor
        )

    def test_tiles_and_grid(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**{**self.image_processor_dict, "crop_to_patches": True})
            # A 2:1 image is laid out on a 1x2 grid (1 column, 2 rows) and gets a thumbnail
            tall_image = np.random.randint(0, 255, size=(80, 40, 3), dtype=np.uint8)
            encoding = image_processor(tall_image, return_tensors="pt")
            self.assertEqual(encoding["pixel_values"].shape, (1, 3, 3, 20, 20))
            self.assertEqual(encoding["rows"], [[2]])
            self.assertEqual(encoding["cols"], [[1]])
            self.assertNotIn("tile_fine_mask", encoding)

            # A small square image fits in a single tile, which gets no thumbnail
            encoding = image_processor(
                np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8), return_tensors="pt"
            )
            self.assertEqual(encoding["pixel_values"].shape, (1, 1, 3, 20, 20))
            self.assertEqual((encoding["rows"], encoding["cols"]), ([[1]], [[1]]))

    def test_do_pad_pads_the_tile_dimension(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**{**self.image_processor_dict, "crop_to_patches": True})
            tall_image = np.random.randint(0, 255, size=(80, 40, 3), dtype=np.uint8)
            square_image = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
            # 3 tiles and 1 tile: the second sample gets 2 all-zero tiles
            padded = image_processor([[tall_image], [square_image]], return_tensors="pt")["pixel_values"]
            self.assertEqual(padded.shape, (2, 3, 3, 20, 20))
            self.assertTrue(torch.all(padded[1, 1:] == 0))
            # Without padding, samples must have the same number of tiles
            unpadded = image_processor([[square_image], [square_image]], return_tensors="pt", do_pad=False)[
                "pixel_values"
            ]
            self.assertEqual(unpadded.shape, (2, 1, 3, 20, 20))

    def test_grid_side_is_capped(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{**self.image_processor_dict, "crop_to_patches": True, "max_patches": 32}
            )
            # Without the cap, this aspect ratio would select a 1x32 grid, past the largest tile position marker
            wide_image = np.random.randint(0, 255, size=(20, 1000, 3), dtype=np.uint8)
            encoding = image_processor(wide_image, return_tensors="pt")
            self.assertEqual((encoding["rows"], encoding["cols"]), ([[1]], [[16]]))
            self.assertEqual(encoding["pixel_values"].shape, (1, 17, 3, 20, 20))
            self.assertEqual(image_processor.get_number_of_image_patches(20, 1000), 17)
            self.assertEqual(image_processor.get_tile_grid(20, 1000), (1, 16))

    def test_nested_images_are_padded(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**{**self.image_processor_dict, "crop_to_patches": True})
            tall_image = np.random.randint(0, 255, size=(80, 40, 3), dtype=np.uint8)
            square_image = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
            encoding = image_processor(
                [[tall_image, square_image], [square_image]], return_tensors="pt", fine_route=True
            )
            # 3 + 1 tiles for the first sample, 1 for the second, padded with all-zero tiles
            self.assertEqual(encoding["pixel_values"].shape, (2, 4, 3, 20, 20))
            self.assertTrue(torch.all(encoding["pixel_values"][1, 1:] == 0))
            self.assertEqual(encoding["rows"], [[2, 1], [1]])
            self.assertEqual(encoding["cols"], [[1, 1], [1]])
            self.assertEqual(
                encoding["tile_fine_mask"].tolist(), [[True, True, True, True], [True, False, False, False]]
            )

    def test_backends_equivalence_tiled(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        image_processor_dict = {**self.image_processor_dict, "crop_to_patches": True}
        tall_image = np.random.randint(0, 255, size=(80, 40, 3), dtype=np.uint8)
        wide_image = np.random.randint(0, 255, size=(20, 1000, 3), dtype=np.uint8)
        images = [[tall_image, wide_image], [tall_image]]

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**image_processor_dict)
            encodings[backend_name] = image_processor(images, return_tensors="pt", fine_route=True)

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            self._assert_encodings_equivalence(
                encodings[reference_backend], encodings[backend_name], reference_backend, backend_name
            )
