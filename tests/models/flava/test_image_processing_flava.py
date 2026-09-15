# Copyright 2022 Meta Platforms authors and HuggingFace Inc.
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

import random
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTester,
    ImageProcessingTestMixin,
    load_coco_image,
)


if is_torch_available():
    import torch

if is_vision_available():
    import PIL

    from transformers.image_utils import PILImageResampling
    from transformers.models.flava.image_processing_flava import (
        FLAVA_CODEBOOK_MEAN,
        FLAVA_CODEBOOK_STD,
        FLAVA_IMAGE_MEAN,
        FLAVA_IMAGE_STD,
    )
else:
    FLAVA_IMAGE_MEAN = FLAVA_IMAGE_STD = FLAVA_CODEBOOK_MEAN = FLAVA_CODEBOOK_STD = None


class FlavaImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_mean", FLAVA_IMAGE_MEAN)
        kwargs.setdefault("image_std", FLAVA_IMAGE_STD)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 224, "width": 224})
        kwargs.setdefault("resample", PILImageResampling.BICUBIC)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": 224, "width": 224})
        kwargs.setdefault("input_size_patches", 14)
        kwargs.setdefault("total_mask_patches", 75)
        kwargs.setdefault("mask_group_min_patches", 16)
        kwargs.setdefault("mask_group_min_aspect_ratio", 0.3)
        kwargs.setdefault("mask_group_max_aspect_ratio", 0.3)
        kwargs.setdefault("codebook_do_resize", True)
        kwargs.setdefault("codebook_size", {"height": 112, "width": 112})
        # LANCZOS resample is natively supported with torchvision >= 0.27.
        # On older versions, the base class falls back to BICUBIC automatically.
        kwargs.setdefault("codebook_resample", PILImageResampling.LANCZOS)
        kwargs.setdefault("codebook_do_center_crop", True)
        kwargs.setdefault("codebook_crop_size", {"height": 112, "width": 112})
        kwargs.setdefault("codebook_do_map_pixels", True)
        kwargs.setdefault("codebook_do_normalize", True)
        kwargs.setdefault("codebook_image_mean", FLAVA_CODEBOOK_MEAN)
        kwargs.setdefault("codebook_image_std", FLAVA_CODEBOOK_STD)
        super().__init__(parent, **kwargs)

    def get_expected_image_size(self):
        return (self.size["height"], self.size["width"])

    def get_expected_mask_size(self):
        return (
            (self.input_size_patches, self.input_size_patches)
            if not isinstance(self.input_size_patches, tuple)
            else self.input_size_patches
        )

    def get_expected_codebook_image_size(self):
        return (self.codebook_size["height"], self.codebook_size["width"])

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]


@require_torch
@require_vision
class FlavaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    maxDiff = None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = FlavaImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "crop_size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "masking_generator"))
            self.assertTrue(hasattr(image_processing, "codebook_do_resize"))
            self.assertTrue(hasattr(image_processing, "codebook_size"))
            self.assertTrue(hasattr(image_processing, "codebook_resample"))
            self.assertTrue(hasattr(image_processing, "codebook_do_center_crop"))
            self.assertTrue(hasattr(image_processing, "codebook_crop_size"))
            self.assertTrue(hasattr(image_processing, "codebook_do_map_pixels"))
            self.assertTrue(hasattr(image_processing, "codebook_do_normalize"))
            self.assertTrue(hasattr(image_processing, "codebook_image_mean"))
            self.assertTrue(hasattr(image_processing, "codebook_image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 224, "width": 224})
            self.assertEqual(image_processor.crop_size, {"height": 224, "width": 224})
            self.assertEqual(image_processor.codebook_size, {"height": 112, "width": 112})
            self.assertEqual(image_processor.codebook_crop_size, {"height": 112, "width": 112})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size=42, crop_size=84, codebook_size=33, codebook_crop_size=66
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})
            self.assertEqual(image_processor.codebook_size, {"height": 33, "width": 33})
            self.assertEqual(image_processor.codebook_crop_size, {"height": 66, "width": 66})

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, PIL.Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt")

            # Test no bool masked pos
            self.assertFalse("bool_masked_pos" in encoded_images)

            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()

            self.assertEqual(
                encoded_images.pixel_values.shape,
                (1, self.image_processor_tester.num_channels, expected_height, expected_width),
            )

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt")
            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()

            # Test no bool masked pos
            self.assertFalse("bool_masked_pos" in encoded_images)

            self.assertEqual(
                encoded_images.pixel_values.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    expected_height,
                    expected_width,
                ),
            )

    def _test_call_framework(self, instance_class, prepare_kwargs):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, **prepare_kwargs)
            for image in image_inputs:
                self.assertIsInstance(image, instance_class)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt")

            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
            self.assertEqual(
                encoded_images.pixel_values.shape,
                (1, self.image_processor_tester.num_channels, expected_height, expected_width),
            )

            encoded_images = image_processing(image_inputs, return_image_mask=True, return_tensors="pt")

            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
            self.assertEqual(
                encoded_images.pixel_values.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    expected_height,
                    expected_width,
                ),
            )

            expected_height, expected_width = self.image_processor_tester.get_expected_mask_size()
            self.assertEqual(
                encoded_images.bool_masked_pos.shape,
                (
                    self.image_processor_tester.batch_size,
                    expected_height,
                    expected_width,
                ),
            )

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
            self.assertEqual(
                encoded_images.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    expected_height,
                    expected_width,
                ),
            )

            # Test masking
            encoded_images = image_processing(image_inputs, return_image_mask=True, return_tensors="pt")

            expected_height, expected_width = self.image_processor_tester.get_expected_image_size()
            self.assertEqual(
                encoded_images.pixel_values.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    expected_height,
                    expected_width,
                ),
            )

            expected_height, expected_width = self.image_processor_tester.get_expected_mask_size()
            self.assertEqual(
                encoded_images.bool_masked_pos.shape,
                (
                    self.image_processor_tester.batch_size,
                    expected_height,
                    expected_width,
                ),
            )

    def test_call_numpy(self):
        self._test_call_framework(np.ndarray, prepare_kwargs={"numpify": True})

    def test_call_numpy_4_channels(self):
        # Get the first backend class to modify num_channels
        first_backend_class = list(self.image_processing_classes.values())[0]
        original_num_channels = (
            first_backend_class.num_channels if hasattr(first_backend_class, "num_channels") else None
        )
        first_backend_class.num_channels = 4
        self._test_call_framework(np.ndarray, prepare_kwargs={"numpify": True})
        if original_num_channels is not None:
            first_backend_class.num_channels = original_num_channels
        else:
            delattr(first_backend_class, "num_channels")

    def test_call_pytorch(self):
        self._test_call_framework(torch.Tensor, prepare_kwargs={"torchify": True})

    def test_masking(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            random.seed(1234)
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_image_mask=True, return_tensors="pt")
            self.assertEqual(encoded_images.bool_masked_pos.sum().item(), 75)

    def test_codebook_pixels(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, PIL.Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_codebook_pixels=True, return_tensors="pt")
            expected_height, expected_width = self.image_processor_tester.get_expected_codebook_image_size()
            self.assertEqual(
                encoded_images.codebook_pixel_values.shape,
                (1, self.image_processor_tester.num_channels, expected_height, expected_width),
            )

            # Test batched
            encoded_images = image_processing(image_inputs, return_codebook_pixels=True, return_tensors="pt")
            expected_height, expected_width = self.image_processor_tester.get_expected_codebook_image_size()
            self.assertEqual(
                encoded_images.codebook_pixel_values.shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    expected_height,
                    expected_width,
                ),
            )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_coco_image("000000039769.jpg")

        # Create processors for each backend
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(
                dummy_image, return_tensors="pt", return_codebook_pixels=True, return_image_mask=True
            )

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            self._assert_tensors_equivalence(
                reference_encoding.codebook_pixel_values, encodings[backend_name].codebook_pixel_values
            )
