# Copyright 2023 HuggingFace Inc.
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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import (
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
)

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import EfficientNetImageProcessor

    if is_torchvision_available():
        from transformers import EfficientNetImageProcessorFast


class EfficientNetImageProcessorTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_offset=True,
        rescale_factor=1 / 127.5,
        resample=PILImageResampling.BILINEAR,  # NEAREST is too different between PIL and torchvision
    ):
        size = size if size is not None else {"height": 18, "width": 18}
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
        self.resample = resample

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "resample": self.resample,
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
class EfficientNetImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = EfficientNetImageProcessor if is_vision_available() else None
    fast_image_processing_class = EfficientNetImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = EfficientNetImageProcessorTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_rescale(self):
        # EfficientNet optionally rescales between -1 and 1 instead of the usual 0 and 1
        image = np.arange(0, 256, 1, dtype=np.uint8).reshape(1, 8, 32)

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            if image_processing_class == EfficientNetImageProcessorFast:
                image = torch.from_numpy(image)

                # Scale between [-1, 1] with rescale_factor 1/127.5 and rescale_offset=True
                rescaled_image = image_processor.rescale(image, scale=1 / 127.5, offset=True)
                expected_image = (image * (1 / 127.5)) - 1
                self.assertTrue(torch.allclose(rescaled_image, expected_image))

                # Scale between [0, 1] with rescale_factor 1/255 and rescale_offset=True
                rescaled_image = image_processor.rescale(image, scale=1 / 255, offset=False)
                expected_image = image / 255.0
                self.assertTrue(torch.allclose(rescaled_image, expected_image))

            else:
                rescaled_image = image_processor.rescale(image, scale=1 / 127.5, dtype=np.float64)
                expected_image = (image * (1 / 127.5)).astype(np.float64) - 1
                self.assertTrue(np.allclose(rescaled_image, expected_image))

                rescaled_image = image_processor.rescale(image, scale=1 / 255, offset=False, dtype=np.float64)
                expected_image = (image / 255.0).astype(np.float64)
                self.assertTrue(np.allclose(rescaled_image, expected_image))

    @require_vision
    @require_torch
    def test_rescale_normalize(self):
        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image = torch.arange(0, 256, 1, dtype=torch.uint8).reshape(1, 8, 32).repeat(3, 1, 1)
        image_mean_0 = (0.0, 0.0, 0.0)
        image_std_0 = (1.0, 1.0, 1.0)
        image_mean_1 = (0.5, 0.5, 0.5)
        image_std_1 = (0.5, 0.5, 0.5)

        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        # Rescale between [-1, 1] with rescale_factor=1/127.5 and rescale_offset=True. Then normalize
        rescaled_normalized = image_processor_fast.rescale_and_normalize(
            image, True, 1 / 127.5, True, image_mean_0, image_std_0, True
        )
        expected_image = (image * (1 / 127.5)) - 1
        expected_image = (expected_image - torch.tensor(image_mean_0).view(3, 1, 1)) / torch.tensor(image_std_0).view(
            3, 1, 1
        )
        self.assertTrue(torch.allclose(rescaled_normalized, expected_image, rtol=1e-3))

        # Rescale between [0, 1] with rescale_factor=1/255 and rescale_offset=False. Then normalize
        rescaled_normalized = image_processor_fast.rescale_and_normalize(
            image, True, 1 / 255, True, image_mean_1, image_std_1, False
        )
        expected_image = image * (1 / 255.0)
        expected_image = (expected_image - torch.tensor(image_mean_1).view(3, 1, 1)) / torch.tensor(image_std_1).view(
            3, 1, 1
        )
        self.assertTrue(torch.allclose(rescaled_normalized, expected_image, rtol=1e-3))
