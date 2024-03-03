# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import requests

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import RTDetrImageProcessor

if is_torch_available():
    import torch


class RTDetrImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        return_tensors="pt",
    ):
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 640, "width": 640}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.return_tensors = return_tensors
        self.num_channels = 3
        self.batch_size = 4

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "return_tensors": self.return_tensors,
            "num_channels": self.num_channels,
        }

    def get_expected_values(self):
        return self.size["height"], self.size["width"]

    def expected_output_image_shape(self, image):
        height, width = self.get_expected_values()
        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=30,
            max_resolution=400,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class RtDetrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = RTDetrImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = RTDetrImageProcessingTester()

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "resample"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "return_tensors"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 640, "width": 640})

    @slow
    def test_image_processor_outputs(self):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")

        image_processing = self.image_processing_class(**self.image_processor_dict)
        encoding = image_processing(images=image, return_tensors="pt")

        # verify pixel values: shape
        expected_shape = torch.Size([1, 3, 640, 640])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        # verify pixel values: output values
        expected_slice = torch.tensor([0.5490196347236633, 0.5647059082984924, 0.572549045085907])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-5))

    def test_multiple_images_processor_outputs(self):
        images_urls = [
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000000776.jpg",
            "http://images.cocodataset.org/val2017/000000000785.jpg",
            "http://images.cocodataset.org/val2017/000000000802.jpg",
            "http://images.cocodataset.org/val2017/000000000872.jpg",
        ]

        images = []
        for url in images_urls:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)

        # apply image processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        encoding = image_processing(images=images, return_tensors="pt")

        # verify if pixel_values is part of the encoding
        self.assertIn("pixel_values", encoding)

        # verify pixel values: shape
        expected_shape = torch.Size([8, 3, 640, 640])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        # verify pixel values: output values
        expected_slices = torch.tensor(
            [
                [0.5333333611488342, 0.5568627715110779, 0.5647059082984924],
                [0.5372549295425415, 0.4705882668495178, 0.4274510145187378],
                [0.3960784673690796, 0.35686275362968445, 0.3686274588108063],
                [0.20784315466880798, 0.1882353127002716, 0.15294118225574493],
                [0.364705890417099, 0.364705890417099, 0.3686274588108063],
                [0.8078432083129883, 0.8078432083129883, 0.8078432083129883],
                [0.4431372880935669, 0.4431372880935669, 0.4431372880935669],
                [0.19607844948768616, 0.21176472306251526, 0.3607843220233917],
            ]
        )
        self.assertTrue(torch.allclose(encoding["pixel_values"][:, 1, 0, :3], expected_slices, atol=1e-5))
