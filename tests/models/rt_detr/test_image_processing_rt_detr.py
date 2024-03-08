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
import json
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

    def test_valid_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        params = {"image_id": 39769, "annotations": target}

        # encode them
        image_processing = RTDetrImageProcessor.from_pretrained("sbchoi/rtdetr_r50vd")

        # legal encodings (single image)
        _ = image_processing(images=image, annotations=params, return_tensors="pt")
        _ = image_processing(images=image, annotations=[params], return_tensors="pt")

        # legal encodings (batch of one image)
        _ = image_processing(images=[image], annotations=params, return_tensors="pt")
        _ = image_processing(images=[image], annotations=[params], return_tensors="pt")

        # legal encoding (batch of more than one image)
        n = 5
        _ = image_processing(images=[image] * n, annotations=[params] * n, return_tensors="pt")

        # example of an illegal encoding (missing the 'image_id' key)
        with self.assertRaises(ValueError) as e:
            image_processing(images=image, annotations={"annotations": target}, return_tensors="pt")

        self.assertTrue(str(e.exception).startswith("Invalid COCO detection annotations"))

        # example of an illegal encoding (unequal lengths of images and annotations)
        with self.assertRaises(ValueError) as e:
            image_processing(images=[image] * n, annotations=[params] * (n - 1), return_tensors="pt")

        self.assertTrue(str(e.exception) == "The number of images (5) and annotations (4) do not match.")

    @slow
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # encode them
        image_processing = RTDetrImageProcessor.from_pretrained("sbchoi/rtdetr_r50vd")
        encoding = image_processing(images=image, annotations=target, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([5887.9600, 11250.2061, 489353.8438, 837122.7500, 147967.5156, 165732.3438])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.5503, 0.2765, 0.0604, 0.2215])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))

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
