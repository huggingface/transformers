# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import MobileViTImageProcessor


class MobileViTImageProcessingTester:
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
        do_center_crop=True,
        crop_size=None,
        do_flip_channel_order=True,
    ):
        size = size if size is not None else {"shortest_edge": 20}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_flip_channel_order = do_flip_channel_order

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_flip_channel_order": self.do_flip_channel_order,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]

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


def prepare_semantic_single_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image = Image.open(dataset[0]["file"])
    map = Image.open(dataset[1]["file"])

    return image, map


def prepare_semantic_batch_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image1 = Image.open(dataset[0]["file"])
    map1 = Image.open(dataset[1]["file"])
    image2 = Image.open(dataset[2]["file"])
    map2 = Image.open(dataset[3]["file"])

    return [image1, image2], [map1, map2]


@require_torch
@require_vision
class MobileViTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = MobileViTImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = MobileViTImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_flip_channel_order"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 20})
        self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, {"shortest_edge": 42})
        self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_segmentation_maps(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        maps = []
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
            maps.append(torch.zeros(image.shape[-2:]).long())

        # Test not batched input
        encoding = image_processing(image_inputs[0], maps[0], return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched
        encoding = image_processing(image_inputs, maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test not batched input (PIL images)
        image, segmentation_map = prepare_semantic_single_inputs()

        encoding = image_processing(image, segmentation_map, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched input (PIL images)
        images, segmentation_maps = prepare_semantic_batch_inputs()

        encoding = image_processing(images, segmentation_maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                2,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                2,
                self.image_processor_tester.crop_size["height"],
                self.image_processor_tester.crop_size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)
