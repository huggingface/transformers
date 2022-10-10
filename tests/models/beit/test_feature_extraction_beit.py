# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import BeitFeatureExtractor


class BeitFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=20,
        do_center_crop=True,
        crop_size=18,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        reduce_labels=False,
    ):
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
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.reduce_labels = reduce_labels

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "reduce_labels": self.reduce_labels,
        }


def prepare_semantic_single_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

    image = Image.open(dataset[0]["file"])
    map = Image.open(dataset[1]["file"])

    return image, map


def prepare_semantic_batch_inputs():
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

    image1 = Image.open(ds[0]["file"])
    map1 = Image.open(ds[1]["file"])
    image2 = Image.open(ds[2]["file"])
    map2 = Image.open(ds[3]["file"])

    return [image1, image2], [map1, map2]


@require_torch
@require_vision
class BeitFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = BeitFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = BeitFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "do_center_crop"))
        self.assertTrue(hasattr(feature_extractor, "center_crop"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )

    def test_call_segmentation_maps(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        maps = []
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
            maps.append(torch.zeros(image.shape[-2:]).long())

        # Test not batched input
        encoding = feature_extractor(image_inputs[0], maps[0], return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched
        encoding = feature_extractor(image_inputs, maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test not batched input (PIL images)
        image, segmentation_map = prepare_semantic_single_inputs()

        encoding = feature_extractor(image, segmentation_map, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched input (PIL images)
        images, segmentation_maps = prepare_semantic_batch_inputs()

        encoding = feature_extractor(images, segmentation_maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                2,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                2,
                self.feature_extract_tester.crop_size,
                self.feature_extract_tester.crop_size,
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_reduce_labels(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)

        # ADE20k has 150 classes, and the background is included, so labels should be between 0 and 150
        image, map = prepare_semantic_single_inputs()
        encoding = feature_extractor(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 150)

        feature_extractor.reduce_labels = True
        encoding = feature_extractor(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)
