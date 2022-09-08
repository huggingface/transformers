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


import json
import pathlib
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DetrFeatureExtractor


class DetrFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=18,
        max_size=1333,  # by setting max_size > max_resolution we're effectively not testing this :p
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "max_size": self.max_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to DetrFeatureExtractor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size * h / w)
                expected_width = self.size
            elif w > h:
                expected_height = self.size
                expected_width = int(self.size * w / h)
            else:
                expected_height = self.size
                expected_width = self.size

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width


@require_torch
@require_vision
class DetrFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = DetrFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = DetrFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "max_size"))

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

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
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

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
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

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.feature_extract_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = feature_extractor(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.feature_extract_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    def test_equivalence_pad_and_create_pixel_mask(self):
        # Initialize feature_extractors
        feature_extractor_1 = self.feature_extraction_class(**self.feat_extract_dict)
        feature_extractor_2 = self.feature_extraction_class(do_resize=False, do_normalize=False)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test whether the method "pad_and_return_pixel_mask" and calling the feature extractor return the same tensors
        encoded_images_with_method = feature_extractor_1.pad_and_create_pixel_mask(image_inputs, return_tensors="pt")
        encoded_images = feature_extractor_2(image_inputs, return_tensors="pt")

        assert torch.allclose(encoded_images_with_method["pixel_values"], encoded_images["pixel_values"], atol=1e-4)
        assert torch.allclose(encoded_images_with_method["pixel_mask"], encoded_images["pixel_mask"], atol=1e-4)

    @slow
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # encode them
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        encoding = feature_extractor(images=image, annotations=target, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        assert torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4)

        # verify area
        expected_area = torch.tensor([5887.9600, 11250.2061, 489353.8438, 837122.7500, 147967.5156, 165732.3438])
        assert torch.allclose(encoding["labels"][0]["area"], expected_area)
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.5503, 0.2765, 0.0604, 0.2215])
        assert torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3)
        # verify image_id
        expected_image_id = torch.tensor([39769])
        assert torch.allclose(encoding["labels"][0]["image_id"], expected_image_id)
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        assert torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd)
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        assert torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        assert torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size)
        # verify size
        expected_size = torch.tensor([800, 1066])
        assert torch.allclose(encoding["labels"][0]["size"], expected_size)

    @slow
    def test_call_pytorch_with_coco_panoptic_annotations(self):
        # prepare image, target and masks_path
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_panoptic_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": target}

        masks_path = pathlib.Path("./tests/fixtures/tests_samples/COCO/coco_panoptic")

        # encode them
        # TODO replace by .from_pretrained facebook/detr-resnet-50-panoptic
        feature_extractor = DetrFeatureExtractor(format="coco_panoptic")
        encoding = feature_extractor(images=image, annotations=target, masks_path=masks_path, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        assert torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4)

        # verify area
        expected_area = torch.tensor([147979.6875, 165527.0469, 484638.5938, 11292.9375, 5879.6562, 7634.1147])
        assert torch.allclose(encoding["labels"][0]["area"], expected_area)
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([0.2625, 0.5437, 0.4688, 0.8625])
        assert torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3)
        # verify image_id
        expected_image_id = torch.tensor([39769])
        assert torch.allclose(encoding["labels"][0]["image_id"], expected_image_id)
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        assert torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd)
        # verify class_labels
        expected_class_labels = torch.tensor([17, 17, 63, 75, 75, 93])
        assert torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels)
        # verify masks
        expected_masks_sum = 822338
        self.assertEqual(encoding["labels"][0]["masks"].sum().item(), expected_masks_sum)
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        assert torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size)
        # verify size
        expected_size = torch.tensor([800, 1066])
        assert torch.allclose(encoding["labels"][0]["size"], expected_size)
