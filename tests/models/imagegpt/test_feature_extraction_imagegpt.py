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
import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ImageGPTFeatureExtractor


class ImageGPTFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=18,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            # here we create 2 clusters for the sake of simplicity
            "clusters": np.asarray(
                [
                    [0.8866443634033203, 0.6618829369544983, 0.3891746401786804],
                    [-0.6042559146881104, -0.02295008860528469, 0.5423797369003296],
                ]
            ),
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
        }


@require_torch
@require_vision
class ImageGPTFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = ImageGPTFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = ImageGPTFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "clusters"))
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))

    def test_feat_extract_to_json_string(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        obj = json.loads(feat_extract.to_json_string())
        for key, value in self.feat_extract_dict.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, obj[key]))
            else:
                self.assertEqual(obj[key], value)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path).to_dict()

        feat_extract_first = feat_extract_first.to_dict()
        for key, value in feat_extract_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, feat_extract_second[key]))
            else:
                self.assertEqual(feat_extract_first[key], value)

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            feat_extract_first.save_pretrained(tmpdirname)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname).to_dict()

        feat_extract_first = feat_extract_first.to_dict()
        for key, value in feat_extract_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, feat_extract_second[key]))
            else:
                self.assertEqual(feat_extract_first[key], value)

    @unittest.skip("ImageGPT requires clusters at initialization")
    def test_init_without_params(self):
        pass


def prepare_images():
    dataset = load_dataset("hf-internal-testing/fixtures_image_utils", split="test")

    image1 = Image.open(dataset[4]["file"])
    image2 = Image.open(dataset[5]["file"])

    images = [image1, image2]

    return images


@require_vision
@require_torch
class ImageGPTFeatureExtractorIntegrationTest(unittest.TestCase):
    @slow
    def test_image(self):
        feature_extractor = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")

        images = prepare_images()

        # test non-batched
        encoding = feature_extractor(images[0], return_tensors="pt")

        self.assertIsInstance(encoding.input_ids, torch.LongTensor)
        self.assertEqual(encoding.input_ids.shape, (1, 1024))

        expected_slice = [306, 191, 191]
        self.assertEqual(encoding.input_ids[0, :3].tolist(), expected_slice)

        # test batched
        encoding = feature_extractor(images, return_tensors="pt")

        self.assertIsInstance(encoding.input_ids, torch.LongTensor)
        self.assertEqual(encoding.input_ids.shape, (2, 1024))

        expected_slice = [303, 13, 13]
        self.assertEqual(encoding.input_ids[1, -3:].tolist(), expected_slice)
