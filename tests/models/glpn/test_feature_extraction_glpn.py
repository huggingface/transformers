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

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import GLPNFeatureExtractor


class GLPNFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size_divisor=32,
        do_rescale=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size_divisor": self.size_divisor,
            "do_rescale": self.do_rescale,
        }


@require_torch
@require_vision
class GLPNFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = GLPNFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = GLPNFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size_divisor"))
        self.assertTrue(hasattr(feature_extractor, "resample"))
        self.assertTrue(hasattr(feature_extractor, "do_rescale"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input (GLPNFeatureExtractor doesn't support batching)
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.feature_extract_tester.size_divisor == 0)
        self.assertTrue(encoded_images.shape[-2] % self.feature_extract_tester.size_divisor == 0)

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input (GLPNFeatureExtractor doesn't support batching)
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.feature_extract_tester.size_divisor == 0)
        self.assertTrue(encoded_images.shape[-2] % self.feature_extract_tester.size_divisor == 0)

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.feature_extract_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input (GLPNFeatureExtractor doesn't support batching)
        encoded_images = feature_extractor(image_inputs[0], return_tensors="pt").pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.feature_extract_tester.size_divisor == 0)
        self.assertTrue(encoded_images.shape[-2] % self.feature_extract_tester.size_divisor == 0)
