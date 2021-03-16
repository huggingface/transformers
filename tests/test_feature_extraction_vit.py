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


import itertools
import random
import unittest

import numpy as np
import torch

from transformers import VIT_PRETRAINED_MODEL_ARCHIVE_LIST, ViTConfig, ViTFeatureExtractor
from transformers.testing_utils import require_torch, require_torchvision, slow

from .test_feature_extraction_common import FeatureExtractionSavingTestMixin


class ViTFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=224,
        min_resolution=30,
        max_resolution=400,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.5, 0.5, 0.5],
        do_normalize=True,
        do_resize=True,
        size=18,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.size = size

    @property
    def prepare_feat_extract_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def prepare_inputs_numpy(self, equal_resolution=False):
        # TO DO 
        input_size = (self.num_channels, self.image_size, self.image_size)
        image_inputs = torch.randn((self.batch_size, *input_size))

        return image_inputs

    def prepare_inputs_pytorch(self, equal_resolution=False):
        # TO DO
        input_size = (self.num_channels, self.image_size, self.image_size)
        image_inputs = torch.randn((self.batch_size, *input_size))

        return image_inputs


class ViTFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = ViTFeatureExtractor
    
    def setUp(self):
        self.feature_extract_tester = ViTFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()
    
    def test_feat_extract_properties(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feat_extract, "image_mean"))
        self.assertTrue(hasattr(feat_extract, "image_std"))
        self.assertTrue(hasattr(feat_extract, "do_normalize"))
        self.assertTrue(hasattr(feat_extract, "do_resize"))
        self.assertTrue(hasattr(feat_extract, "size"))

    def test_batch_feature(self):
        image_inputs = self.feat_extract_tester.prepare_inputs_for_common()
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        input_name = feat_extract.model_input_names[0]

        processed_features = BatchFeature({input_name: image_inputs})

        self.assertTrue(all(len(x) == len(y) for x, y in zip(image_inputs, processed_features[input_name])))

        image_inputs = self.feat_extract_tester.prepare_inputs_for_common(equal_length=True)
        processed_features = BatchFeature({input_name: image_inputs}, tensor_type="np")

        batch_features_input = processed_features[input_name]

        if len(batch_features_input.shape) < 3:
            batch_features_input = batch_features_input[:, :, None]

        self.assertTrue(
            batch_features_input.shape
            == (self.feat_extract_tester.batch_size, len(image_inputs[0]), self.feat_extract_tester.feature_size)
        )
    
    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extract = self.feature_extraction_class(**self.feature_extract_tester.feat_extract_dict())
        # create three inputs of resolution 800, 1000, and 1200
        image_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_image_inputs = [np.asarray(speech_input) for speech_input in image_inputs]

        # Test not batched input
        encoded_images_1 = feature_extractor(image_inputs[0], return_tensors="np").input_values
        encoded_images_2 = feature_extractor(np_image_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_images_1, encoded_images_2, atol=1e-3))

        # Test batched
        encoded_images_1 = feature_extractor(image_inputs, return_tensors="np").input_values
        encoded_images_2 = feature_extractor(np_image_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_images_1, encoded_images_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extract = self.feature_extraction_class(**self.feature_extract_tester.feat_extract_dict())
        # create three inputs of resolution 800, 1000, and 1200
        image_inputs = floats_tensor()

        # Test not batched input
        encoded_images_1 = feature_extractor(image_inputs[0], return_tensors="pt").input_values
        encoded_images_2 = feature_extractor(np_image_inputs[0], return_tensors="pt").input_values
        self.assertTrue(np.allclose(encoded_images_1, encoded_images_2, atol=1e-3))

        # Test batched
        encoded_images_1 = feature_extractor(image_inputs, return_tensors="pt").input_values
        encoded_images_2 = feature_extractor(np_image_inputs, return_tensors="pt").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_images_1, encoded_images_2):
            self.assertTrue(torch.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_normalization(self):
        pass
