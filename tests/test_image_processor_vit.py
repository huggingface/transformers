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

from transformers import VIT_PRETRAINED_MODEL_ARCHIVE_LIST, ViTConfig, ViTImageProcessor
from transformers.testing_utils import slow

from .test_image_processor_common import ImageProcessorMixin


global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class ViTImageProcessorTester(unittest.TestCase):
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
        padding_value=0.0,
        return_attention_mask=True,
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
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.size = size

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "padding_value": self.padding_value,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def prepare_inputs_for_common(self):
        input_size = (self.num_channels, self.image_size, self.image_size)
        image_inputs = torch.randn((self.batch_size, *input_size))

        return image_inputs


class ViTImageProcessorTest(ImageProcessorMixin, unittest.TestCase):

    image_processor_class = ViTImageProcessor

    def setUp(self):
        self.image_processor_tester = VitImageProcessorTester(self)

    def test_call_numpy(self):
        # Initialize image_processor
        image_processor = self.image_processor_class(**self.image_processor_tester.prepare_image_processor_dict())
        # create three inputs of resolution 800, 1000, and 1200
        image_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_image_inputs = [np.asarray(speech_input) for speech_input in image_inputs]

        # Test not batched input
        encoded_images_1 = image_processor(image_inputs[0], return_tensors="np").input_values
        encoded_images_2 = image_processor(np_image_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_images_1, encoded_images_2, atol=1e-3))

        # Test batched
        encoded_images_1 = image_processor(image_inputs, return_tensors="np").input_values
        encoded_images_2 = image_processor(np_image_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_images_1, encoded_images_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_call_pytorch(self):
        # Initialize image_processor
        image_processor = self.image_processor_class(**self.image_processor_tester.prepare_image_processor_dict())
        # create three inputs of resolution 800, 1000, and 1200
        image_inputs = None

        # Test not batched input
        encoded_images_1 = image_processor(image_inputs[0], return_tensors="pt").input_values
        encoded_images_2 = image_processor(np_image_inputs[0], return_tensors="pt").input_values
        self.assertTrue(np.allclose(encoded_images_1, encoded_images_2, atol=1e-3))

        # Test batched
        encoded_images_1 = image_processor(image_inputs, return_tensors="pt").input_values
        encoded_images_2 = image_processor(np_image_inputs, return_tensors="pt").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_images_1, encoded_images_2):
            self.assertTrue(torch.allclose(enc_seq_1, enc_seq_2, atol=1e-3))
    
    def test_normalization(self):
        pass

    @slow
    def test_pretrained_checkpoints_are_set_correctly(self):
        pass