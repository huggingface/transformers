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
        min_resolution=400,
        max_resolution=2000,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.5, 0.5, 0.5],
        padding_value=0.0,
        return_pixel_mask=True,
        do_normalize=True,
        do_resize=True,
        size=18,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.image_mean = image_mean
        self.image_std = image_std
        self.padding_value = padding_value
        self.return_pixel_mask = return_pixel_mask
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.size = size

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "padding_value": self.padding_value,
            "return_pixel_mask": self.return_pixel_mask,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }

    def prepare_inputs_for_common(self, equal_resolution=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_resolution:
            image_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            image_inputs = [
                _flatten(floats_list((x, self.feature_size)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            image_inputs = [np.asarray(x) for x in image_inputs]

        return image_inputs


class ViTImageProcessorTest(ImageProcessorMixin, unittest.TestCase):

    image_processor_class = ViTImageProcessor

    def setUp(self):
        self.image_processor_tester = VitImageProcessorTester(self)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        image_processor = self.image_processor_class(**self.image_processor_tester.prepare_image_processor_dict())
        # create three inputs of resolution 800, 1000, and 1200
        image_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_image_inputs = [np.asarray(speech_input) for speech_input in image_inputs]

        # Test not batched input
        encoded_sequences_1 = image_processor(image_inputs[0], return_tensors="np").input_values
        encoded_sequences_2 = image_processor(np_image_inputs[0], return_tensors="np").input_values
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = image_processor(image_inputs, return_tensors="np").input_values
        encoded_sequences_2 = image_processor(np_image_inputs, return_tensors="np").input_values
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_normalization(self):
        pass

    @slow
    def test_pretrained_checkpoints_are_set_correctly(self):
        pass