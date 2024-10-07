# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from transformers import is_torch_available, is_vision_available
from transformers.processing_utils import _validate_images_text_input_order
from transformers.testing_utils import require_torch, require_vision


if is_vision_available():
    import PIL

if is_torch_available():
    import torch


@require_vision
class ProcessingUtilTester(unittest.TestCase):
    def test_validate_images_text_input_order(self):
        # text string and PIL images inputs
        images = PIL.Image.new("RGB", (224, 224))
        text = "text"
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)

        # text list of string and numpy images inputs
        images = np.random.rand(224, 224, 3)
        text = ["text1", "text2"]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertTrue(np.array_equal(valid_images, images))
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertTrue(np.array_equal(valid_images, images))
        self.assertEqual(valid_text, text)

        # text nested list of string and list of pil images inputs
        images = [PIL.Image.new("RGB", (224, 224)), PIL.Image.new("RGB", (224, 224))]
        text = [["text1", "text2, text3"], ["text3", "text4"]]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)

        # list of strings and list of numpy images inputs
        images = [np.random.rand(224, 224, 3), np.random.rand(224, 224, 3)]
        text = ["text1", "text2"]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertTrue(np.array_equal(valid_images[0], images[0]))
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertTrue(np.array_equal(valid_images[0], images[0]))
        self.assertEqual(valid_text, text)

        # list of strings and list of url images inputs
        images = ["https://url1", "https://url2"]
        text = ["text1", "text2"]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)

        # list of strings and nested list of numpy images inputs
        images = [[np.random.rand(224, 224, 3), np.random.rand(224, 224, 3)], [np.random.rand(224, 224, 3)]]
        text = ["text1", "text2"]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertTrue(np.array_equal(valid_images[0][0], images[0][0]))
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertTrue(np.array_equal(valid_images[0][0], images[0][0]))
        self.assertEqual(valid_text, text)

        # nested list of strings and nested list of PIL images inputs
        images = [
            [PIL.Image.new("RGB", (224, 224)), PIL.Image.new("RGB", (224, 224))],
            [PIL.Image.new("RGB", (224, 224))],
        ]
        text = [["text1", "text2, text3"], ["text3", "text4"]]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(valid_images, images)
        self.assertEqual(valid_text, text)

        # None images
        images = None
        text = "text"
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(images, None)
        self.assertEqual(text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(images, None)
        self.assertEqual(text, text)

        # None text
        images = PIL.Image.new("RGB", (224, 224))
        text = None
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertEqual(images, images)
        self.assertEqual(text, None)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertEqual(images, images)
        self.assertEqual(text, None)

        # incorrect inputs
        images = "text"
        text = "text"
        with self.assertRaises(ValueError):
            _validate_images_text_input_order(images=images, text=text)

    @require_torch
    def test_validate_images_text_input_order_torch(self):
        # text string and torch images inputs
        images = torch.rand(224, 224, 3)
        text = "text"
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertTrue(torch.equal(valid_images, images))
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertTrue(torch.equal(valid_images, images))
        self.assertEqual(valid_text, text)

        # text list of string and list of torch images inputs
        images = [torch.rand(224, 224, 3), torch.rand(224, 224, 3)]
        text = ["text1", "text2"]
        # test correct text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=images, text=text)
        self.assertTrue(torch.equal(valid_images[0], images[0]))
        self.assertEqual(valid_text, text)
        # test incorrect text and images order
        valid_images, valid_text = _validate_images_text_input_order(images=text, text=images)
        self.assertTrue(torch.equal(valid_images[0], images[0]))
        self.assertEqual(valid_text, text)
