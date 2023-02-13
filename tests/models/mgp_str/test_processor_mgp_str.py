# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the MGPSTRProcessor. """

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from transformers import MGPSTRTokenizer
from transformers.models.mgp_str.tokenization_mgp_str import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available, IMAGE_PROCESSOR_NAME

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

    from transformers import MGPSTRProcessor


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor, MGPSTRProcessor


@require_torch
@require_vision
class ViTImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):

    def setUp(self):
        self.image_size = (3, 32, 128)
        self.tmpdirname = tempfile.mkdtemp()

        # fmt: off
        vocab = ['[GO]', '[s]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        # fmt: on
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")


        image_processor_map = {
            "do_normalize": False,
            "do_resize": True,
            "feature_extractor_type": "ViTFeatureExtractor",
            "resample": 3,
            "size": {"height": 32, "width": 128}
        }
        self.image_processor_file = os.path.join(self.tmpdirname, IMAGE_PROCESSOR_NAME)
        with open(self.image_processor_file, "w", encoding="utf-8") as fp:
            json.dump(image_processor_map, fp)

    def get_tokenizer(self, **kwargs):
        return MGPSTRTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return MGPSTRProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images.
        """

        image_input = np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)

        image_input = Image.fromarray(np.moveaxis(image_input, 0, -1))

        return image_input

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = MGPSTRProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)
        processor = MGPSTRProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, MGPSTRTokenizer)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MGPSTRProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_processor = processor(images=image_input)

        self.assertEqual(input_processor[0].shape, self.image_size)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MGPSTRProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "test"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)
        import pdb;pdb.set_trace()
        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor_batch_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = MGPSTRProcessor(tokenizer=tokenizer, image_processor=image_processor)

        char_input = torch.randn(1, 27, 38)
        bpe_input = torch.randn(1, 27, 50257)
        wp_input = torch.randn(1,27, 30522)

        results = processor.batch_decode([char_input, bpe_input, wp_input]])

        self.assertListEqual(list(results.keys()), ['generated_text', 'scores', 'char_preds', 'bpe_preds', 'wp_preds'])





    # @property
    # def image_processor_dict(self):
    #     return self.image_processor_tester.prepare_image_processor_dict()

    # def test_image_processor_properties(self):
    #     image_processing = self.image_processing_class(**self.image_processor_dict)
    #     self.assertTrue(hasattr(image_processing, "image_mean"))
    #     self.assertTrue(hasattr(image_processing, "image_std"))
    #     self.assertTrue(hasattr(image_processing, "do_normalize"))
    #     self.assertTrue(hasattr(image_processing, "do_resize"))
    #     self.assertTrue(hasattr(image_processing, "size"))

    # def test_image_processor_from_dict_with_kwargs(self):
    #     image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
    #     self.assertEqual(image_processor.size, {"height": 18, "width": 18})

    #     image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
    #     self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    # def test_batch_feature(self):
    #     pass

    # def test_call_pil(self):
    #     # Initialize image_processing
    #     image_processing = self.image_processing_class(**self.image_processor_dict)
    #     # create random PIL images
    #     image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
    #     for image in image_inputs:
    #         self.assertIsInstance(image, Image.Image)

    #     # Test not batched input
    #     encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             1,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )

    #     # Test batched
    #     encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             self.image_processor_tester.batch_size,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )

    # def test_call_numpy(self):
    #     # Initialize image_processing
    #     image_processing = self.image_processing_class(**self.image_processor_dict)
    #     # create random numpy tensors
    #     image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
    #     for image in image_inputs:
    #         self.assertIsInstance(image, np.ndarray)

    #     # Test not batched input
    #     encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             1,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )

    #     # Test batched
    #     encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             self.image_processor_tester.batch_size,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )

    # def test_call_pytorch(self):
    #     # Initialize image_processing
    #     image_processing = self.image_processing_class(**self.image_processor_dict)
    #     # create random PyTorch tensors
    #     image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
    #     for image in image_inputs:
    #         self.assertIsInstance(image, torch.Tensor)

    #     # Test not batched input
    #     encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             1,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )

    #     # Test batched
    #     encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
    #     self.assertEqual(
    #         encoded_images.shape,
    #         (
    #             self.image_processor_tester.batch_size,
    #             self.image_processor_tester.num_channels,
    #             self.image_processor_tester.size["height"],
    #             self.image_processor_tester.size["width"],
    #         ),
    #     )


