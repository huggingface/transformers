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


import shutil

import numpy as np
import pytest
from PIL import Image

from transformers import AutoProcessor
from transformers.testing_utils import require_torch


@require_torch
class ProcessorTesterMixin:
    processor_class = None

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_fast_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_default(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_fast_tokenizer()
        image_processor = self.get_image_processor()

        processor_slow = self.processor_class(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor_slow.save_pretrained(self.tmpdirname)
        processor_slow = self.processor_class.from_pretrained(self.tmpdirname, use_fast=False)

        processor_fast = self.processor_class(tokenizer=tokenizer_fast, image_processor=image_processor)
        processor_fast.save_pretrained(self.tmpdirname)
        processor_fast = self.processor_class.from_pretrained(self.tmpdirname)

        self.assertEqual(processor_slow.tokenizer.get_vocab(), tokenizer_slow.get_vocab())
        self.assertEqual(processor_fast.tokenizer.get_vocab(), tokenizer_fast.get_vocab())
        self.assertEqual(tokenizer_slow.get_vocab(), tokenizer_fast.get_vocab())
        self.assertIsInstance(processor_slow.tokenizer, tokenizer_slow)
        self.assertIsInstance(processor_fast.tokenizer, tokenizer_fast)

        self.assertEqual(processor_slow.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor_fast.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor_slow.image_processor, tokenizer_slow)
        self.assertIsInstance(processor_fast.image_processor, tokenizer_fast)

    def test_save_load_pretrained_additional_features(self):
        tokenizer_slow = self.get_tokenizer()
        tokenizer_fast = self.get_fast_tokenizer()
        image_processor = self.get_image_processor()

        processor = self.processor_class(tokenizer=tokenizer_slow, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = self.processor_class.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, tokenizer_fast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, image_processor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_processor = image_processor(image_input, return_tensors="np")
        inputs = processor(images=image_input, return_tensors="np")

        for key in input_image_processor.keys():
            self.assertAlmostEqual(input_image_processor[key].sum(), inputs[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"

        inputs_tokenizer = tokenizer(input_str, return_token_type_ids=False)
        inputs = processor(text=input_str)

        for key in inputs_tokenizer.keys():
            self.assertListEqual(inputs[key], inputs[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)
