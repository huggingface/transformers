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

import numpy as np

from transformers import BatchImages
from transformers.testing_utils import require_tf, require_torch


class ImageProcessorMixin:

    # to overwrite at image processor specific tests
    image_processor_tester = None
    image_processor_class = None

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_common_properties(self):
        image_processor = self.image_processor_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "padding_value"))

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processor_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processor_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processor_class.from_json_file(json_file_path)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processor_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_first.save_pretrained(tmpdirname)
            image_processor_second = self.image_processor_class.from_pretrained(tmpdirname)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_init_without_params(self):
        image_processor = self.image_processor_class()
        self.assertIsNotNone(image_processor)

    def test_batch_images_numpy(self):
        image_inputs = self.image_processor_tester.prepare_inputs_numpy_for_common()
        image_processor = self.image_processor_class(**self.image_processor_dict)
        input_name = image_processor.model_input_names[0]

        processed_images = BatchImages({input_name: image_inputs})

        self.assertTrue(all(len(x) == len(y) for x, y in zip(image_inputs, processed_images[input_name])))

        image_inputs = self.image_processor_tester.prepare_inputs_numpy_for_common(equal_resolution=True)
        processed_images = BatchImages({input_name: image_inputs}, tensor_type="np")

        batch_images_input = processed_images[input_name]

        if len(batch_images_input.shape) < 3:
            batch_images_input = batch_images_input[:, :, None]

        # self.assertTrue(
        #     batch_images_input.shape
        #     == (self.image_processor_tester.batch_size, len(image_inputs[0]), self.image_processor_tester.feature_size)
        # )

    @require_torch
    def test_batch_images_pt(self):
        image_inputs = self.image_processor_tester.prepare_inputs_pytorch_for_common(equal_length=True)
        image_processor = self.image_processor_class(**self.image_processor_dict)
        input_name = image_processor.model_input_names[0]

        processed_images = BatchImages({input_name: image_inputs}, tensor_type="pt")

        batch_images_input = processed_images[input_name]

        if len(batch_images_input.shape) < 3:
            batch_images_input = batch_images_input[:, :, None]

        # self.assertTrue(
        #     batch_images_input.shape
        #     == (self.image_processor_tester.batch_size, len(image_inputs[0]), self.image_processor_tester.feature_size)
        # )

    @require_tf
    def test_batch_images_tf(self):
        pass

    def _check_padding(self, numpify=False):
        pass

    def test_padding_from_list(self):
        self._check_padding(numpify=False)

    def test_padding_from_array(self):
        self._check_padding(numpify=True)

    @require_torch
    def test_padding_accepts_tensors_pt(self):
        pass

    @require_tf
    def test_padding_accepts_tensors_tf(self):
        pass

    def test_attention_mask(self):
        feat_dict = self.image_processor_dict
        feat_dict["return_attention_mask"] = True
        image_processor = self.image_processor_class(**feat_dict)
        image_inputs = self.image_processor_tester.prepare_inputs_pytorch_for_common()
        input_lenghts = [len(x) for x in image_inputs]
        input_name = image_processor.model_input_names[0]

        processed = BatchImages({input_name: image_inputs})

        processed = image_processor.pad(processed, padding="biggest", return_tensors="np")
        self.assertIn("attention_mask", processed)
        self.assertListEqual(list(processed.attention_mask.shape), list(processed[input_name].shape[:2]))
        self.assertListEqual(processed.attention_mask.sum(-1).tolist(), input_lenghts)