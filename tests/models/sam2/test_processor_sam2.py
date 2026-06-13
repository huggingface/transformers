# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import (
    require_torch,
    require_torchvision,
    require_vision,
)
from transformers.utils import is_tf_available, is_torch_available, is_vision_available


if is_vision_available():
    from transformers import AutoProcessor, Sam2ImageProcessorFast, Sam2Processor

if is_torch_available():
    import torch

if is_tf_available():
    pass


@require_vision
@require_torchvision
class Sam2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = Sam2ImageProcessorFast()
        processor = Sam2Processor(image_processor)
        processor.save_pretrained(self.tmpdirname)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        image_inputs = torch.randint(0, 256, size=(1, 3, 30, 400), dtype=torch.uint8)
        # image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
        return image_inputs

    def prepare_mask_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        mask_inputs = torch.randint(0, 256, size=(1, 30, 400), dtype=torch.uint8)
        # mask_inputs = [Image.fromarray(x) for x in mask_inputs]
        return mask_inputs

    def test_save_load_pretrained_additional_features(self):
        image_processor = self.get_image_processor()

        processor = Sam2Processor(image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)

        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = Sam2Processor.from_pretrained(self.tmpdirname, do_normalize=False, padding_value=1.0)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, Sam2ImageProcessorFast)

    def test_image_processor_no_masks(self):
        image_processor = self.get_image_processor()

        processor = Sam2Processor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input)
        input_processor = processor(images=image_input)

        for key in input_feat_extract.keys():
            if key == "pixel_values":
                for input_feat_extract_item, input_processor_item in zip(
                    input_feat_extract[key], input_processor[key]
                ):
                    np.testing.assert_array_equal(input_feat_extract_item, input_processor_item)
            else:
                self.assertEqual(input_feat_extract[key], input_processor[key])

        for image in input_feat_extract.pixel_values:
            self.assertEqual(image.shape, (3, 1024, 1024))

        for original_size in input_feat_extract.original_sizes:
            np.testing.assert_array_equal(original_size, np.array([30, 400]))

    def test_image_processor_with_masks(self):
        image_processor = self.get_image_processor()

        processor = Sam2Processor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()
        mask_input = self.prepare_mask_inputs()

        input_feat_extract = image_processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")
        input_processor = processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

        for label in input_feat_extract.labels:
            self.assertEqual(label.shape, (256, 256))

    @require_torch
    def test_post_process_masks(self):
        image_processor = self.get_image_processor()

        processor = Sam2Processor(image_processor=image_processor)
        dummy_masks = [torch.ones((1, 3, 5, 5))]

        original_sizes = [[1764, 2646]]

        masks = processor.post_process_masks(dummy_masks, original_sizes)
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        masks = processor.post_process_masks(dummy_masks, torch.tensor(original_sizes))
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        # should also work with np
        dummy_masks = [np.ones((1, 3, 5, 5))]
        masks = processor.post_process_masks(dummy_masks, np.array(original_sizes))

        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        dummy_masks = [[1, 0], [0, 1]]
        with self.assertRaises(ValueError):
            masks = processor.post_process_masks(dummy_masks, np.array(original_sizes))
