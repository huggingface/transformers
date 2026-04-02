# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import EfficientvitsamImageProcessor, EfficientvitsamProcessor
from transformers.testing_utils import require_torch, require_vision


@require_torch
@require_vision
class EfficientvitsamProcessorTest(unittest.TestCase):
    def test_processor_uses_prompt_resolution_for_target_size(self):
        processor = EfficientvitsamProcessor(EfficientvitsamImageProcessor())
        self.assertEqual(processor.target_size, 1024)

    def test_processor_scales_points_with_prompt_resolution(self):
        from PIL import Image

        image = Image.fromarray(np.zeros((300, 500, 3), dtype=np.uint8))
        processor = EfficientvitsamProcessor(EfficientvitsamImageProcessor())
        inputs = processor(images=image, input_points=[[[250, 150]]], input_labels=[[1]], return_tensors="pt")

        expected_x = 250 * (1024 / 500)
        expected_y = 150 * (614 / 300)

        self.assertAlmostEqual(inputs["input_points"][0, 0, 0, 0].item(), expected_x, places=4)
        self.assertAlmostEqual(inputs["input_points"][0, 0, 0, 1].item(), expected_y, places=4)


'''
# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import pytest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import EfficientvitsamProcessor

if is_torch_available():
    import torch

    from transformers.models.sam.image_processing_sam import _mask_to_rle


pytest.skip("Legacy SAM-wrapper processor tests are obsolete after the MIT EfficientViT-SAM port.", allow_module_level=True)


@require_vision
@require_torchvision
class EfficientvitsamProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EfficientvitsamProcessor

    def prepare_mask_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        mask_inputs = [np.random.randint(255, size=(30, 400), dtype=np.uint8)]
        mask_inputs = [Image.fromarray(x) for x in mask_inputs]
        return mask_inputs

    def test_chat_template_save_loading(self):
        self.skipTest("EfficientvitsamProcessor does not have a tokenizer")

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        self.skipTest("EfficientvitsamProcessor does not have a tokenizer")

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        self.skipTest("EfficientvitsamProcessor does not have a tokenizer")

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        self.skipTest("EfficientvitsamProcessor does not have a tokenizer")

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        self.skipTest("EfficientvitsamProcessor does not have a tokenizer")

    def test_image_processor_no_masks(self):
        image_processor = self.get_component("image_processor")

        processor = EfficientvitsamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

        for image in input_feat_extract.pixel_values:
            self.assertEqual(image.shape, (3, 1024, 1024))

        for original_size in input_feat_extract.original_sizes:
            np.testing.assert_array_equal(original_size, np.array([30, 400]))

        for reshaped_input_size in input_feat_extract.reshaped_input_sizes:
            np.testing.assert_array_equal(
                reshaped_input_size, np.array([77, 1024])
            )  # reshaped_input_size value is before padding

    def test_image_processor_with_masks(self):
        image_processor = self.get_component("image_processor")

        processor = EfficientvitsamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()
        mask_input = self.prepare_mask_inputs()

        input_feat_extract = image_processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")
        input_processor = processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

        for label in input_feat_extract.labels:
            self.assertEqual(label.shape, (256, 256))

    @require_torch
    def test_post_process_masks(self):
        image_processor = self.get_component("image_processor")

        processor = EfficientvitsamProcessor(image_processor=image_processor)
        dummy_masks = [torch.ones((1, 3, 5, 5))]

        original_sizes = [[1764, 2646]]

        reshaped_input_size = [[683, 1024]]
        masks = processor.post_process_masks(dummy_masks, original_sizes, reshaped_input_size)
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        masks = processor.post_process_masks(
            dummy_masks, torch.tensor(original_sizes), torch.tensor(reshaped_input_size)
        )
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        # should also work with np
        dummy_masks = [np.ones((1, 3, 5, 5))]
        masks = processor.post_process_masks(dummy_masks, np.array(original_sizes), np.array(reshaped_input_size))

        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        dummy_masks = [[1, 0], [0, 1]]
        with self.assertRaises(TypeError):
            masks = processor.post_process_masks(dummy_masks, np.array(original_sizes), np.array(reshaped_input_size))

    def test_rle_encoding(self):
        """
        Test the run-length encoding function.
        """
        # Test that a mask of all zeros returns a single run [height * width].
        input_mask = torch.zeros((1, 2, 2), dtype=torch.long)  # shape: 1 x 2 x 2
        rle = _mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        # For a 2x2 all-zero mask, we expect a single run of length 4:
        self.assertEqual(rle[0]["counts"], [4])

        # Test that a mask of all ones returns [0, height * width].
        input_mask = torch.ones((1, 2, 2), dtype=torch.long)  # shape: 1 x 2 x 2
        rle = _mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        # For a 2x2 all-one mask, we expect two runs: [0, 4].
        self.assertEqual(rle[0]["counts"], [0, 4])

        # Test a mask with mixed 0s and 1s to ensure the run-length encoding is correct.
        # Example mask:
        # Row 0: [0, 1]
        # Row 1: [1, 1]
        # This is shape (1, 2, 2).
        # Flattened in Fortran order -> [0, 1, 1, 1].
        # The RLE for [0,1,1,1] is [1, 3].
        input_mask = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.long)
        rle = _mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        self.assertEqual(rle[0]["counts"], [1, 3])  # 1 zero, followed by 3 ones
'''
