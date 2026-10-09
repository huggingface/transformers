# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from unittest.mock import patch

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import SamHQProcessor

if is_torch_available():
    import torch


@require_vision
@require_torchvision
class SamHQProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = SamHQProcessor

    def prepare_mask_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        mask_inputs = [np.random.randint(255, size=(30, 400), dtype=np.uint8)]
        mask_inputs = [Image.fromarray(x) for x in mask_inputs]
        return mask_inputs

    def test_image_processor_no_masks(self):
        image_processor = self.get_component("image_processor")

        processor = SamHQProcessor(image_processor=image_processor)

        image_input = self.prepare_images_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="pt")
        input_processor = processor(images=image_input, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum().item(), input_processor[key].sum().item(), delta=1e-2)

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

        processor = SamHQProcessor(image_processor=image_processor)

        image_input = self.prepare_images_inputs()
        mask_input = self.prepare_mask_inputs()

        input_feat_extract = image_processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")
        input_processor = processor(images=image_input, segmentation_maps=mask_input, return_tensors="pt")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum().item(), input_processor[key].sum().item(), delta=1e-2)

        for label in input_feat_extract.labels:
            self.assertEqual(label.shape, (256, 256))

    @require_torch
    def test_prompt_inputs_legacy_kwargs(self):
        image_processor = self.get_component(
            "image_processor", size={"longest_edge": 64}, pad_size={"height": 64, "width": 64}
        )
        processor = SamHQProcessor(image_processor=image_processor)
        images = [Image.new("RGB", (32, 16)) for _ in range(2)]
        prompt_inputs = {
            "input_points": [[[4, 6]], [[8, 10], [12, 14]]],
            "input_labels": [[1], [1, 0]],
            "input_boxes": [[[2, 3, 10, 12]], [[3, 4, 11, 13]]],
            "point_pad_value": -7,
        }

        expected = processor(images=images, **prompt_inputs, return_tensors="pt")
        self.assertEqual(expected.input_points.tolist(), [[[[8, 12], [-7, -7]]], [[[16, 20], [24, 28]]]])
        self.assertEqual(expected.input_labels.tolist(), [[[1, -7]], [[1, 0]]])
        self.assertEqual(expected.input_boxes.tolist(), [[[4, 6, 20, 24]], [[6, 8, 22, 26]]])

        for direct_kwargs in ({}, dict.fromkeys(prompt_inputs)):
            with self.subTest(direct_kwargs=direct_kwargs):
                with patch("transformers.models.sam_hq.processing_sam_hq.logger.warning_once") as warning:
                    actual = processor(
                        images=images, images_kwargs=prompt_inputs, **direct_kwargs, return_tensors="pt"
                    )
                for key in prompt_inputs:
                    warning.assert_any_call(
                        f"Passing `{key}` in `images_kwargs` is deprecated "
                        "and will be removed in v5.29.0. "
                        "Pass it directly to the processor instead."
                    )
                for key in ("input_points", "input_labels", "input_boxes"):
                    torch.testing.assert_close(actual[key], expected[key])

    @require_torch
    def test_post_process_masks(self):
        image_processor = self.get_component("image_processor")

        processor = SamHQProcessor(image_processor=image_processor)
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
