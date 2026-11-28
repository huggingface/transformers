# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class LlavaOnevision2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        super().__init__(
            batch_size=3,
            min_resolution=12,
            max_resolution=32,
            size={"shortest_edge": 16, "longest_edge": 1024},
            patch_size=2,
            **kwargs,
        )


@require_torch
@require_torchvision
@require_vision
class LlavaOnevision2ImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = LlavaOnevision2ImageProcessingTester

    def _check_packed_patch_grid(self, **input_kwargs):
        num_channels = self.image_processor_tester.num_channels
        images = self.image_processor_tester.prepare_image_inputs(**input_kwargs)
        kwargs = {}
        if num_channels == 4:
            kwargs = {"do_convert_rgb": False, "input_data_format": "channels_last", "do_normalize": False}
        for image_processing_class in self.image_processing_classes.values():
            processor = image_processing_class(**self.image_processor_dict)
            for inputs in (images[:1], images):
                output = processor(inputs[0] if len(inputs) == 1 else inputs, return_tensors="pt", **kwargs)
                self.assertEqual(output.image_grid_thw.shape, (len(inputs), 3))
                self.assertEqual(
                    tuple(output.pixel_values.shape),
                    (int(output.image_grid_thw.prod(-1).sum()), num_channels * processor.patch_size**2),
                )
                self.assertTrue((output.image_grid_thw[:, 0] == 1).all())

    def test_call_pil(self):
        self._check_packed_patch_grid()

    def test_call_numpy(self):
        self._check_packed_patch_grid(numpify=True)

    def test_call_pytorch(self):
        self._check_packed_patch_grid(torchify=True)

    def test_call_numpy_4_channels(self):
        self.image_processor_tester.num_channels = 4
        self._check_packed_patch_grid(numpify=True)

    def test_patch_order_and_backends(self):
        import torch

        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        pil = self.image_processing_classes["pil"](**self.image_processor_dict)
        torchvision = self.image_processing_classes["torchvision"](**self.image_processor_dict)
        outputs = pil(image, return_tensors="pt")
        self.assertEqual(outputs.image_grid_thw.tolist(), [[1, 4, 4]])
        torch.testing.assert_close(outputs.pixel_values, torchvision(image, return_tensors="pt").pixel_values)
