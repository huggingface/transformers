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

from transformers.testing_utils import require_torch, require_torchvision, require_vision, torch_device
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    if is_torchvision_available():
        from transformers import HYV4VLImageProcessor


class HYV4VLImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("min_resolution", 32)
        kwargs.setdefault("max_resolution", 64)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("merge_size", 2)
        kwargs.setdefault("size", {"shortest_edge": 1024, "longest_edge": 1024})
        super().__init__(parent=parent, **kwargs)

    def prepare_image_processor_dict(self):
        return {
            "device": torch_device,
            "size": self.size.copy(),
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_convert_rgb": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }


@require_torch
@require_vision
@require_torchvision
class HYV4VLImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    test_cast_dtype = True
    image_processor_tester_class = HYV4VLImageProcessingTester

    @property
    def image_processor_dict(self):
        processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        if getattr(self, "_compare_backends", False):
            # The PIL backend always returns CPU tensors, so compare both backends on CPU.
            processor_dict.pop("device")
        return processor_dict

    def test_backends_equivalence(self):
        self._compare_backends = True
        super().test_backends_equivalence()

    def test_backends_equivalence_batched(self):
        self._compare_backends = True
        super().test_backends_equivalence_batched()

    def _assert_output(self, output, batch_size):
        self.assertEqual(output.pixel_values.device.type, torch.device(torch_device).type)
        self.assertEqual(output.pixel_values.shape, (batch_size * 4, 3 * 16 * 16))
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2]] * batch_size)
        self.assertEqual(output.image_grid_thw.dtype, torch.long)

    def test_call_pil(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
        self._assert_output(processor(images[0], return_tensors="pt"), 1)
        self._assert_output(processor(images, return_tensors="pt"), len(images))

    def test_call_numpy(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
        self._assert_output(processor(images[0], return_tensors="pt"), 1)
        self._assert_output(processor(images, return_tensors="pt"), len(images))

    def test_call_pytorch(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
        self._assert_output(processor(images[0], return_tensors="pt"), 1)
        self._assert_output(processor(images, return_tensors="pt"), len(images))

    def test_call_numpy_4_channels(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        image = np.full((32, 32, 4), 255, dtype=np.uint8)
        # PIL RGBA conversion is the supported route to RGB, not arbitrary four-channel model input.
        self._assert_output(processor(Image.fromarray(image), return_tensors="pt"), 1)

    def test_non_square_grid_patch_counts_and_batch_order(self):
        processor = HYV4VLImageProcessor(size={"shortest_edge": 1024, "longest_edge": 8192}, device=torch_device)
        sizes = [(32, 64), (64, 32), (32, 64)]
        images = [np.full((h, w, 3), i * 100, dtype=np.uint8) for i, (h, w) in enumerate(sizes)]
        batched = processor(images, return_tensors="pt")
        self.assertEqual(batched.image_grid_thw.tolist(), [[1, 2, 4], [1, 4, 2], [1, 2, 4]])
        singles = [processor(image, return_tensors="pt") for image in images]
        torch.testing.assert_close(batched.pixel_values, torch.cat([output.pixel_values for output in singles]))
        for (h, w), output in zip(sizes, singles):
            patches_h, patches_w = processor.get_number_of_image_patches(h, w)
            self.assertEqual(patches_h * patches_w, output.pixel_values.shape[0])

    def test_temporal_patch_repeats_image(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        image = np.arange(32 * 32 * 3, dtype=np.uint16).astype(np.uint8).reshape(32, 32, 3)
        output = processor(image, temporal_patch_size=2, return_tensors="pt")
        self.assertEqual(output.pixel_values.shape, (4, 3 * 2 * 16 * 16))
        patches = output.pixel_values.reshape(4, 3, 2, 16, 16)
        torch.testing.assert_close(patches[:, :, 0], patches[:, :, 1])
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2]])

    def test_normalization(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        images = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (0, 255)]
        output = processor(images, return_tensors="pt")
        torch.testing.assert_close(output.pixel_values[:4], -torch.ones(4, 768, device=torch_device))
        torch.testing.assert_close(output.pixel_values[4:], torch.ones(4, 768, device=torch_device))

    def test_cast_dtype_device(self):
        processor = HYV4VLImageProcessor(**self.image_processor_dict)
        images = self.image_processor_tester.prepare_image_inputs()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = processor(images, return_tensors="pt").to(dtype=dtype, device=torch_device)
                self.assertEqual(output.pixel_values.dtype, dtype)
                self.assertEqual(output.pixel_values.device.type, torch.device(torch_device).type)
                self.assertEqual(output.image_grid_thw.dtype, torch.long)
