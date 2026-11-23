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
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Gr00tImageProcessor


@require_torch
@require_torchvision
@require_vision
class Gr00tImageProcessingTest(unittest.TestCase):
    def get_image_processor(self, **kwargs):
        # 32x32 images with 16x16 patches merged 2x2: a 2x2 grid of patches, i.e. a single image token
        kwargs.setdefault("image_resolution", 32)
        kwargs.setdefault("size", {"shortest_edge": 32 * 32, "longest_edge": 64 * 64})
        return Gr00tImageProcessor(**kwargs)

    @staticmethod
    def gradient_image():
        y, x = np.mgrid[0:45, 0:60]
        return np.stack([(x * 4) % 256, (y * 5) % 256, ((x * y) // 3) % 256], -1).astype(np.uint8)

    def test_crop_and_resize_matches_opencv(self):
        # Reference values computed with the original GR00T evaluation transform: letterbox padding,
        # `cv2.resize(..., interpolation=cv2.INTER_AREA)`, center crop of 95% of each side and `INTER_AREA` again
        image = torch.from_numpy(self.gradient_image()).permute(2, 0, 1)[None]
        output = self.get_image_processor().crop_and_resize(image, 32, 0.95)[0]

        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(tuple(output.shape), (3, 32, 32))
        self.assertEqual(output.reshape(3, -1).sum(-1).tolist(), [96581, 90048, 83350])
        self.assertEqual(output[0, 16, ::4].tolist(), [9, 37, 65, 93, 122, 150, 178, 206])
        self.assertEqual(output[1, ::4, 16].tolist(), [0, 12, 47, 82, 117, 152, 187, 128])
        # Letterbox padding of the 45x60 image
        self.assertEqual(output[:, :2].max().item(), 0)

    def test_constant_image_is_unchanged(self):
        image = torch.full((1, 3, 40, 40), 77, dtype=torch.uint8)
        output = self.get_image_processor().crop_and_resize(image, 32, 0.95)
        self.assertTrue(output.eq(77).all())

    def test_output_grid_is_independent_of_input_size(self):
        image_processor = self.get_image_processor()
        for height, width in [(32, 32), (45, 60), (120, 90)]:
            image = Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))
            outputs = image_processor(image, return_tensors="pt")
            self.assertEqual(outputs["image_grid_thw"].tolist(), [[1, 2, 2]])
            self.assertEqual(tuple(outputs["pixel_values"].shape), (4, 3 * 2 * 16 * 16))

    def test_float_inputs_match_uint8_inputs(self):
        image = self.gradient_image()
        image_processor = self.get_image_processor()
        from_uint8 = image_processor(image, return_tensors="pt")["pixel_values"]
        from_float = image_processor(image.astype(np.float32) / 255, do_rescale=False, return_tensors="pt")
        torch.testing.assert_close(from_float["pixel_values"], from_uint8)

    def test_save_load(self):
        image_processor = self.get_image_processor(crop_fraction=0.9)
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor.save_pretrained(tmpdirname)
            reloaded = Gr00tImageProcessor.from_pretrained(tmpdirname)
        self.assertEqual(reloaded.crop_fraction, 0.9)
        self.assertEqual(reloaded.image_resolution, 32)
        image = self.gradient_image()
        torch.testing.assert_close(
            reloaded(image, return_tensors="pt")["pixel_values"],
            image_processor(image, return_tensors="pt")["pixel_values"],
        )
