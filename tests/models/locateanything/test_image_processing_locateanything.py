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
from tempfile import TemporaryDirectory

import numpy as np

from transformers import AutoImageProcessor, LocateAnythingImageProcessor
from transformers.testing_utils import require_torch, require_vision


@require_torch
@require_vision
class LocateAnythingImageProcessingTest(unittest.TestCase):
    def test_preprocess_patchifies_images(self):
        from PIL import Image

        image_processor = LocateAnythingImageProcessor()
        image = Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8))

        outputs = image_processor(image, return_tensors="pt")

        self.assertEqual(outputs["pixel_values"].shape, (4, 3, 14, 14))
        self.assertEqual(outputs["image_grid_hws"].tolist(), [[2, 2]])
        self.assertEqual(outputs["pixel_values"].min().item(), -1.0)
        self.assertEqual(outputs["pixel_values"].max().item(), -1.0)

    def test_auto_image_processor_from_pretrained(self):
        with TemporaryDirectory() as tmp_dir:
            LocateAnythingImageProcessor().save_pretrained(tmp_dir)
            image_processor = AutoImageProcessor.from_pretrained(tmp_dir)

        self.assertIsInstance(image_processor, LocateAnythingImageProcessor)


if __name__ == "__main__":
    unittest.main()
