# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch
from transformers.utils.bounding_box_utils import BoundingBoxFormat, transform_box_format


if is_torch_available():
    import torch


@require_torch
class UtilBoundingBoxConverters(unittest.TestCase):
    def test_enumerators(self):
        self.assertEqual(BoundingBoxFormat.XYXY, "xyxy")
        self.assertEqual(BoundingBoxFormat.XYWH, "xywh")
        self.assertEqual(BoundingBoxFormat.XCYCWH, "xcycwh")
        self.assertEqual(BoundingBoxFormat.RELATIVE_XYWH, "relative_xywh")
        self.assertEqual(BoundingBoxFormat.RELATIVE_XCYCWH, "relative_xcycwh")

    def test_conversion_cases(self):
        # Identical bounding boxes differing in format only
        testing_cases = {
            "xywh": [
                [387, 441, 44, 9],
                [134, 57, 434, 383],
                [306, 274, 36, 154],
                [22, 252, 479, 56],
                [498, 349, 120, 79],
            ],
            "xyxy": [
                [387, 441, 431, 450.0],
                [134, 57, 568, 440.0],
                [306, 274, 342, 428.0],
                [22, 252, 501, 308.0],
                [498, 349, 618, 428],
            ],
            "xcycwh": [
                [409.0, 445.5, 44.0, 9.0],
                [351.0, 248.5, 434.0, 383.0],
                [324.0, 351.0, 36.0, 154.0],
                [261.5, 280.0, 479.0, 56.0],
                [558.0, 388.5, 120.0, 79.0],
            ],
            "relative_xywh": [
                [0.6047, 0.9587, 0.0688, 0.0196],
                [0.2094, 0.1239, 0.6781, 0.8326],
                [0.4781, 0.5957, 0.05625, 0.3348],
                [0.0344, 0.5478, 0.7484, 0.1217],
                [0.7781, 0.7587, 0.1875, 0.1717],
            ],
            "relative_xcycwh": [
                [0.6391, 0.9685, 0.0688, 0.0196],
                [0.5484, 0.5402, 0.6781, 0.8326],
                [0.5063, 0.7630, 0.0563, 0.3348],
                [0.4086, 0.6087, 0.7484, 0.1217],
                [0.8719, 0.8446, 0.1875, 0.1717],
            ],
        }
        # Resolution of the image where the boxes belong to
        original_img_shape = [460, 640]

        # Define which formats are relative and absolute
        rel_formats = ("relative_xywh", "relative_xcycwh")

        # Loop through each format in the testing cases
        for origin_format, bboxes in testing_cases.items():
            origin_format = BoundingBoxFormat(origin_format)
            bboxes = torch.Tensor(bboxes)

            # Loop through all formats to be used as destination formats
            for dest_format, expected_bboxes in testing_cases.items():
                dest_format = BoundingBoxFormat(dest_format)
                expected_bboxes = torch.Tensor(expected_bboxes)
                img_shape = None
                # If original or destination formats are relative, the image shape is required
                if dest_format in rel_formats or origin_format in rel_formats:
                    img_shape = original_img_shape

                # Apply transformation
                result = transform_box_format(
                    bboxes, orig_format=origin_format, dest_format=dest_format, img_shape=img_shape, do_round=True
                )
                # Compare results
                self.assertTrue(torch.allclose(result, expected_bboxes, atol=1e-4))
