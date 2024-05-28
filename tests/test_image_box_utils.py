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

import random
import unittest
from itertools import product

import numpy as np
from parameterized import parameterized

import transformers.image_box_utils as image_box_utils
from transformers import is_torch_available
from transformers.testing_utils import require_torch


random.seed(42463)

if is_torch_available():
    import torch


NORMAL_TESTING_CASES = {
    "image_size": [460, 640],
    "boxes": {
        "absolute_xyxy": [
            [387, 441, 431, 450],
            [134, 57, 568, 440],
            [306, 274, 342, 428],
            [22, 252, 501, 308],
            [498, 349, 618, 428],
        ],
        "absolute_xywh": [
            [387, 441, 44, 9],
            [134, 57, 434, 383],
            [306, 274, 36, 154],
            [22, 252, 479, 56],
            [498, 349, 120, 79],
        ],
        "absolute_xcycwh": [
            [409.0, 445.5, 44.0, 9.0],
            [351.0, 248.5, 434.0, 383.0],
            [324.0, 351.0, 36.0, 154.0],
            [261.5, 280.0, 479.0, 56.0],
            [558.0, 388.5, 120.0, 79.0],
        ],
        "relative_xyxy": [
            [0.6047, 0.9587, 0.6734, 0.9783],
            [0.2094, 0.1239, 0.8875, 0.9565],
            [0.4781, 0.5957, 0.5344, 0.9304],
            [0.0344, 0.5478, 0.7828, 0.6696],
            [0.7781, 0.7587, 0.9656, 0.9304],
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
    },
}


OOB_TESTING_CASES = {
    "image_size": [100, 200],  # height, width
    "boxes": {
        "absolute_xyxy": [
            [-1, 0, 9, 15],  # x_min oob
            [50, -1, 60, 9],  # y_min oob
            [190, 0, 200, 10],  # x_max oob
            [189, 90, 199, 100],  # y_max oob
        ],
        "absolute_xywh": [
            [-1, 0, 10, 15],  # x_min oob
            [50, -1, 10, 10],  # y_min oob
            [190, 0, 10, 10],  # x_max oob
            [189, 90, 10, 10],  # y_max oob
        ],
        "absolute_xcycwh": [
            [4, 7.5, 10, 15],  # x_min oob
            [55, 4, 10, 10],  # y_min oob
            [195, 5, 10, 10],  # x_max oob
            [194, 95, 10, 10],  # y_max oob
        ],
    },
}


def rescale_boxes(boxes, boxes_format, src_image_size, dst_image_size):
    if "relative" in boxes_format:
        return boxes
    src_height, src_width = src_image_size
    dst_height, dst_width = dst_image_size
    rescaled_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1 = (x1 / src_width) * dst_width
        x2 = (x2 / src_width) * dst_width
        y1 = (y1 / src_height) * dst_height
        y2 = (y2 / src_height) * dst_height
        rescaled_box = [x1, y1, x2, y2]
        # rescaled_box = [round(x, 4) for x in rescaled_box]
        rescaled_boxes.append(rescaled_box)
    return rescaled_boxes


def make_3d_cases_from_2d(cases_2d: dict):
    cases_3d = {
        "image_size": [],
        "boxes": {box_format: [] for box_format in cases_2d["boxes"]},
    }

    n_boxes = len(cases_2d["boxes"]["absolute_xyxy"])

    # we will sample form 2d cases with rescaling
    for n_samples in range(n_boxes + 1):
        scale = random.uniform(0.5, 2.0)
        choices = random.sample(range(n_boxes), n_samples)

        # rescale image size
        original_image_size = cases_2d["image_size"]
        new_image_size = [int(dim * scale) for dim in original_image_size]
        cases_3d["image_size"].append(new_image_size)

        # select and rescale boxes
        for boxes_format, boxes in cases_2d["boxes"].items():
            boxes = [boxes[i] for i in choices]
            rescaled_boxes = rescale_boxes(boxes, boxes_format, original_image_size, new_image_size)
            cases_3d["boxes"][boxes_format].append(rescaled_boxes)

    return cases_3d


@require_torch
class UtilBoundingBoxConverters(unittest.TestCase):
    @parameterized.expand(product(NORMAL_TESTING_CASES["boxes"].keys(), repeat=2))
    def test_normal_cases_1d(self, input_format, output_format):
        image_size = NORMAL_TESTING_CASES["image_size"]
        boxes_expected = NORMAL_TESTING_CASES["boxes"][output_format]
        boxes_input = NORMAL_TESTING_CASES["boxes"][input_format]

        # we will just iterate over 2d cases
        for box_input, box_expected in zip(boxes_input, boxes_expected):
            box_converted = image_box_utils.convert_boxes(
                boxes=box_input, input_format=input_format, output_format=output_format, image_size=image_size
            )

            box_converted = np.array(box_converted)
            box_expected = np.array(box_expected)

            tolerance = 1e-4 if "relative" in output_format else 0.05
            self.assertTrue(np.allclose(box_converted, box_expected, atol=tolerance))

    @parameterized.expand(product(NORMAL_TESTING_CASES["boxes"].keys(), repeat=2))
    def test_normal_cases_2d(self, input_format, output_format):
        image_size = NORMAL_TESTING_CASES["image_size"]
        boxes_expected = NORMAL_TESTING_CASES["boxes"][output_format]
        boxes_input = NORMAL_TESTING_CASES["boxes"][input_format]

        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_input, input_format=input_format, output_format=output_format, image_size=image_size
        )

        boxes_converted = np.array(boxes_converted)
        boxes_expected = np.array(boxes_expected)

        tolerance = 1e-4 if "relative" in output_format else 0.05
        self.assertTrue(np.allclose(boxes_converted, boxes_expected, atol=tolerance))

    @parameterized.expand(product(NORMAL_TESTING_CASES["boxes"].keys(), repeat=2))
    def test_normal_cases_3d(self, input_format, output_format):
        cases_3d = make_3d_cases_from_2d(NORMAL_TESTING_CASES)
        image_size = cases_3d["image_size"]
        boxes_expected = cases_3d["boxes"][output_format]
        boxes_input = cases_3d["boxes"][input_format]

        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_input, input_format=input_format, output_format=output_format, image_size=image_size
        )

        for image_boxes_expected, image_boxes_converted in zip(boxes_expected, boxes_converted):
            image_boxes_converted = np.array(image_boxes_converted)
            image_boxes_expected = np.array(image_boxes_expected)

            tolerance = 1e-4 if "relative" in output_format else 0.1
            self.assertTrue(np.allclose(image_boxes_converted, image_boxes_expected, atol=tolerance))

    @parameterized.expand(product(NORMAL_TESTING_CASES["boxes"].keys(), repeat=2))
    def test_normal_cases_3d_arrays(self, input_format, output_format):
        image_size = NORMAL_TESTING_CASES["image_size"]
        boxes_expected = NORMAL_TESTING_CASES["boxes"][output_format]
        boxes_input = NORMAL_TESTING_CASES["boxes"][input_format]

        # numpy
        image_size = np.array([image_size] * 3)
        boxes_input = np.array([boxes_input] * 3)
        boxes_expected = np.array([boxes_expected] * 3)

        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_input, input_format=input_format, output_format=output_format, image_size=image_size
        )

        tolerance = 1e-4 if "relative" in output_format else 0.1
        self.assertIsInstance(boxes_converted, np.ndarray)
        self.assertTrue(np.allclose(boxes_converted, boxes_expected, atol=tolerance))

        # torch
        boxes_input = torch.tensor(boxes_input).float()
        boxes_expected = torch.tensor(boxes_expected).float()

        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_input, input_format=input_format, output_format=output_format, image_size=image_size
        )

        tolerance = 1e-4 if "relative" in output_format else 0.1
        self.assertIsInstance(boxes_converted, torch.Tensor)
        self.assertTrue(torch.allclose(boxes_converted, boxes_expected, atol=tolerance))

    @parameterized.expand(product(OOB_TESTING_CASES["boxes"].keys(), repeat=2))
    def test_oob_cases(self, input_format, output_format):
        image_size = OOB_TESTING_CASES["image_size"]
        boxes_expected = OOB_TESTING_CASES["boxes"][output_format]
        boxes_input = OOB_TESTING_CASES["boxes"][input_format]

        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_input, input_format=input_format, output_format=output_format, image_size=image_size
        )

        boxes_converted = np.array(boxes_converted)
        boxes_expected = np.array(boxes_expected)

        self.assertTrue(np.allclose(boxes_converted, boxes_expected, atol=1e-4))

    @parameterized.expand(
        [
            [[-0.5, 0.1, 0.5, 0.9], "relative_xyxy"],  # negative x_min
            [[0.5, 0.1, 0.5, 1.1], "relative_xyxy"],  # x_max > 1
            [[0.1, 0.1, 0.21, 0.21], "relative_xcycwh"],  # xc - width < 0
            [[0.8, 0.7, 0.21, 0.21], "relative_xywh"],  # xc + width > 1
            [[0.7, 0.8, 0.21, 0.21], "relative_xywh"],  # yc + height > 1
            [[0.7, 0.8, -0.1, 0.21], "relative_xywh"],  # negative width
            [[-5, 0, 10, 10], "absolute_xywh"],  # negative x_min
            [[0, -5, 10, 10], "absolute_xywh"],  # negative y_min
            [[20, 20, -10, 10], "absolute_xywh"],  # negative width
            [[-1, 0, 10, 10], "absolute_xyxy", (100, 100)],  # negative x_min
            [[10, 0, 9, 10], "absolute_xyxy", (100, 100)],  # x_max < x_min -> negative width
            # check for upper bound is not implemented for absolute boxes
        ]
    )
    def test_oob_exceptions(self, box, box_format, image_size=(100, 100)):
        with self.assertRaises(image_box_utils.BoxOutOfBoundsError):
            image_box_utils.convert_boxes(
                boxes=box, input_format=box_format, output_format="absolute_xyxy", image_size=image_size, check="raise"
            )

    @parameterized.expand(
        [
            ["tensor", 3],
            ["tensor", 2],
            ["array", 3],
            ["array", 2],
            ["tuple", 1],
            ["tuple", 2],
            ["tuple", 3],
            ["list", 1],
            ["list", 2],
            ["list", 3],
        ]
    )
    def test_different_types_support(self, dtype, from_depth):
        type_mapping = {
            "tensor": torch.tensor,
            "array": np.array,
            "tuple": tuple,
            "list": list,
        }
        boxes_absolute_xyxy = [
            [
                [10, 20, 30, 40],
                [20, 30, 40, 50],
            ],
            [
                [30, 40, 50, 60],
            ],
            [
                # image with no boxes
            ],
        ]
        image_sizes = [(100, 100), (200, 200), (300, 300)]

        def convert_to_dtype(data, to_dtype, depth):
            if depth == 1:
                return to_dtype(data)
            return [convert_to_dtype(d, to_dtype, depth - 1) for d in data]

        to_dtype = type_mapping[dtype]
        boxes_original = convert_to_dtype(boxes_absolute_xyxy, to_dtype, from_depth)

        # make two conversions
        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_original,
            input_format="absolute_xyxy",
            output_format="relative_xcycwh",
            image_size=image_sizes,
        )
        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_converted,
            input_format="relative_xcycwh",
            output_format="absolute_xyxy",
            image_size=image_sizes,
        )

        # check if the conversion is correct
        for boxes_src, boxes_dst in zip(boxes_original, boxes_converted):
            boxes_src = np.array(boxes_src)
            boxes_dst = np.array(boxes_dst)
            self.assertTrue(np.allclose(boxes_src, boxes_dst, atol=1e-4))

    @parameterized.expand(
        [
            ["coco", "absolute_xywh"],
            ["pascal_voc", "absolute_xyxy"],
            ["albumentations", "relative_xyxy"],
            ["yolo", "relative_xcycwh"],
            ["xyxy", "absolute_xyxy"],
            ["xywh", "absolute_xywh"],
            ["xcycwh", "relative_xcycwh"],
        ]
    )
    def test_aliases(self, alias, box_format):
        boxes_expected = np.array(NORMAL_TESTING_CASES["boxes"][box_format])
        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes_expected,
            input_format=alias,
            output_format=box_format,
        )
        self.assertTrue(np.allclose(boxes_converted, boxes_expected, atol=1e-4))

    def test_wrong_input_exceptions(self):
        # 4d tensor/array
        boxes = np.ones((2, 2, 2, 4))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="absolute_xyxy", output_format="absolute_xywh")

        # incorrect boxes, just 3 coordinates
        boxes = np.ones((2, 2, 3))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="absolute_xyxy", output_format="absolute_xywh")

        # image size is not provided
        boxes = np.ones((2, 2, 4))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="absolute_xyxy", output_format="relative_xywh")

        # image size with wrong length
        boxes = np.ones((2, 2, 4))
        image_size = np.ones((1, 2))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(
                boxes=boxes, input_format="absolute_xyxy", output_format="relative_xywh", image_size=image_size
            )

        # incorrect input format
        boxes = np.ones((2, 2, 4))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="format", output_format="relative_xywh")

        # incorrect output format
        boxes = np.ones((2, 2, 4))
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="absolute_xyxy", output_format="format")

        # variable length boxes
        boxes = [
            [10, 20, 30, 40],
            [20, 30, 40],
        ]
        with self.assertRaises(ValueError):
            image_box_utils.convert_boxes(boxes=boxes, input_format="absolute_xyxy", output_format="absolute_xywh")

    def test_normal_boxes_with_extra(self):
        # random boxes relative_xcycwh, add to make them valid
        boxes = np.random.rand(5, 6) / 2 + 0.5
        boxes_converted = image_box_utils.convert_boxes(
            boxes=boxes, input_format="relative_xcycwh", output_format="absolute_xyxy", image_size=[100, 100]
        )

        extra = boxes[..., 4:]
        converted_extra = boxes_converted[..., 4:]
        self.assertTrue(np.allclose(extra, converted_extra, atol=1e-4))
