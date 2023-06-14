# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from transformers.image_processing_utils import get_size_dict


class ImageProcessingUtilsTester(unittest.TestCase):
    def test_get_size_dict(self):
        # Test a dict with the wrong keys raises an error
        inputs = {"wrong_key": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        inputs = {"height": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        inputs = {"width": 224, "shortest_edge": 224}
        with self.assertRaises(ValueError):
            get_size_dict(inputs)

        # Test a dict with the correct keys is returned as is
        inputs = {"height": 224, "width": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, inputs)

        inputs = {"shortest_edge": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {"shortest_edge": 224})

        inputs = {"longest_edge": 224, "shortest_edge": 224}
        outputs = get_size_dict(inputs)
        self.assertEqual(outputs, {"longest_edge": 224, "shortest_edge": 224})

        # Test a single int value which  represents (size, size)
        outputs = get_size_dict(224)
        self.assertEqual(outputs, {"height": 224, "width": 224})

        # Test a single int value which represents the shortest edge
        outputs = get_size_dict(224, default_to_square=False)
        self.assertEqual(outputs, {"shortest_edge": 224})

        # Test a tuple of ints which represents (height, width)
        outputs = get_size_dict((150, 200))
        self.assertEqual(outputs, {"height": 150, "width": 200})

        # Test a tuple of ints which represents (width, height)
        outputs = get_size_dict((150, 200), height_width_order=False)
        self.assertEqual(outputs, {"height": 200, "width": 150})

        # Test an int representing the shortest edge and max_size which represents the longest edge
        outputs = get_size_dict(224, max_size=256, default_to_square=False)
        self.assertEqual(outputs, {"shortest_edge": 224, "longest_edge": 256})

        # Test int with default_to_square=True and max_size fails
        with self.assertRaises(ValueError):
            get_size_dict(224, max_size=256, default_to_square=True)
