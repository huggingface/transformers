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

from transformers.utils.backbone_utils import (
    BackboneMixin,
    get_aligned_output_features_output_indices,
    verify_out_features_out_indices,
)


class BackboneUtilsTester(unittest.TestCase):
    def test_get_aligned_output_features_output_indices(self):
        stage_names = ["a", "b", "c"]

        # Defaults to last layer if both are None
        out_features, out_indices = get_aligned_output_features_output_indices(None, None, stage_names)
        self.assertEqual(out_features, ["c"])
        self.assertEqual(out_indices, [2])

        # Out indices set to match out features
        out_features, out_indices = get_aligned_output_features_output_indices(["a", "c"], None, stage_names)
        self.assertEqual(out_features, ["a", "c"])
        self.assertEqual(out_indices, [0, 2])

        # Out features set to match out indices
        out_features, out_indices = get_aligned_output_features_output_indices(None, [0, 2], stage_names)
        self.assertEqual(out_features, ["a", "c"])
        self.assertEqual(out_indices, [0, 2])

        # Out features selected from negative indices
        out_features, out_indices = get_aligned_output_features_output_indices(None, [-3, -1], stage_names)
        self.assertEqual(out_features, ["a", "c"])
        self.assertEqual(out_indices, [-3, -1])

    def test_verify_out_features_out_indices(self):
        # Stage names must be set
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(["a", "b"], (0, 1), None)

        # Out features must be a list
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(("a", "b"), (0, 1), ["a", "b"])

        # Out features must be a subset of stage names
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(["a", "b"], (0, 1), ["a"])

        # Out indices must be a list or tuple
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(None, 0, ["a", "b"])

        # Out indices must be a subset of stage names
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(None, (0, 1), ["a"])

        # Out features and out indices must be the same length
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(["a", "b"], (0,), ["a", "b", "c"])

        # Out features should match out indices
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(["a", "b"], (0, 2), ["a", "b", "c"])

        # Out features and out indices should be in order
        with self.assertRaises(ValueError):
            verify_out_features_out_indices(["b", "a"], (0, 1), ["a", "b"])

        # Check passes with valid inputs
        verify_out_features_out_indices(["a", "b", "d"], (0, 1, -1), ["a", "b", "c", "d"])

    def test_backbone_mixin(self):
        backbone = BackboneMixin()

        backbone.stage_names = ["a", "b", "c"]
        backbone._out_features = ["a", "c"]
        backbone._out_indices = [0, 2]

        # Check that the output features and indices are set correctly
        self.assertEqual(backbone.out_features, ["a", "c"])
        self.assertEqual(backbone.out_indices, [0, 2])

        # Check out features and indices are updated correctly
        backbone.out_features = ["a", "b"]
        self.assertEqual(backbone.out_features, ["a", "b"])
        self.assertEqual(backbone.out_indices, [0, 1])

        backbone.out_indices = [-3, -1]
        self.assertEqual(backbone.out_features, ["a", "c"])
        self.assertEqual(backbone.out_indices, [-3, -1])
