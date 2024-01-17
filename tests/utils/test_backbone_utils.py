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

import pytest

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
        with pytest.raises(ValueError, match="Stage_names must be set for transformers backbones"):
            verify_out_features_out_indices(["a", "b"], (0, 1), None)

        # Out features must be a list
        with pytest.raises(ValueError, match="out_features must be a list got <class 'tuple'>"):
            verify_out_features_out_indices(("a", "b"), (0, 1), ["a", "b"])

        # Out features must be a subset of stage names
        with pytest.raises(
            ValueError, match=r"out_features must be a subset of stage_names: \['a'\] got \['a', 'b'\]"
        ):
            verify_out_features_out_indices(["a", "b"], (0, 1), ["a"])

        # Out features must contain no duplicates
        with pytest.raises(ValueError, match=r"out_features must not contain any duplicates, got \['a', 'a'\]"):
            verify_out_features_out_indices(["a", "a"], None, ["a"])

        # Out indices must be a list or tuple
        with pytest.raises(ValueError, match="out_indices must be a list or tuple, got <class 'int'>"):
            verify_out_features_out_indices(None, 0, ["a", "b"])

        # Out indices must be a subset of stage names
        with pytest.raises(
            ValueError, match=r"out_indices must be valid indices for stage_names \['a'\], got \(0, 1\)"
        ):
            verify_out_features_out_indices(None, (0, 1), ["a"])

        # Out indices must contain no duplicates
        with pytest.raises(ValueError, match=r"out_indices must not contain any duplicates, got \(0, 0\)"):
            verify_out_features_out_indices(None, (0, 0), ["a"])

        # Out features and out indices must be the same length
        with pytest.raises(
            ValueError, match="out_features and out_indices should have the same length if both are set"
        ):
            verify_out_features_out_indices(["a", "b"], (0,), ["a", "b", "c"])

        # Out features should match out indices
        with pytest.raises(
            ValueError, match="out_features and out_indices should correspond to the same stages if both are set"
        ):
            verify_out_features_out_indices(["a", "b"], (0, 2), ["a", "b", "c"])

        # Out features and out indices should be in order
        with pytest.raises(
            ValueError,
            match=r"out_features must be in the same order as stage_names, expected \['a', 'b'\] got \['b', 'a'\]",
        ):
            verify_out_features_out_indices(["b", "a"], (0, 1), ["a", "b"])

        with pytest.raises(
            ValueError, match=r"out_indices must be in the same order as stage_names, expected \(-2, 1\) got \(1, -2\)"
        ):
            verify_out_features_out_indices(["a", "b"], (1, -2), ["a", "b"])

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
