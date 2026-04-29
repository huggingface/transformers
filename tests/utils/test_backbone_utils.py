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

from transformers import PreTrainedConfig
from transformers.backbone_utils import (
    BackboneConfigMixin,
    BackboneMixin,
)
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    from transformers import PreTrainedModel


class AnyBackboneConfig(BackboneConfigMixin, PreTrainedConfig):
    def __init__(
        self,
        stage_names: list | None = None,
        out_indices: list | None = None,
        out_features: list | None = None,
        **kwargs,
    ):
        self.stage_names = stage_names
        self.set_output_features_output_indices(out_features=out_features, out_indices=out_indices)

        super().__init__(**kwargs)


@require_torch
class AnyBackbone(BackboneMixin, PreTrainedModel): ...


class BackboneUtilsTester(unittest.TestCase):
    def test_get_aligned_output_features_output_indices(self):
        stage_names = ["a", "b", "c"]

        # Defaults to last layer if both, `out_indices` and `out_features`, are None
        config = AnyBackboneConfig(stage_names)
        self.assertEqual(config.out_features, ["c"])
        self.assertEqual(config.out_indices, [2])

        # Out indices set to match out features
        config = AnyBackboneConfig(stage_names=stage_names, out_features=["a", "c"])
        self.assertEqual(config.out_features, ["a", "c"])
        self.assertEqual(config.out_indices, [0, 2])

        # Out features set to match out indices
        config = AnyBackboneConfig(stage_names=stage_names, out_indices=[0, 2])
        self.assertEqual(config.out_features, ["a", "c"])
        self.assertEqual(config.out_indices, [0, 2])

        # Out features selected from negative indices
        config = AnyBackboneConfig(stage_names=stage_names, out_indices=[-3, -1])
        self.assertEqual(config.out_features, ["a", "c"])
        self.assertEqual(config.out_indices, [-3, -1])

    def test_config_verify_out_features_out_indices(self):
        # Stage names must be set
        with pytest.raises(ValueError, match="Stage_names must be set for transformers backbones"):
            AnyBackboneConfig(stage_names=None, out_features=["a", "b"], out_indices=(0, 1))

        # Out features must be a list
        with pytest.raises(ValueError, match="out_features must be a list got <class 'tuple'>"):
            AnyBackboneConfig(stage_names=["a", "b"], out_features=("a", "b"), out_indices=[0, 1])

        # Out features must be a subset of stage names
        with pytest.raises(
            ValueError, match=r"out_features must be a subset of stage_names: \['a'\] got \['a', 'b'\]"
        ):
            AnyBackboneConfig(stage_names=["a"], out_features=["a", "b"], out_indices=[0, 1])

        # Out features must contain no duplicates
        with pytest.raises(ValueError, match=r"out_features must not contain any duplicates, got \['a', 'a'\]"):
            AnyBackboneConfig(stage_names=["a"], out_features=["a", "a"], out_indices=None)

        # Out indices must be a list
        with pytest.raises(ValueError, match="out_indices must be a list, got <class 'int'>"):
            AnyBackboneConfig(stage_names=["a", "b"], out_features=None, out_indices=0)

        # Out indices must be a subset of stage names
        with pytest.raises(
            ValueError, match=r"out_indices must be valid indices for stage_names \['a'\], got \[0, 1\]"
        ):
            AnyBackboneConfig(stage_names=["a"], out_features=None, out_indices=[0, 1])

        # Out indices must contain no duplicates
        with pytest.raises(ValueError, match=r"out_indices must not contain any duplicates, got \[0, 0\]"):
            AnyBackboneConfig(stage_names=["a"], out_features=None, out_indices=[0, 0])

        # Out features and out indices must be the same length
        with pytest.raises(
            ValueError, match="out_features and out_indices should have the same length if both are set"
        ):
            AnyBackboneConfig(stage_names=["a", "b", "c"], out_features=["a", "b"], out_indices=[0])

        # Out features should match out indices
        with pytest.raises(
            ValueError, match="out_features and out_indices should correspond to the same stages if both are set"
        ):
            AnyBackboneConfig(stage_names=["a", "b", "c"], out_features=["a", "b"], out_indices=[0, 2])

        # Out features and out indices should be in order
        with pytest.raises(
            ValueError,
            match=r"out_features must be in the same order as stage_names, expected \['a', 'b'\] got \['b', 'a'\]",
        ):
            AnyBackboneConfig(stage_names=["a", "b"], out_features=["b", "a"], out_indices=[0, 1])

        with pytest.raises(
            ValueError, match=r"out_indices must be in the same order as stage_names, expected \[-2, 1\] got \[1, -2\]"
        ):
            AnyBackboneConfig(stage_names=["a", "b"], out_features=["a", "b"], out_indices=[1, -2])

        # Check passes with valid inputs
        AnyBackboneConfig(stage_names=["a", "b", "c", "d"], out_features=["a", "b", "d"], out_indices=[0, 1, -1])

    @require_torch
    def test_backbone_mixin(self):
        config = AnyBackboneConfig(stage_names=["a", "b", "c"], out_features=["a", "c"], out_indices=[0, 2])
        backbone = AnyBackbone(config)
        backbone.config = config

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
