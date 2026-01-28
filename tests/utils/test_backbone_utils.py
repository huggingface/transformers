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

from transformers import DetrConfig, MaskFormerConfig, PreTrainedConfig, ResNetBackbone, ResNetConfig, TimmBackbone
from transformers.modeling_backbone_utils import (
    BackboneConfigMixin,
    BackboneMixin,
    load_backbone,
)
from transformers.testing_utils import require_torch, slow
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import BertPreTrainedModel


class AnyBackboneConfig(BackboneConfigMixin, PreTrainedConfig):
    def __init__(
        self,
        stage_names: list | None = None,
        out_indices: list | None = None,
        out_features: list | None = None,
        **kwargs,
    ):
        self.stage_names = stage_names
        self._out_features, self._out_indices = out_features, out_indices
        self.align_output_features_output_indices()

        super().__init__(**kwargs)


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

        with pytest.raises(ValueError, match="out_indices must be a list, got <class 'tuple'>"):
            AnyBackboneConfig(stage_names=["a", "b"], out_features=None, out_indices=(0, 1))

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

    def test_backbone_mixin(self):
        config = AnyBackboneConfig(stage_names=["a", "b", "c"], out_features=["a", "c"], out_indices=[0, 2])
        backbone = BackboneMixin()
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

    @slow
    @require_torch
    def test_load_backbone_from_config(self):
        """
        Test that load_backbone correctly loads a backbone from a backbone config.
        """
        config = MaskFormerConfig(backbone_config=ResNetConfig(out_indices=(0, 2)))
        backbone = load_backbone(config)
        self.assertEqual(backbone.out_features, ["stem", "stage2"])
        self.assertEqual(backbone.out_indices, (0, 2))
        self.assertIsInstance(backbone, ResNetBackbone)

    @slow
    @require_torch
    def test_load_backbone_from_checkpoint(self):
        """
        Test that load_backbone correctly loads a backbone from a checkpoint.
        """
        config = MaskFormerConfig(backbone="microsoft/resnet-18", backbone_config=None)
        backbone = load_backbone(config)
        self.assertEqual(backbone.out_indices, [4])
        self.assertEqual(backbone.out_features, ["stage4"])
        self.assertIsInstance(backbone, ResNetBackbone)

        config = MaskFormerConfig(
            backbone="resnet18",
            use_timm_backbone=True,
        )
        backbone = load_backbone(config)
        # We can't know ahead of time the exact output features and indices, or the layer names before
        # creating the timm model, so it defaults to the last layer (-1,) and has a different layer name
        self.assertEqual(backbone.out_indices, (-1,))
        self.assertEqual(backbone.out_features, ["layer4"])
        self.assertIsInstance(backbone, TimmBackbone)

    @slow
    @require_torch
    def test_load_backbone_backbone_kwargs(self):
        """
        Test that load_backbone correctly configures the loaded backbone with the provided kwargs.
        """
        config = MaskFormerConfig(backbone="resnet18", use_timm_backbone=True, backbone_kwargs={"out_indices": (0, 1)})
        backbone = load_backbone(config)
        self.assertEqual(backbone.out_indices, (0, 1))
        self.assertIsInstance(backbone, TimmBackbone)

        config = MaskFormerConfig(backbone="microsoft/resnet-18", backbone_kwargs={"out_indices": (0, 2)})
        backbone = load_backbone(config)
        self.assertEqual(backbone.out_indices, (0, 2))
        self.assertIsInstance(backbone, ResNetBackbone)

        # Check can't be passed with a backone config
        with pytest.raises(ValueError):
            config = MaskFormerConfig(
                backbone="microsoft/resnet-18",
                backbone_config=ResNetConfig(out_indices=(0, 2)),
                backbone_kwargs={"out_indices": (0, 1)},
            )

    @slow
    @require_torch
    def test_load_backbone_in_new_model(self):
        """
        Tests that new model can be created, with its weights instantiated and pretrained backbone weights loaded.
        """

        # Inherit from PreTrainedModel to ensure that the weights are initialized
        class NewModel(BertPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.backbone = load_backbone(config)
                self.layer_0 = torch.nn.Linear(config.hidden_size, config.hidden_size)
                self.layer_1 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        def get_equal_not_equal_weights(model_0, model_1):
            equal_weights = []
            not_equal_weights = []
            for (k0, v0), (k1, v1) in zip(model_0.named_parameters(), model_1.named_parameters()):
                self.assertEqual(k0, k1)
                weights_are_equal = torch.allclose(v0, v1)
                if weights_are_equal:
                    equal_weights.append(k0)
                else:
                    not_equal_weights.append(k0)
            return equal_weights, not_equal_weights

        config = MaskFormerConfig(use_pretrained_backbone=False, backbone="microsoft/resnet-18")
        model_0 = NewModel(config)
        model_1 = NewModel(config)
        equal_weights, not_equal_weights = get_equal_not_equal_weights(model_0, model_1)

        # Norm layers are always initialized with the same weights
        equal_weights = [w for w in equal_weights if "normalization" not in w]
        self.assertEqual(len(equal_weights), 0)
        self.assertEqual(len(not_equal_weights), 24)

        # Now we create a new model with backbone weights that are pretrained
        config.use_pretrained_backbone = True
        model_0 = NewModel(config)
        model_1 = NewModel(config)
        equal_weights, not_equal_weights = get_equal_not_equal_weights(model_0, model_1)

        # Norm layers are always initialized with the same weights
        equal_weights = [w for w in equal_weights if "normalization" not in w]
        self.assertEqual(len(equal_weights), 20)
        # Linear layers are still initialized randomly
        self.assertEqual(len(not_equal_weights), 4)

        # Check loading in timm backbone
        config = DetrConfig(use_pretrained_backbone=False, backbone="resnet18", use_timm_backbone=True)
        model_0 = NewModel(config)
        model_1 = NewModel(config)
        equal_weights, not_equal_weights = get_equal_not_equal_weights(model_0, model_1)

        # Norm layers are always initialized with the same weights
        equal_weights = [w for w in equal_weights if "bn" not in w and "downsample.1" not in w]
        self.assertEqual(len(equal_weights), 0)
        self.assertEqual(len(not_equal_weights), 24)

        # Now we create a new model with backbone weights that are pretrained
        config.use_pretrained_backbone = True
        model_0 = NewModel(config)
        model_1 = NewModel(config)
        equal_weights, not_equal_weights = get_equal_not_equal_weights(model_0, model_1)

        # Norm layers are always initialized with the same weights
        equal_weights = [w for w in equal_weights if "bn" not in w and "downsample.1" not in w]
        self.assertEqual(len(equal_weights), 20)
        # Linear layers are still initialized randomly
        self.assertEqual(len(not_equal_weights), 4)
