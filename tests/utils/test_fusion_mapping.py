# Copyright 2026 The HuggingFace Inc. team.
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

import sys
import types
import unittest
from copy import deepcopy

import torch.nn as nn

from transformers import PretrainedConfig, conversion_mapping
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import Conv3dToLinear, WeightConverter
from transformers.fusion_mapping import _FUSION_DISCOVERY_CACHE, register_fusion_patches
from transformers.modeling_utils import PreTrainedModel
from transformers.monkey_patching import apply_patches, clear_patch_mapping, get_patch_mapping


DUMMY_TRANSFORMERS_MODULE_NAME = "transformers.test_fusion_mapping_dummy"
# `apply_patches()` only rewrites classes exposed from `transformers.*` modules.
DUMMY_TRANSFORMERS_MODULE = types.ModuleType(DUMMY_TRANSFORMERS_MODULE_NAME)
sys.modules[DUMMY_TRANSFORMERS_MODULE_NAME] = DUMMY_TRANSFORMERS_MODULE


class DummyFusionConfig(PretrainedConfig):
    model_type = "dummy_fusion"

    def __init__(self, patch_embed_stride=(2, 2, 2), patch_embed_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed_stride = patch_embed_stride
        self.patch_embed_bias = patch_embed_bias


class DummyPatchEmbedding(nn.Module):
    def __init__(self, stride=(2, 2, 2), bias=False):
        super().__init__()
        self.embed_dim = 8
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=(2, 2, 2), stride=stride, bias=bias)


class DummyFusionModel(PreTrainedModel):
    config_class = DummyFusionConfig
    patchable_classes = {
        "DummyPatchEmbedding": DummyPatchEmbedding,
    }

    def __init__(self, config):
        super().__init__(config)
        # Instantiate through the fake module so `apply_patches()` sees the replacement.
        self.patch_embed = DUMMY_TRANSFORMERS_MODULE.DummyPatchEmbedding(
            stride=tuple(config.patch_embed_stride), bias=config.patch_embed_bias
        )


for class_name, patchable_class in DummyFusionModel.patchable_classes.items():
    setattr(DUMMY_TRANSFORMERS_MODULE, class_name, patchable_class)


class FusionMappingTest(unittest.TestCase):
    """Covers discovery, no-match, and conflict handling for fusion mapping."""

    fusion_config = {"patch_embeddings": True}

    def setUp(self):
        clear_patch_mapping()
        _FUSION_DISCOVERY_CACHE.clear()
        self.checkpoint_conversion_mapping_cache = deepcopy(conversion_mapping._checkpoint_conversion_mapping_cache)

    def tearDown(self):
        clear_patch_mapping()
        _FUSION_DISCOVERY_CACHE.clear()
        conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(self.checkpoint_conversion_mapping_cache)

    def test_register_fusion_patches_is_effective_on_dummy_model(self):
        # Registers and applies a fusion on a dummy model.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig(patch_embed_stride=(2, 2, 2), patch_embed_bias=True)

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(config.model_type))
        self.assertIsInstance(DummyFusionModel(config).patch_embed.proj, nn.Conv3d)

        register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

        self.assertEqual(len(get_patch_mapping()), 1)
        self.assertEqual(len(get_checkpoint_conversion_mapping(config.model_type)), 2)

        with apply_patches():
            fused_model = DummyFusionModel(config)

        fused_projection = getattr(
            fused_model.patch_embed, "linear_proj", getattr(fused_model.patch_embed, "proj", None)
        )
        self.assertIsInstance(fused_projection, nn.Linear)

    def test_register_fusion_patches_skips_when_no_modules_match(self):
        # Leaves registries untouched when nothing is fusable.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig(patch_embed_stride=(1, 1, 1))

        register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(config.model_type))

    def test_register_fusion_patches_raises_on_transform_conflicts(self):
        # Rejects transforms that would shadow an existing source pattern.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig(patch_embed_stride=(2, 2, 2))
        model_type = config.model_type

        # build a conflicting conversion mapping with the same source pattern but different target pattern
        register_checkpoint_conversion_mapping(
            model_type,
            [
                WeightConverter(
                    source_patterns="patch_embed.proj.weight",
                    target_patterns="patch_embed.other_linear_proj.weight",
                    operations=[Conv3dToLinear(in_channels=3, kernel_size=(2, 2, 2))],
                )
            ],
            overwrite=True,
        )

        with self.assertRaisesRegex(ValueError, "conflicts with an existing conversion mapping"):
            register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)
