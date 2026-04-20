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
import tempfile
import types
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch.nn as nn

import transformers.conversion_mapping as conversion_mapping
import transformers.fusion_mapping as fusion_mapping
import transformers.monkey_patching as monkey_patching
from transformers import PretrainedConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import Conv3dToLinear, WeightConverter
from transformers.fusion_mapping import register_fusion_patches
from transformers.modeling_utils import PreTrainedModel
from transformers.monkey_patching import apply_patches, get_patch_mapping


DUMMY_TRANSFORMERS_MODULE_NAME = "transformers.test_fusion_mapping_dummy"
# `apply_patches()` scans `sys.modules` and only rewrites class attributes exposed
# from `transformers.*` modules, so this dummy class must be reachable through a
# fake `transformers` module instead of only through a local symbol.
DUMMY_TRANSFORMERS_MODULE = types.ModuleType(DUMMY_TRANSFORMERS_MODULE_NAME)
sys.modules[DUMMY_TRANSFORMERS_MODULE_NAME] = DUMMY_TRANSFORMERS_MODULE


class DummyVisionConfig(PretrainedConfig):
    model_type = "dummy_fusion_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        patch_embed_stride=(2, 2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_embed_stride = patch_embed_stride


class DummyFusionConfig(PretrainedConfig):
    model_type = "dummy_fusion"
    sub_configs = {"vision_config": DummyVisionConfig}

    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = DummyVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DummyVisionConfig(**vision_config)

        self.vision_config = vision_config


class DummyPatchEmbedding(nn.Module):
    def __init__(self, stride=(2, 2, 2), bias=False):
        super().__init__()
        self.embed_dim = 8
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=(2, 2, 2), stride=stride, bias=bias)


DUMMY_PATCHABLE_CLASSES = {"DummyPatchEmbedding": DummyPatchEmbedding}

for class_name, patchable_class in DUMMY_PATCHABLE_CLASSES.items():
    setattr(DUMMY_TRANSFORMERS_MODULE, class_name, patchable_class)


class DummyFusionModel(PreTrainedModel):
    config_class = DummyFusionConfig

    def __init__(self, config):
        super().__init__(config)
        # Resolve the class through the fake `transformers.*` module so monkey patching
        # can replace it before instantiation.
        self.patch_embed = DUMMY_TRANSFORMERS_MODULE.DummyPatchEmbedding(
            stride=config.vision_config.patch_embed_stride, bias=True
        )
        self.post_init()


class FusionMappingTest(unittest.TestCase):
    """Covers registration, no-match, and conflict handling for fusion mapping."""

    fusion_config = {"patch_embeddings": True}

    def setUp(self):
        self.patch_mapping_patcher = patch.object(monkey_patching, "_monkey_patch_mapping_cache", {})
        self.patch_mapping_patcher.start()
        self.discovery_cache_patcher = patch.object(fusion_mapping, "_FUSION_DISCOVERY_CACHE", {})
        self.discovery_cache_patcher.start()
        self.checkpoint_conversion_mapping_cache = deepcopy(conversion_mapping._checkpoint_conversion_mapping_cache)

    def tearDown(self):
        self.patch_mapping_patcher.stop()
        self.discovery_cache_patcher.stop()
        conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(self.checkpoint_conversion_mapping_cache)

    def test_register_fusion_patches_is_effective_on_dummy_model(self):
        # Registers and applies a fusion on a dummy model.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig()

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
        config = DummyFusionConfig(vision_config={"patch_embed_stride": (1, 1, 1)})

        register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(config.model_type))

    def test_register_fusion_patches_raises_on_transform_conflicts(self):
        # Rejects transforms that would shadow an existing source pattern.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig()
        model_type = config.model_type

        # build a conflicting conversion mapping with the same source pattern but different target pattern
        register_checkpoint_conversion_mapping(
            model_type,
            [
                WeightConverter(
                    source_patterns=r"patch_embed\.proj\.weight$",
                    target_patterns=r"patch_embed\.other_linear_proj\.weight$",
                    operations=[Conv3dToLinear(in_channels=3, kernel_size=(2, 2, 2))],
                )
            ],
            overwrite=True,
        )

        with self.assertRaisesRegex(ValueError, "conflicts with an existing conversion mapping"):
            register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

    def test_from_pretrained_uses_serialized_fusion_config(self):
        # A serialized `fusion_config` is reused on a later load.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"

        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as fused_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)

            fused_model = DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            fused_model.save_pretrained(fused_dir)

            # Simulate a fresh process so the second load comes only from the serialized config.
            monkey_patching._monkey_patch_mapping_cache.clear()
            fusion_mapping._FUSION_DISCOVERY_CACHE.clear()
            conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(
                self.checkpoint_conversion_mapping_cache
            )

            reloaded_model = DummyFusionModel.from_pretrained(fused_dir)

        fused_projection = getattr(
            reloaded_model.patch_embed, "linear_proj", getattr(reloaded_model.patch_embed, "proj", None)
        )
        self.assertIsInstance(fused_projection, nn.Linear)
