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

import torch
import torch.nn as nn

import transformers.conversion_mapping as conversion_mapping
import transformers.fusion_mapping as fusion_mapping
import transformers.monkey_patching as monkey_patching
from transformers import PretrainedConfig
from transformers.conversion_mapping import get_model_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import Conv3dToLinear, WeightConverter
from transformers.fusion_mapping import get_fusion_patch_mapping, get_fusion_weight_transforms
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
    """Covers discovery, no-match, conflict handling and per-load scoping for fusion mapping."""

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

    def test_fusion_patch_mapping_fuses_dummy_model(self):
        config = DummyFusionConfig()

        patch_mapping = get_fusion_patch_mapping(DummyFusionModel, config, self.fusion_config)
        with apply_patches(patch_mapping):
            fused_model = DummyFusionModel(config)

        self.assertEqual(list(patch_mapping), ["DummyPatchEmbedding"])
        self.assertIsInstance(fused_model.patch_embed.linear_proj, nn.Linear)
        self.assertEqual(
            [transform.source_patterns for transform in get_fusion_weight_transforms(fused_model)],
            [[r"patch_embed\.proj\.weight"], [r"patch_embed\.proj\.bias"]],
        )
        self.assertEqual(get_patch_mapping(), {})
        self.assertIsInstance(DummyFusionModel(config).patch_embed.proj, nn.Conv3d)

    def test_fusion_patch_mapping_is_empty_when_no_modules_match(self):
        config = DummyFusionConfig(vision_config={"patch_embed_stride": (1, 1, 1)})

        self.assertEqual(get_fusion_patch_mapping(DummyFusionModel, config, self.fusion_config), {})

    def test_fusion_transforms_raise_on_conversion_mapping_conflicts(self):
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig()
        register_checkpoint_conversion_mapping(
            config.model_type,
            [
                WeightConverter(
                    source_patterns=r"patch_embed\.proj\.weight",
                    target_patterns=r"patch_embed\.other_linear_proj\.weight",
                    operations=[Conv3dToLinear(in_channels=3, kernel_size=(2, 2, 2))],
                )
            ],
            overwrite=True,
        )
        with apply_patches(get_fusion_patch_mapping(DummyFusionModel, config, self.fusion_config)):
            fused_model = DummyFusionModel(config)

        with self.assertRaisesRegex(ValueError, "conflict with its conversion mapping"):
            get_model_conversion_mapping(fused_model)

    def test_fusion_only_applies_to_the_load_that_requests_it(self):
        model = DummyFusionModel(DummyFusionConfig())

        with tempfile.TemporaryDirectory() as source_dir:
            model.save_pretrained(source_dir)
            DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            plain_model, plain_loading_info = DummyFusionModel.from_pretrained(source_dir, output_loading_info=True)
            fused_model, fused_loading_info = DummyFusionModel.from_pretrained(
                source_dir, fusion_config=self.fusion_config, output_loading_info=True
            )

        self.assertIsInstance(plain_model.patch_embed.proj, nn.Conv3d)
        self.assertFalse(plain_loading_info["missing_keys"] or plain_loading_info["unexpected_keys"])
        self.assertFalse(fused_loading_info["missing_keys"] or fused_loading_info["unexpected_keys"])
        torch.testing.assert_close(
            fused_model.patch_embed.linear_proj.weight, model.patch_embed.proj.weight.flatten(1)
        )

    def test_from_pretrained_uses_serialized_fusion_config(self):
        # A serialized `fusion_config` is reused on a later load.
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as fused_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)

            fused_model = DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            fused_model.save_pretrained(fused_dir)

            reloaded_model = DummyFusionModel.from_pretrained(fused_dir)

        self.assertIsInstance(reloaded_model.patch_embed.linear_proj, nn.Linear)

    def test_from_pretrained_fuses_model_with_class_name_mapping(self):
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        register_checkpoint_conversion_mapping(DummyFusionModel.__name__, [], overwrite=True)
        model = DummyFusionModel(DummyFusionConfig())

        with tempfile.TemporaryDirectory() as source_dir:
            model.save_pretrained(source_dir)
            fused_model, loading_info = DummyFusionModel.from_pretrained(
                source_dir, fusion_config=self.fusion_config, output_loading_info=True
            )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        torch.testing.assert_close(
            fused_model.patch_embed.linear_proj.weight, model.patch_embed.proj.weight.flatten(1)
        )
