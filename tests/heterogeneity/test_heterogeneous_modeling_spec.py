# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib
import unittest
from unittest.mock import patch

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        SkipDescriptor,
        get_heterogeneous_modeling_spec,
        nest_skip_descriptor_paths,
    )


@require_torch
class TestHeterogeneousModelingSpec(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_nest_skip_descriptor_paths_returns_nested_copies(self):
        skip_descriptors = {
            "mixer": SkipDescriptor(
                replacements={
                    "norm": torch.nn.Identity,
                    ("mixer", torch.nn.Linear): torch.nn.Identity,
                },
                replaces_kv_cache_updater=True,
            )
        }

        nested_descriptors = nest_skip_descriptor_paths(skip_descriptors, parent_path="wrapper.block")

        self.assertEqual(
            set(nested_descriptors["mixer"].replacements),
            {"wrapper.block.norm", ("wrapper.block.mixer", torch.nn.Linear)},
        )
        self.assertTrue(nested_descriptors["mixer"].replaces_kv_cache_updater)
        self.assertEqual(set(skip_descriptors["mixer"].replacements), {"norm", ("mixer", torch.nn.Linear)})
        self.assertIsNone(nest_skip_descriptor_paths(None, parent_path="wrapper.block"))

    def test_get_heterogeneous_modeling_spec_uses_custom_model_spec(self):
        spec = HeterogeneousModelingSpec(layer_cls=torch.nn.Linear, layer_idx_variable_name="layer_idx")

        class CustomModel:
            pass

        CustomModel.__module__ = "remote_code.modeling_custom"
        CustomModel._heterogeneous_modeling_spec = spec

        self.assertIs(get_heterogeneous_modeling_spec(CustomModel()), spec)

    def test_get_heterogeneous_modeling_spec_uses_supported_model_registry(self):
        spec = HeterogeneousModelingSpec(layer_cls=torch.nn.Linear, layer_idx_variable_name="layer_idx")

        class BuiltInModel:
            pass

        BuiltInModel.__module__ = "transformers.models.test_model.modeling_test_model"

        supported_models = importlib.import_module("transformers.integrations.heterogeneity.supported_models")
        with patch.dict(supported_models.MODEL_TO_SPEC_FACTORY, {"test_model": lambda: spec}):
            self.assertIs(get_heterogeneous_modeling_spec(BuiltInModel()), spec)

    def test_get_heterogeneous_modeling_spec_raises_for_unsupported_builtin_model(self):
        class UnsupportedModel:
            pass

        UnsupportedModel.__module__ = "transformers.models.fake.modeling_fake"

        with self.assertRaisesRegex(ValueError, "No heterogeneous modeling spec is defined for `fake`"):
            get_heterogeneous_modeling_spec(UnsupportedModel())
