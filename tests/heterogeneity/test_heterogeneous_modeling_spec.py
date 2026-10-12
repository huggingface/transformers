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

import unittest

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        LayerIdxFromArgument,
        get_heterogeneous_modeling_spec,
        nest_spec_member_paths,
    )


@require_torch
class TestHeterogeneousModelingSpec(unittest.TestCase):
    def test_nest_spec_member_paths_returns_nested_copy(self):
        spec = HeterogeneousModelingSpec(
            layer_cls=torch.nn.Module,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={"mixer": {"norm": torch.nn.Identity, ("mixer", torch.nn.Linear): torch.nn.Identity}},
            attention_mask_consumers=[("mixer", torch.nn.Linear)],
            cache_writers={("mixer", torch.nn.Linear): ["key_value"]},
        )

        nested_spec = nest_spec_member_paths(spec, parent_path="wrapper.block")

        nested_mixer = ("wrapper.block.mixer", torch.nn.Linear)
        self.assertEqual(
            nested_spec.skip_descriptors,
            {"mixer": {"wrapper.block.norm": torch.nn.Identity, nested_mixer: torch.nn.Identity}},
        )
        self.assertEqual(nested_spec.attention_mask_consumers, {nested_mixer})
        self.assertEqual(nested_spec.cache_writers, {nested_mixer: {"key_value"}})
        self.assertEqual(set(spec.skip_descriptors["mixer"]), {"norm", ("mixer", torch.nn.Linear)})

    def test_get_heterogeneous_modeling_spec_returns_none_for_unregistered_model_type(self):
        class UnsupportedConfig:
            model_type = "fake"

        class UnsupportedModel:
            def __init__(self, config):
                self.config = config

        self.assertIsNone(get_heterogeneous_modeling_spec(UnsupportedModel(UnsupportedConfig())))
