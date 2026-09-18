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
        get_heterogeneous_modeling_spec,
        nest_skip_descriptor_paths,
    )


@require_torch
class TestHeterogeneousModelingSpec(unittest.TestCase):
    def test_nest_skip_descriptor_paths_returns_nested_copies(self):
        skip_descriptors = {
            "mixer": {
                "norm": torch.nn.Identity,
                ("mixer", torch.nn.Linear): torch.nn.Identity,
            }
        }

        nested_descriptors = nest_skip_descriptor_paths(skip_descriptors, parent_path="wrapper.block")

        self.assertEqual(
            set(nested_descriptors["mixer"]),
            {"wrapper.block.norm", ("wrapper.block.mixer", torch.nn.Linear)},
        )
        nested_targets = nested_descriptors["mixer"]
        self.assertIs(nested_targets["wrapper.block.norm"], torch.nn.Identity)
        self.assertIs(nested_targets[("wrapper.block.mixer", torch.nn.Linear)], torch.nn.Identity)
        self.assertEqual(set(skip_descriptors["mixer"]), {"norm", ("mixer", torch.nn.Linear)})
        self.assertIsNone(nest_skip_descriptor_paths(None, parent_path="wrapper.block"))

    def test_get_heterogeneous_modeling_spec_returns_none_for_unregistered_model_type(self):
        class UnsupportedConfig:
            model_type = "fake"

        class UnsupportedModel:
            def __init__(self, config):
                self.config = config

        self.assertIsNone(get_heterogeneous_modeling_spec(UnsupportedModel(UnsupportedConfig())))
