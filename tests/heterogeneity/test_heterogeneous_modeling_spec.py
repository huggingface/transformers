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

from parameterized import parameterized

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        LayerIdxFromArgument,
        get_heterogeneous_modeling_spec,
    )


@require_torch
class TestHeterogeneousModelingSpec(unittest.TestCase):
    def test_get_heterogeneous_modeling_spec_returns_none_for_unregistered_model_type(self):
        class UnsupportedConfig:
            model_type = "fake"

        class UnsupportedModel:
            def __init__(self, config):
                self.config = config

        self.assertIsNone(get_heterogeneous_modeling_spec(UnsupportedModel(UnsupportedConfig())))

    def test_get_heterogeneous_modeling_spec_returns_none_when_model_sets_none(self):
        class RegisteredConfig:
            model_type = "llama"

        class OptedOutModel:
            _heterogeneous_modeling_spec = None

            def __init__(self, config):
                self.config = config

        self.assertIsNone(get_heterogeneous_modeling_spec(OptedOutModel(RegisteredConfig())))

    @parameterized.expand(
        [
            (
                "missing_attention_mask_consumers",
                {"cache_writers": {"self_attn": ["key_value"]}},
                "must also set `attention_mask_consumers`",
            ),
            ("missing_cache_writers", {"attention_mask_consumers": ["self_attn"]}, "must also set"),
            (
                "unknown_cache",
                {"attention_mask_consumers": ["self_attn"], "cache_writers": {"self_attn": ["kv"]}},
                r"unknown caches \['kv'\]",
            ),
        ]
    )
    def test_spec_with_skips_requires_valid_mask_consumers_and_cache_writers(self, _, declarations, message):
        with self.assertRaisesRegex(ValueError, message):
            HeterogeneousModelingSpec(
                layer_cls=torch.nn.Module,
                layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
                skip_descriptors={"attention": {"self_attn": torch.nn.Identity}},
                **declarations,
            )
