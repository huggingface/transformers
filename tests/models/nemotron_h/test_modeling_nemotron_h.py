# Copyright 2024-2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the backward-compat ``nemotron_h`` dispatcher.

The original ``nemotron_h`` model has been split into ``nemotron_h_dense`` and
``nemotron_h_sparse``. The ``NemotronHConfig`` / ``NemotronHModel`` /
``NemotronHForCausalLM`` symbols live on as thin dispatchers that route to the
right subclass based on the ``hybrid_override_pattern``.
"""

import unittest

from transformers import (
    NemotronHConfig,
    NemotronHDenseConfig,
    NemotronHForCausalLM,
    NemotronHModel,
    NemotronHSparseConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers import (
        NemotronHDenseForCausalLM,
        NemotronHDenseModel,
        NemotronHSparseForCausalLM,
        NemotronHSparseModel,
    )


class NemotronHBCDispatcherTest(unittest.TestCase):
    def test_config_dispatch_dense_pattern(self):
        config = NemotronHConfig(hybrid_override_pattern="M-M-*-")
        self.assertIsInstance(config, NemotronHDenseConfig)
        self.assertEqual(config.layer_types, ["mamba", "mamba", "attention"])

    def test_config_dispatch_sparse_pattern(self):
        config = NemotronHConfig(hybrid_override_pattern="ME*E")
        self.assertIsInstance(config, NemotronHSparseConfig)
        self.assertEqual(config.layer_types, ["mamba", "attention"])

    def test_config_dispatch_on_mtp_kwargs(self):
        # Legacy Nemotron-3 configs signal sparse via `mtp_hybrid_override_pattern`.
        config = NemotronHConfig(hybrid_override_pattern="M*", mtp_hybrid_override_pattern="*E")
        self.assertIsInstance(config, NemotronHSparseConfig)


@require_torch
class NemotronHBCModelDispatcherTest(unittest.TestCase):
    def _dense_config(self):
        return NemotronHConfig(
            vocab_size=100,
            hidden_size=32,
            hybrid_override_pattern="M-M-*-",
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=32,
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=1,
            ssm_state_size=4,
            chunk_size=4,
            use_mamba_kernels=False,
        )

    def _sparse_config(self):
        return NemotronHConfig(
            vocab_size=100,
            hidden_size=32,
            hybrid_override_pattern="ME*E",
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=1,
            ssm_state_size=4,
            chunk_size=4,
            use_mamba_kernels=False,
            n_routed_experts=4,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=32,
            num_experts_per_tok=2,
            n_shared_experts=1,
        )

    def test_model_dispatch_dense(self):
        model = NemotronHModel(self._dense_config())
        self.assertIsInstance(model, NemotronHDenseModel)

    def test_model_dispatch_sparse(self):
        model = NemotronHModel(self._sparse_config())
        self.assertIsInstance(model, NemotronHSparseModel)

    def test_for_causal_lm_dispatch_dense(self):
        model = NemotronHForCausalLM(self._dense_config())
        self.assertIsInstance(model, NemotronHDenseForCausalLM)

    def test_for_causal_lm_dispatch_sparse(self):
        model = NemotronHForCausalLM(self._sparse_config())
        self.assertIsInstance(model, NemotronHSparseForCausalLM)
