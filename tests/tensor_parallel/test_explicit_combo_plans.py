# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from types import SimpleNamespace

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers.distributed.plan_utils import merge_dense_and_ep_plan
    from transformers.distributed.tensor_parallel import resolve_parallel_plan
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class MockModelFromConfig:
    def __init__(self, config):
        self.config = config
        self._tp_plan = dict(config.base_model_tp_plan or {})
        self._sp_plan = dict(config.base_model_sp_plan or {})
        self._ep_plan = dict(config.base_model_ep_plan or {})
        self._tp_ep_plan = dict(config.base_model_tp_ep_plan or {})
        self._sp_ep_plan = dict(config.base_model_sp_ep_plan or {})


@require_torch
class ExplicitComboPlansTest(unittest.TestCase):
    def test_mixtral_tp_ep_matches_legacy_merge(self):
        config = MixtralConfig()
        model = MockModelFromConfig(config)
        expected = resolve_parallel_plan(model, None, enable_sp=False, enable_ep=True)
        self.assertEqual(config.base_model_tp_ep_plan, expected)

    def test_mixtral_sp_ep_matches_legacy_merge(self):
        config = MixtralConfig()
        model = MockModelFromConfig(config)
        expected = resolve_parallel_plan(model, None, enable_sp=True, enable_ep=True)
        self.assertEqual(config.base_model_sp_ep_plan, expected)

    def test_qwen3_moe_tp_ep_matches_legacy_merge(self):
        config = Qwen3MoeConfig()
        model = MockModelFromConfig(config)
        expected = resolve_parallel_plan(model, None, enable_sp=False, enable_ep=True)
        self.assertEqual(config.base_model_tp_ep_plan, expected)

    def test_qwen3_moe_sp_ep_matches_legacy_merge(self):
        config = Qwen3MoeConfig()
        model = MockModelFromConfig(config)
        expected = resolve_parallel_plan(model, None, enable_sp=True, enable_ep=True)
        self.assertEqual(config.base_model_sp_ep_plan, expected)

    def test_merge_dense_and_ep_plan_strips_moe_tp(self):
        dense = {"layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise"}
        ep = {"layers.*.mlp.experts.gate_up_proj": "grouped_gemm"}
        self.assertEqual(
            merge_dense_and_ep_plan(dense, ep),
            {"layers.*.mlp.experts.gate_up_proj": "grouped_gemm"},
        )


if __name__ == "__main__":
    unittest.main()
