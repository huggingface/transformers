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

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers.distributed.plan_utils import merge_dense_and_ep_plan
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


@require_torch
class ExplicitComboPlansTest(unittest.TestCase):
    def test_mixtral_tp_ep_matches_dense_ep_merge(self):
        config = MixtralConfig()
        expected = merge_dense_and_ep_plan(config.base_model_tp_plan, config.base_model_ep_plan)
        self.assertEqual(config.base_model_tp_ep_plan, expected)

    def test_mixtral_sp_ep_matches_dense_ep_merge(self):
        config = MixtralConfig()
        expected = merge_dense_and_ep_plan(config.base_model_sp_plan, config.base_model_ep_plan)
        self.assertEqual(config.base_model_sp_ep_plan, expected)

    def test_qwen3_moe_tp_ep_matches_dense_ep_merge(self):
        config = Qwen3MoeConfig()
        expected = merge_dense_and_ep_plan(config.base_model_tp_plan, config.base_model_ep_plan)
        self.assertEqual(config.base_model_tp_ep_plan, expected)

    def test_qwen3_moe_sp_ep_matches_dense_ep_merge(self):
        config = Qwen3MoeConfig()
        expected = merge_dense_and_ep_plan(config.base_model_sp_plan, config.base_model_ep_plan)
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
