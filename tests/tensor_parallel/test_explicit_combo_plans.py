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
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


@require_torch
class ExplicitComboPlansTest(unittest.TestCase):
    def test_mixtral_combo_plans_are_populated(self):
        config = MixtralConfig()
        self.assertTrue(config.base_model_tp_ep_plan)
        self.assertTrue(config.base_model_sp_ep_plan)
        self.assertEqual(config.base_model_tp_ep_plan["layers.*.mlp.gate"], "ep_router")
        self.assertEqual(config.base_model_tp_ep_plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")
        self.assertEqual(config.base_model_sp_ep_plan["layers.*.mlp.gate"], "ep_router")
        self.assertEqual(config.base_model_sp_ep_plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")

    def test_qwen3_moe_combo_plans_are_populated(self):
        config = Qwen3MoeConfig()
        self.assertTrue(config.base_model_tp_ep_plan)
        self.assertTrue(config.base_model_sp_ep_plan)
        self.assertEqual(config.base_model_tp_ep_plan["layers.*.mlp.gate"], "ep_router")
        self.assertEqual(config.base_model_tp_ep_plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")
        self.assertEqual(config.base_model_sp_ep_plan["layers.*.mlp.gate"], "ep_router")
        self.assertEqual(config.base_model_sp_ep_plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")


if __name__ == "__main__":
    unittest.main()
