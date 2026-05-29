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

from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.tensor_parallel import resolve_parallel_plan
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.testing_utils import require_torch


class _PlanModel:
    def __init__(self):
        self.config = Qwen3MoeConfig()
        self._tp_plan = dict(self.config.base_model_tp_plan)
        self._sp_plan = dict(self.config.base_model_sp_plan)
        self._ep_plan = dict(self.config.base_model_ep_plan)


@require_torch
class ResolveParallelPlanTest(unittest.TestCase):
    def setUp(self):
        self.model = _PlanModel()

    def test_resolve_plan_training_sp_ep(self):
        dist_config = DistributedConfig(
            tp_size=2,
            enable_sequence_parallel=True,
            enable_expert_parallel=True,
        )
        plan = resolve_parallel_plan(self.model, dist_config)

        self.assertEqual(plan["layers.*.mlp.gate"], "ep_router")
        self.assertEqual(plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")
        self.assertEqual(plan["layers.*.mlp.experts.down_proj"], "grouped_gemm")
        self.assertNotIn("moe_tp_gate_up_colwise", plan.values())
        self.assertNotIn("moe_tp_down_rowwise", plan.values())
        self.assertEqual(plan["layers.*.self_attn.o_proj"], "rowwise_reduce_scatter")
        self.assertEqual(plan["embed_tokens"], "vocab_reduce_scatter")

    def test_resolve_plan_inference_tp_ep(self):
        dist_config = DistributedConfig(
            tp_size=2,
            enable_sequence_parallel=False,
            enable_expert_parallel=True,
        )
        plan = resolve_parallel_plan(self.model, dist_config)

        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")
        self.assertEqual(plan["layers.*.self_attn.o_proj"], "rowwise_allreduce")
        self.assertEqual(plan["layers.*.mlp.experts.gate_up_proj"], "grouped_gemm")
        self.assertEqual(plan["layers.*.mlp.gate"], "ep_router")
        self.assertNotIn("moe_tp_gate_up_colwise", plan.values())

    def test_resolve_plan_tp_only(self):
        dist_config = DistributedConfig(tp_size=2)
        plan = resolve_parallel_plan(self.model, dist_config)

        self.assertEqual(
            plan["layers.*.mlp.experts.gate_up_proj"],
            self.model._tp_plan["layers.*.mlp.experts.gate_up_proj"],
        )
        self.assertEqual(plan["layers.*.mlp.experts.gate_up_proj"], "moe_tp_gate_up_colwise")

    def test_resolve_plan_sp_only(self):
        dist_config = DistributedConfig(tp_size=2, enable_sequence_parallel=True)
        plan = resolve_parallel_plan(self.model, dist_config)

        self.assertEqual(plan["embed_tokens"], "vocab_reduce_scatter")
        self.assertNotIn("layers.*.mlp.gate", plan)
        self.assertEqual(plan["layers.*.mlp.experts"], "moe_experts_allreduce")

    def test_resolve_plan_sp_only_no_intra_expert_tp_in_source(self):
        """After base_model_sp_plan hygiene, expert param TP keys live only in _tp_plan / EP merge."""
        self.model._sp_plan = {
            k: v
            for k, v in self.model._sp_plan.items()
            if k not in ("layers.*.mlp.experts.gate_up_proj", "layers.*.mlp.experts.down_proj")
        }
        dist_config = DistributedConfig(tp_size=2, enable_sequence_parallel=True)
        plan = resolve_parallel_plan(self.model, dist_config)
        self.assertNotIn("layers.*.mlp.experts.gate_up_proj", plan)

    def test_resolve_plan_explicit_override(self):
        override = {"layers.*.self_attn.q_proj": "rowwise_allreduce"}
        dist_config = DistributedConfig(tp_size=2, tp_plan=override)
        plan = resolve_parallel_plan(self.model, dist_config)
        self.assertEqual(plan, override)

    def test_resolve_plan_none_config(self):
        plan = resolve_parallel_plan(self.model, None)
        self.assertEqual(plan, self.model._tp_plan)


if __name__ == "__main__":
    unittest.main()
