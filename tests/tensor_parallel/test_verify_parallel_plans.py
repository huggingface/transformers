# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import logging

from transformers.distributed.tensor_parallel import verify_tp_sp_ep_plan
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test, require_torch


@require_torch
@is_tensor_parallel_test
class VerifyParallelPlansTest(TestCasePlus):
    """Unit tests for verify_tp_sp_ep_plan (load-time plan vs state_dict key checks)."""

    def _expected_keys(self):
        return [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]

    def test_tp_warns_orphan_plan_key(self):
        """A TP plan entry that matches no parameter should log an unused-rule warning."""
        with self.assertLogs("transformers.distributed.tensor_parallel", level=logging.WARNING) as logs:
            verify_tp_sp_ep_plan(
                self._expected_keys(),
                tp_plan={
                    "model.layers.*.self_attn.q_proj": "colwise",
                    "model.layers.*.nonexistent_proj": "colwise",
                },
            )
        self.assertTrue(any("TP rules were not applied" in m for m in logs.output))

    def test_tp_warns_parameter_not_in_plan(self):
        """TP checks every parameter: one with no matching plan entry logs an unsharded warning."""
        with self.assertLogs("transformers.distributed.tensor_parallel", level=logging.WARNING) as logs:
            verify_tp_sp_ep_plan(
                self._expected_keys(),
                tp_plan={"model.layers.*.self_attn.q_proj": "colwise"},
            )
        self.assertTrue(any("not sharded by the TP plan" in m for m in logs.output))
        self.assertIn("model.layers.*.mlp.experts.gate_up_proj", logs.output[0])

    def test_sp_no_warn_comm_hooks_only(self):
        """SP entries that only install comm hooks (activation, allgather) are not validated as weight rules."""
        expected_keys = ["model.layers.0.self_attn.q_proj.weight"]
        with self.assertNoLogs("transformers.distributed.tensor_parallel", level=logging.WARNING):
            verify_tp_sp_ep_plan(
                expected_keys,
                sp_plan={
                    "model.layers.*.input_layernorm": "activation",
                    "model.layers.*.self_attn": "module_allgather_hidden_states",
                    "model.layers.*.self_attn.q_proj": "colwise",
                },
            )

    def test_sp_warns_orphan_plan_key(self):
        """An SP weight rule with no matching parameter logs unused-rule; SP does not require full-model coverage."""
        with self.assertLogs("transformers.distributed.tensor_parallel", level=logging.WARNING) as logs:
            verify_tp_sp_ep_plan(
                self._expected_keys(),
                sp_plan={
                    "model.layers.*.self_attn.q_proj": "colwise",
                    "model.layers.*.nonexistent_proj": "rowwise_reduce_scatter",
                },
            )
        self.assertTrue(any("SP rules were not applied" in m for m in logs.output))
        self.assertFalse(any("not sharded by the SP plan" in m for m in logs.output))

    def test_ep_no_warn_comm_hooks_only(self):
        """EP entries that only install comm hooks (ep_router, moe_experts_allreduce) are not weight rules."""
        expected_keys = ["model.layers.0.mlp.experts.gate_up_proj"]
        with self.assertNoLogs("transformers.distributed.tensor_parallel", level=logging.WARNING):
            verify_tp_sp_ep_plan(
                expected_keys,
                ep_plan={
                    "model.layers.*.mlp.gate": "ep_router",
                    "model.layers.*.mlp.experts": "moe_experts_allreduce",
                    "model.layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                },
            )

    def test_ep_warns_orphan_plan_key(self):
        """An EP weight rule with no matching parameter logs unused-rule; EP does not require full-model coverage."""
        with self.assertLogs("transformers.distributed.tensor_parallel", level=logging.WARNING) as logs:
            verify_tp_sp_ep_plan(
                self._expected_keys(),
                ep_plan={
                    "model.layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                    "model.layers.*.mlp.experts.down_proj": "grouped_gemm",
                },
            )
        self.assertTrue(any("EP rules were not applied" in m for m in logs.output))
        self.assertFalse(any("not sharded by the EP plan" in m for m in logs.output))

    def test_no_warn_when_plans_are_none(self):
        """Passing tp_plan=sp_plan=ep_plan=None is a no-op."""
        with self.assertNoLogs("transformers.distributed.tensor_parallel", level=logging.WARNING):
            verify_tp_sp_ep_plan([], tp_plan=None, sp_plan=None, ep_plan=None)
