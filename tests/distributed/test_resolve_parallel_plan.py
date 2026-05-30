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
from unittest.mock import patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan


TP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
    "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",
    "layers.*.mlp.experts": "moe_experts_allreduce",
}
SP_PLAN = {
    "embed_tokens": "vocab_reduce_scatter",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
    "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",
    "layers.*.mlp.experts": "moe_experts_allreduce",
}
EP_PLAN = {
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_experts_allreduce",
}

# Explicit DistributedConfig.tp_plan overrides. Distinct from the model defaults so the
# assertions unambiguously show the override won. USER_TP_PLAN has no sequence-parallel style
# (valid only when SP is off); USER_SP_PLAN shards the sequence (valid under SP).
USER_TP_PLAN = {
    "layers.*.self_attn.o_proj": "rowwise_allreduce",  # dense — survives the EP overlay
    "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",  # expert TP — replaced under EP
    "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",  # expert TP — replaced under EP
}
USER_SP_PLAN = {
    "embed_tokens": "vocab_reduce_scatter",  # SP style — passes the SP guard
    "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
    "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",
}


class MockTPModel:
    """Model surface for apply_tensor_parallel: declared plans, a DistributedConfig, empty
    params/modules (nothing is actually sharded), and a no-op SP hook registration.
    """

    base_model_prefix = "model"

    def __init__(self, user_tp_plan=None, enable_sp=False, enable_ep=False):
        self._tp_plan = TP_PLAN
        self._sp_plan = SP_PLAN
        self._ep_plan = EP_PLAN
        self.config = SimpleNamespace(
            distributed_config=DistributedConfig(
                tp_size=2,
                tp_plan=user_tp_plan,
                enable_sequence_parallel=enable_sp,
                enable_expert_parallel=enable_ep,
            ),
            base_model_sp_plan=SP_PLAN,  # non-None so enable_sp tracks the flag
            base_model_ep_plan=EP_PLAN,  # non-None so enable_ep tracks the flag
            tie_word_embeddings=False,
        )

    def named_parameters(self):
        return []

    def named_modules(self):
        return []

    def register_forward_pre_hook(self, *args, **kwargs):
        # apply_tensor_parallel registers an SP position_ids hook when enable_sp; no-op here.
        return None

@require_torch
class ApplyTensorParallelTest(unittest.TestCase):
    """End-to-end through apply_tensor_parallel over (user_tp_plan, enable_sp, enable_ep): the SP
    guard runs, then we assert the plan it resolved (or that it rejected an invalid override)."""

    def _resolved_plan(self, model):
        captured = {}

        def spy(*args, **kwargs):
            captured["plan"] = resolve_parallel_plan(*args, **kwargs)
            return captured["plan"]

        with patch("transformers.distributed.tensor_parallel.resolve_parallel_plan", spy):
            apply_tensor_parallel(model, tp_mesh=None)
        return captured["plan"]

    def test_auto_sp_false_ep_false_uses_tp_plan(self):
        # Dense TP: plain _tp_plan — experts keep moe_tp_*.
        self.assertEqual(self._resolved_plan(MockTPModel(enable_sp=False, enable_ep=False)), TP_PLAN)

    def test_auto_sp_true_ep_false_uses_sp_plan(self):
        # SP training: plain _sp_plan.
        self.assertEqual(self._resolved_plan(MockTPModel(enable_sp=True, enable_ep=False)), SP_PLAN)

    def test_auto_sp_false_ep_true_tp_union_ep(self):
        # Inference TP+EP: _tp_plan overlaid with _ep_plan — experts switch to grouped_gemm,
        # the EP router is added.
        expected = {
            "layers.*.self_attn.q_proj": "colwise",  # tp
            "layers.*.self_attn.v_proj": "colwise",  # ep
            "layers.*.mlp.gate": "ep_router",  # ep
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",  # ep
            "layers.*.mlp.experts.down_proj": "grouped_gemm",  # ep
            "layers.*.mlp.experts": "moe_experts_allreduce",  # ep & tp
        }
        self.assertEqual(self._resolved_plan(MockTPModel(enable_sp=False, enable_ep=True)), expected)

    def test_auto_sp_true_ep_true_sp_union_ep(self):
        # Training SP+EP: _sp_plan overlaid with _ep_plan.
        expected = {
            "embed_tokens": "vocab_reduce_scatter",  # sp
            "layers.*.self_attn.k_proj": "colwise",  # sp
            "layers.*.self_attn.v_proj": "colwise",  # ep
            "layers.*.mlp.gate": "ep_router",  # ep
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",  # ep
            "layers.*.mlp.experts.down_proj": "grouped_gemm",  # ep
            "layers.*.mlp.experts": "moe_experts_allreduce",  # ep & sp
        }
        self.assertEqual(self._resolved_plan(MockTPModel(enable_sp=True, enable_ep=True)), expected)

    def test_override_sp_false_ep_false_uses_user_plan(self):
        # Explicit plan replaces _tp_plan verbatim. SP off, so a non-SP plan is accepted
        # (regression: this used to raise before the guard was gated on enable_sp).
        self.assertEqual(self._resolved_plan(MockTPModel(USER_TP_PLAN, enable_sp=False, enable_ep=False)), USER_TP_PLAN)

    def test_override_sp_false_ep_true_user_union_ep(self):
        # Inference TP+EP with an explicit dense plan: user plan ∪ _ep_plan.
        expected = {
            "layers.*.self_attn.o_proj": "rowwise_allreduce",  # user
            "layers.*.self_attn.v_proj": "colwise",  # ep
            "layers.*.mlp.gate": "ep_router",  # ep
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",  # ep (replaces user moe_tp_*)
            "layers.*.mlp.experts.down_proj": "grouped_gemm",  # ep (replaces user moe_tp_*)
            "layers.*.mlp.experts": "moe_experts_allreduce",  # ep
        }
        self.assertEqual(self._resolved_plan(MockTPModel(USER_TP_PLAN, enable_sp=False, enable_ep=True)), expected)

    def test_override_sp_true_ep_false_uses_user_plan(self):
        # SP on: the override must itself shard the sequence; it replaces _sp_plan (not merged).
        self.assertEqual(self._resolved_plan(MockTPModel(USER_SP_PLAN, enable_sp=True, enable_ep=False)), USER_SP_PLAN)

    def test_override_sp_true_ep_true_user_union_ep(self):
        # Training SP+EP with an explicit dense plan: user plan ∪ _ep_plan.
        expected = {
            "embed_tokens": "vocab_reduce_scatter",  # user
            "layers.*.self_attn.v_proj": "colwise",  # ep
            "layers.*.mlp.gate": "ep_router",  # ep
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",  # ep (replaces user moe_tp_*)
            "layers.*.mlp.experts.down_proj": "grouped_gemm",  # ep (replaces user moe_tp_*)
            "layers.*.mlp.experts": "moe_experts_allreduce",  # ep
        }
        self.assertEqual(self._resolved_plan(MockTPModel(USER_SP_PLAN, enable_sp=True, enable_ep=True)), expected)

    def test_override_sp_true_non_sp_plan_raises(self):
        # enable_sp=True + explicit plan with no sequence-parallel style → rejected (ep irrelevant:
        # the guard runs before resolution).
        model = MockTPModel(USER_TP_PLAN, enable_sp=True, enable_ep=False)
        with self.assertRaises(ValueError):
            apply_tensor_parallel(model, tp_mesh=None)

    def test_override_plan_is_copied(self):
        # The resolved plan must be a fresh dict, never the caller's DistributedConfig.tp_plan.
        # EP off, so the dict() copy is the only thing protecting the caller's dict.
        user_plan = dict(USER_TP_PLAN)
        plan = self._resolved_plan(MockTPModel(user_plan, enable_sp=False, enable_ep=False))
        self.assertIsNot(plan, user_plan)
        self.assertEqual(user_plan, USER_TP_PLAN)


if __name__ == "__main__":
    unittest.main()
