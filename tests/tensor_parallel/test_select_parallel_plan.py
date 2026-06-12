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
    from transformers.distributed.plan_utils import merge_dense_and_ep_plan
    from transformers.distributed.tensor_parallel import apply_tensor_parallel, select_parallel_plan


TP_PLAN = {"layers.*.self_attn.q_proj": "colwise"}
SP_PLAN = {"embed_tokens": "vocab_reduce_scatter", "layers.*.self_attn.q_proj": "colwise"}
EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
}
TP_EP_PLAN = merge_dense_and_ep_plan(
    {
        **TP_PLAN,
        "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
        "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",
    },
    EP_PLAN,
)
SP_EP_PLAN = merge_dense_and_ep_plan(
    {
        **SP_PLAN,
        "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise",
        "layers.*.mlp.experts.down_proj": "moe_tp_down_rowwise",
    },
    EP_PLAN,
)

USER_TP_PLAN = {"layers.*.self_attn.o_proj": "rowwise_allreduce"}
USER_SP_PLAN = {"embed_tokens": "vocab_reduce_scatter", "layers.*.mlp.experts.gate_up_proj": "moe_tp_gate_up_colwise"}


class MockModel:
    base_model_prefix = "model"

    def __init__(self, *, user_tp_plan=None, enable_sp=False, enable_ep=False, explicit_combo=True):
        self._tp_plan = TP_PLAN
        self._sp_plan = SP_PLAN
        self._tp_ep_plan = TP_EP_PLAN if explicit_combo else {}
        self._sp_ep_plan = SP_EP_PLAN if explicit_combo else {}
        self.config = SimpleNamespace(
            model_type="mock",
            distributed_config=DistributedConfig(
                tp_size=2,
                tp_plan=user_tp_plan,
                enable_sequence_parallel=enable_sp,
                enable_expert_parallel=enable_ep,
            ),
            tie_word_embeddings=False,
        )

    def named_parameters(self):
        return []

    def named_modules(self):
        return []

    def register_forward_pre_hook(self, *args, **kwargs):
        return None


@require_torch
class SelectParallelPlanTest(unittest.TestCase):
    def test_user_override(self):
        user = {"layers.*.self_attn.o_proj": "rowwise_allreduce"}
        plan = select_parallel_plan(MockModel(user_tp_plan=user))
        self.assertEqual(plan, user)
        self.assertIsNot(plan, user)

    def test_explicit_tp_plan(self):
        self.assertEqual(select_parallel_plan(MockModel(enable_sp=False, enable_ep=False)), TP_PLAN)

    def test_explicit_sp_plan(self):
        self.assertEqual(select_parallel_plan(MockModel(enable_sp=True, enable_ep=False)), SP_PLAN)

    def test_explicit_tp_ep_plan(self):
        self.assertEqual(select_parallel_plan(MockModel(enable_sp=False, enable_ep=True)), TP_EP_PLAN)

    def test_explicit_sp_ep_plan(self):
        self.assertEqual(select_parallel_plan(MockModel(enable_sp=True, enable_ep=True)), SP_EP_PLAN)

    def test_missing_combo_plan_raises(self):
        model = MockModel(enable_sp=True, enable_ep=True, explicit_combo=False)
        with self.assertRaises(ValueError):
            select_parallel_plan(model)


@require_torch
class ApplyTensorParallelPlanSelectionTest(unittest.TestCase):
    """End-to-end through apply_tensor_parallel: SP guard + selected plan."""

    def _selected_plan(self, model):
        captured = {}

        def spy(model_arg):
            captured["plan"] = select_parallel_plan(model_arg)
            return captured["plan"]

        with patch("transformers.distributed.tensor_parallel.select_parallel_plan", spy):
            apply_tensor_parallel(model, tp_mesh=None)
        return captured["plan"]

    def test_auto_sp_false_ep_false_uses_tp_plan(self):
        self.assertEqual(self._selected_plan(MockModel(enable_sp=False, enable_ep=False)), TP_PLAN)

    def test_auto_sp_true_ep_true_uses_sp_ep_plan(self):
        self.assertEqual(self._selected_plan(MockModel(enable_sp=True, enable_ep=True)), SP_EP_PLAN)

    def test_override_sp_true_non_sp_plan_raises(self):
        model = MockModel(USER_TP_PLAN, enable_sp=True, enable_ep=False)
        with self.assertRaises(ValueError):
            apply_tensor_parallel(model, tp_mesh=None)

    def test_override_plan_is_copied(self):
        user_plan = dict(USER_TP_PLAN)
        model = MockModel(user_tp_plan=user_plan, enable_sp=False, enable_ep=False)
        plan = select_parallel_plan(model)
        self.assertIsNot(plan, user_plan)
        self.assertEqual(user_plan, USER_TP_PLAN)


if __name__ == "__main__":
    unittest.main()
