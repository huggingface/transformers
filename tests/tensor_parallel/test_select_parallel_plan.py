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
    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.tensor_parallel import resolve_parallel_plan, select_parallel_plan


TP_PLAN = {"layers.*.self_attn.q_proj": "colwise"}
SP_PLAN = {"embed_tokens": "vocab_reduce_scatter", "layers.*.self_attn.q_proj": "colwise"}
TP_EP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
}
SP_EP_PLAN = {
    "embed_tokens": "vocab_reduce_scatter",
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
}


class MockModel:
    def __init__(self, *, user_tp_plan=None, enable_sp=False, enable_ep=False, explicit_combo=True):
        self._tp_plan = TP_PLAN
        self._sp_plan = SP_PLAN
        self._tp_ep_plan = TP_EP_PLAN if explicit_combo else {}
        self._sp_ep_plan = SP_EP_PLAN if explicit_combo else {}
        self._ep_plan = {"layers.*.mlp.gate": "ep_router"}
        self.config = SimpleNamespace(
            distributed_config=DistributedConfig(
                tp_size=2,
                tp_plan=user_tp_plan,
                enable_sequence_parallel=enable_sp,
                enable_expert_parallel=enable_ep,
            ),
            base_model_sp_plan=SP_PLAN,
            base_model_ep_plan={"layers.*.mlp.gate": "ep_router"},
            tie_word_embeddings=False,
        )


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

    def test_fallback_to_resolve_when_combo_missing(self):
        model = MockModel(enable_sp=True, enable_ep=True, explicit_combo=False)
        expected = resolve_parallel_plan(model, None, enable_sp=True, enable_ep=True)
        self.assertEqual(select_parallel_plan(model), expected)


if __name__ == "__main__":
    unittest.main()
