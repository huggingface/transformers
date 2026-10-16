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
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh

from transformers import AutoModelForCausalLM
from transformers.distributed import expert_parallel
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.expert_parallel import (
    ALL_EXPERT_PARALLEL_STYLES,
    EpDispatchExpertsParallel,
    _ep_plan_styles,
)
from transformers.distributed.mixin import DistributedMixin
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test

from ..test_tensor_parallel_mixin import _init_distributed


EP_STYLES = {"grouped_gemm", "ep_router", "ep_dispatch_experts", "megamoe_router", "megamoe_experts"}


class _MockDeviceMesh:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.shape = (world_size,)
        self.ndim = 1

    def size(self):
        return self.world_size

    def get_local_rank(self):
        return self.rank


class _MoeModel(DistributedMixin, torch.nn.Module):
    """Dense gate plus a routed experts module, named like the model plans (`layers.*.mlp...`)."""

    def __init__(self, num_layers=2, num_experts=8, hidden=16):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = torch.nn.Linear(hidden, num_experts)
            layer.mlp.experts = torch.nn.Module()
            layer.mlp.experts.num_experts = num_experts
            layer.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.empty(num_experts, hidden, 2 * hidden))
            layer.mlp.experts.down_proj = torch.nn.Parameter(torch.empty(num_experts, hidden, hidden))
            self.layers.append(layer)
        self._tp_plan = {
            "layers.*.mlp.gate": "colwise_gather_output",
            "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "layers.*.mlp.experts.down_proj": "rowwise",
            "layers.*.mlp.experts": "moe_tp_experts",
        }
        self._ep_plan = {
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts": "ep_dispatch_experts",
        }


@is_tensor_parallel_test
class TestExpertParallelRegistry(TestCasePlus):
    def test_registries_are_split(self):
        self.assertEqual(set(ALL_EXPERT_PARALLEL_STYLES.keys()), EP_STYLES)
        for style in EP_STYLES:
            self.assertNotIn(style, ALL_PARALLEL_STYLES)
        # Tensor-parallel MoE styles stay with the tensor-parallel registry: they shard experts in TP-only runs.
        self.assertIn("moe_tp_experts", ALL_PARALLEL_STYLES)
        self.assertIn("moe_identity_expert", ALL_PARALLEL_STYLES)

    def test_ep_plan_validation_accepts_tensor_parallel_styles(self):
        """An `ep_plan` names the experts, but may also carry TP rules, e.g. `colwise` on a shared gate."""
        model = _MoeModel()
        ep_plan = {**model.ep_plan, "layers.*.mlp.gate": "colwise", "layers.*.mlp.experts": "moe_tp_experts"}
        model._validate_parallel_plan(ep_plan, _ep_plan_styles(), "ep_plan")

        with self.assertRaises(ValueError) as context:
            model._validate_parallel_plan(
                {"a": "invalid_style", "b": "another_invalid_style"}, _ep_plan_styles(), "ep_plan"
            )
        self.assertIn("'invalid_style'", str(context.exception))
        self.assertIn("'another_invalid_style'", str(context.exception))
        self.assertIn("Supported styles are", str(context.exception))

    def test_tp_plan_validation_rejects_expert_parallel_styles(self):
        model = _MoeModel()
        with self.assertRaises(ValueError) as context:
            model._validate_parallel_plan(
                {"layers.*.mlp.experts.down_proj": "grouped_gemm"}, ALL_PARALLEL_STYLES, "tp_plan"
            )
        self.assertIn("'grouped_gemm'", str(context.exception))

    def test_moved_names_still_importable_from_tensor_parallel(self):
        """The EP styles shipped from `tensor_parallel` in v5.20: the old path keeps working with a FutureWarning."""
        from transformers.distributed import tensor_parallel

        with self.assertWarns(FutureWarning):
            self.assertIs(tensor_parallel.EpDispatchExpertsParallel, EpDispatchExpertsParallel)
        with self.assertWarns(FutureWarning):
            self.assertIs(tensor_parallel.apply_expert_parallelism, expert_parallel.apply_expert_parallelism)
        with self.assertRaises(AttributeError):
            tensor_parallel.does_not_exist

    def test_plan_setters(self):
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        ep_plan = {"model.layers.*.mlp": "ep_dispatch_experts", "model.layers.*.mlp.gate_proj": "colwise"}
        model.ep_plan = ep_plan
        self.assertEqual(model.ep_plan, ep_plan)

        with self.assertRaises(ValueError) as context:
            model.tp_plan = {"model.layers.*.mlp.down_proj": "grouped_gemm"}
        self.assertIn("'grouped_gemm'", str(context.exception))

    def test_expand_parallel_plan_resolves_ep_and_tp_styles(self):
        model = _MoeModel(num_layers=1)
        ep_plan = {**model.ep_plan, "layers.*.mlp.gate": "colwise"}

        expanded = model._expand_parallel_plan(ep_plan)

        self.assertEqual(
            expanded,
            {
                "layers.0.mlp.gate.weight": "colwise",
                "layers.0.mlp.gate.bias": "colwise",
                "layers.0.mlp.gate": "colwise",
                "layers.0.mlp.experts.gate_up_proj": "grouped_gemm",
                "layers.0.mlp.experts.down_proj": "grouped_gemm",
                "layers.0.mlp.experts": "ep_dispatch_experts",
            },
        )

    def test_resolve_parallel_plans_gives_experts_to_ep(self):
        model = _MoeModel(num_layers=1)
        tp_plan, ep_plan, fsdp_plan = model._resolve_parallel_plans(DistributedConfig(tp_size=2, ep_size=2))

        self.assertEqual(ep_plan, model._expand_parallel_plan(model.ep_plan))
        # EP owns the experts and their parameters; the dense gate keeps its TP rule.
        self.assertEqual(set(tp_plan), {"layers.0.mlp.gate.weight", "layers.0.mlp.gate.bias", "layers.0.mlp.gate"})
        self.assertEqual(fsdp_plan, {})

    def test_resolve_parallel_plans_without_ep_keeps_moe_tp_rules(self):
        model = _MoeModel(num_layers=1)
        tp_plan, ep_plan, _ = model._resolve_parallel_plans(DistributedConfig(tp_size=2))

        self.assertEqual(ep_plan, {})
        self.assertEqual(tp_plan["layers.0.mlp.experts"], "moe_tp_experts")
        self.assertEqual(tp_plan["layers.0.mlp.experts.gate_up_proj"], "packed_colwise")


@is_tensor_parallel_test
class TestExpertParallelLayer(TestCasePlus):
    def _get_parameter_placements(self, module, style, mesh):
        placements = {}
        with patch.object(
            expert_parallel, "distribute_tensor", side_effect=lambda tensor, *args, **kwargs: tensor
        ) as distribute:
            for parameter_name in list(module._parameters):
                style.shard_param(module, parameter_name, mesh)
                placements[parameter_name] = distribute.call_args.args[2][0]

        return placements

    def test_grouped_gemm_updates_local_expert_count(self):
        module = torch.nn.Module()
        module.num_experts = 8
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))
        grouped_gemm = ALL_EXPERT_PARALLEL_STYLES["grouped_gemm"]

        placements = self._get_parameter_placements(module, grouped_gemm, _MockDeviceMesh(world_size=4, rank=0))

        self.assertEqual(placements["weight"].dim, 0)
        self.assertEqual(module.num_experts, 2)
        self.assertTrue(module._is_expert_parallel)

    def test_grouped_gemm_rejects_indivisible_expert_count(self):
        module = torch.nn.Module()
        module.num_experts = 6
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(6, 16, 32)))

        with self.assertRaises(ValueError) as context:
            self._get_parameter_placements(
                module, ALL_EXPERT_PARALLEL_STYLES["grouped_gemm"], _MockDeviceMesh(world_size=4, rank=0)
            )
        self.assertIn("evenly shard", str(context.exception))

    def test_sharding_does_not_create_unrelated_module_attributes(self):
        module = torch.nn.Module()
        module.random_attr = 123
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))

        self._get_parameter_placements(
            module, ALL_EXPERT_PARALLEL_STYLES["grouped_gemm"], _MockDeviceMesh(world_size=4, rank=0)
        )

        self.assertEqual(module.random_attr, 123)
        self.assertFalse(hasattr(module, "num_experts"))

    def test_apply_expert_parallelism_picks_the_mesh_from_the_experts_rule(self):
        """Token dispatch shards the experts on the EP mesh; masked EP keeps every token and uses the TP mesh."""
        ep_mesh, tp_mesh = _MockDeviceMesh(world_size=4, rank=0), _MockDeviceMesh(world_size=2, rank=0)
        dispatch_plan = _MoeModel().ep_plan
        masked_plan = {**dispatch_plan, "layers.*.mlp.experts": "moe_tp_experts", "layers.*.mlp.gate": "ep_router"}
        for ep_plan, expected_mesh in ((dispatch_plan, ep_mesh), (masked_plan, tp_mesh)):
            with self.subTest(experts_rule=ep_plan["layers.*.mlp.experts"]):
                model = _MoeModel()
                plan = model._expand_parallel_plan(ep_plan)
                with (
                    patch.object(expert_parallel, "distribute_tensor", side_effect=lambda t, *a, **k: t) as distribute,
                    patch.object(ALL_EXPERT_PARALLEL_STYLES["ep_dispatch_experts"], "install_forward") as dispatch,
                ):
                    expert_parallel.apply_expert_parallelism(model, ep_mesh, tp_mesh, plan)

                self.assertEqual({call.args[1] for call in distribute.call_args_list}, {expected_mesh})
                self.assertEqual(model.layers[0].mlp.experts.num_experts, 8 // expected_mesh.size())
                self.assertTrue(model.layers[0].mlp.experts._is_hooked)
                if expected_mesh is ep_mesh:
                    dispatch.assert_called_with(model.layers[1].mlp.experts, ep_mesh=ep_mesh, tp_mesh=tp_mesh)
                else:
                    dispatch.assert_not_called()

    def test_dispatch_hands_experts_the_received_rows(self):
        """The all-to-all's pending result (`AsyncCollectiveTensor`) holds a null pointer until a torch op waits on
        it, and experts that launch kernels on raw pointers would read it: the dispatch waits before any experts run."""
        _init_distributed(tp=2, backend="gloo")(_dispatch_hands_experts_the_received_rows)()


class _DummyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2

    def forward(self, tokens, expert_ids, weights):
        assert type(tokens) is torch.Tensor, type(tokens)
        return tokens


# Worker functions for the expert-parallel layer tests, spawned through `_init_distributed`.
def _dispatch_hands_experts_the_received_rows(rank):
    hidden_states = torch.arange(32.0).view(4, 8) + 100 * rank
    experts = EpDispatchExpertsParallel().install_forward(_DummyExperts(), init_device_mesh("cpu", (2,)))
    experts(hidden_states, torch.tensor([[0, 3], [1, 2], [2, 1], [3, 0]]), torch.ones(4, 2))
