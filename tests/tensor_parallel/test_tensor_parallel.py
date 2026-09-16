# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import warnings
from unittest.mock import Mock, patch

import torch

from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeModel
from transformers.distributed import tensor_parallel
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.sharding_utils import DtensorShardOperation
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    PackedColwiseParallel,
    PackedRowwiseParallel,
    RowwiseParallel,
)
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test, require_torch


# Qwen3 MoE's predefined plans, as resolved on `Qwen3MoeModel` (no `model.` prefix).
DENSE_TP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
EXPERT_TP_PLAN = {
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
}
EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


@require_torch
class TestParallelPlanResolution(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
        )
        with torch.device("meta"):
            self.model = Qwen3MoeModel(self.config)

    def test_model_exposes_both_plans(self):
        self.assertEqual(self.model.tp_plan, DENSE_TP_PLAN | EXPERT_TP_PLAN)
        self.assertEqual(self.model.ep_plan, EP_PLAN)

    def test_ep_plan_setter(self):
        self.model.ep_plan = None
        self.assertEqual(self.model.ep_plan, {})
        with self.assertRaisesRegex(ValueError, "Can only set a dictionary"):
            self.model.ep_plan = "auto"
        with self.assertRaisesRegex(ValueError, "Unsupported parallel styles"):
            self.model.ep_plan = {"layers.*.mlp.experts": "invalid_style"}
        self.model.ep_plan = EP_PLAN
        self.assertEqual(self.model.ep_plan, EP_PLAN)

    def test_disabled_parallelism_has_no_plans(self):
        for config in (DistributedConfig(), DistributedConfig(fsdp_size=8), DistributedConfig(pp_size=2)):
            with self.subTest(config=config):
                self.assertEqual(tensor_parallel.resolve_parallel_plans(self.model, config), ({}, {}))

    def test_tp_only_keeps_experts_in_tp_plan(self):
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4))
        self.assertEqual(tp_plan, DENSE_TP_PLAN | EXPERT_TP_PLAN)
        self.assertEqual(ep_plan, {})

    def test_ep_takes_experts_and_router_out_of_tp_plan(self):
        for config in (
            DistributedConfig(tp_size=4, ep_size=4),
            DistributedConfig(tp_size=2, fsdp_size=2, ep_size=2),
        ):
            with self.subTest(config=config):
                tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, config)
                self.assertEqual(tp_plan, DENSE_TP_PLAN)
                self.assertEqual(ep_plan, EP_PLAN)

    def test_legacy_flag_is_an_alias_for_ep_size(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = DistributedConfig(tp_size=4, enable_expert_parallel=True)
        self.assertEqual([w.category for w in caught], [FutureWarning])
        self.assertIn("Use ep_size=4 instead", str(caught[0].message))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            explicit = DistributedConfig(tp_size=4, ep_size=4)
            disabled = DistributedConfig(tp_size=4, ep_size=1, enable_expert_parallel=True)
        self.assertEqual(caught, [])
        self.assertEqual(legacy, explicit)
        self.assertFalse(disabled.enable_expert_parallel)
        self.assertEqual(
            tensor_parallel.resolve_parallel_plans(self.model, legacy),
            tensor_parallel.resolve_parallel_plans(self.model, explicit),
        )

    def test_ep_plan_must_be_a_dict(self):
        with self.assertRaisesRegex(ValueError, "`ep_plan` must be a dictionary or None"):
            DistributedConfig(tp_size=4, ep_size=4, ep_plan="auto")

    def test_ep_plan_round_trip(self):
        config = DistributedConfig(tp_size=4, ep_size=4, ep_plan={"layers.*.mlp.gate": "ep_router"})
        self.assertEqual(config.to_dict()["ep_plan"], {"layers.*.mlp.gate": "ep_router"})
        self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)

    def test_overrides_merge_into_the_predefined_plans(self):
        config = DistributedConfig(
            tp_size=4,
            ep_size=4,
            tp_plan={"layers.*.self_attn.q_proj": "colwise_rep"},
            ep_plan={"layers.*.mlp.experts.down_proj": "rowwise"},
        )
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, config)
        self.assertEqual(tp_plan, DENSE_TP_PLAN | {"layers.*.self_attn.q_proj": "colwise_rep"})
        self.assertEqual(ep_plan, EP_PLAN | {"layers.*.mlp.experts.down_proj": "rowwise"})
        # The merged plans are stored on the model, the config defaults are untouched.
        self.assertEqual(self.model.tp_plan["layers.*.self_attn.q_proj"], "colwise_rep")
        self.assertEqual(self.model.ep_plan["layers.*.mlp.experts.down_proj"], "rowwise")
        self.assertEqual(self.model.config.base_model_tp_plan["layers.*.self_attn.q_proj"], "colwise")
        self.assertEqual(self.model.config.base_model_ep_plan["layers.*.mlp.experts.down_proj"], "grouped_gemm")
        # The overrides are not rewritten with the merged plans.
        self.assertEqual(config.tp_plan, {"layers.*.self_attn.q_proj": "colwise_rep"})
        self.assertEqual(config.ep_plan, {"layers.*.mlp.experts.down_proj": "rowwise"})

    def test_ep_rules_take_precedence_over_tp_rules_for_the_same_modules(self):
        config = DistributedConfig(
            tp_size=4,
            ep_size=4,
            tp_plan={"layers.*.mlp.experts.gate_up_proj": "packed_rowwise", "layers.*.mlp.gate": "colwise"},
        )
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, config)
        self.assertEqual(tp_plan, DENSE_TP_PLAN)
        self.assertEqual(ep_plan, EP_PLAN)
        # The custom TP rules are kept on the model and apply as soon as EP is disabled.
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4))
        self.assertEqual(tp_plan["layers.*.mlp.experts.gate_up_proj"], "packed_rowwise")
        self.assertEqual(tp_plan["layers.*.mlp.gate"], "colwise")
        self.assertEqual(ep_plan, {})

    def test_ep_override_is_merged_but_not_applied_when_ep_is_disabled(self):
        config = DistributedConfig(tp_size=4, ep_plan={"layers.*.mlp.experts.down_proj": "rowwise"})
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, config)
        self.assertEqual(tp_plan, DENSE_TP_PLAN | EXPERT_TP_PLAN)
        self.assertEqual(ep_plan, {})
        self.assertEqual(self.model.ep_plan["layers.*.mlp.experts.down_proj"], "rowwise")

    def test_ep_requires_an_expert_plan(self):
        self.model.ep_plan = None
        with self.assertRaisesRegex(ValueError, "does not define an expert-parallel plan"):
            tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4, ep_size=4))
        config = DistributedConfig(tp_size=4, ep_size=4, ep_plan=EP_PLAN)
        self.assertEqual(tensor_parallel.resolve_parallel_plans(self.model, config), (DENSE_TP_PLAN, EP_PLAN))

    def test_unmatched_override_keys_raise_without_changing_plans(self):
        original_tp_plan, original_ep_plan = self.model.tp_plan.copy(), self.model.ep_plan.copy()
        for plan_name in ("tp_plan", "ep_plan"):
            for key in ("layers.*.mlp.experst", "layers.*.mlp.experts.missing_weight", "model.layers.*.mlp.experts"):
                with self.subTest(plan_name=plan_name, key=key):
                    config = DistributedConfig(tp_size=4, ep_size=4, **{plan_name: {key: "grouped_gemm"}})
                    with self.assertRaisesRegex(ValueError, f"The `{plan_name}` pattern .* does not match") as error:
                        tensor_parallel.resolve_parallel_plans(self.model, config)
                    self.assertIn(key, str(error.exception))
                    self.assertIn("Qwen3MoeModel", str(error.exception))
                    self.assertEqual(self.model.tp_plan, original_tp_plan)
                    self.assertEqual(self.model.ep_plan, original_ep_plan)

    def test_override_keys_can_match_modules_parameters_or_existing_plan_keys(self):
        for plan_name in ("tp_plan", "ep_plan"):
            # `gate_proj` is in the predefined TP plan even though this MoE model has no such module.
            for key in ("layers.*.mlp", "layers.0.self_attn.q_proj.weight", "layers.*.mlp.gate_proj"):
                with self.subTest(plan_name=plan_name, key=key):
                    original = getattr(self.model, plan_name).copy()
                    if plan_name == "ep_plan":
                        self.model.ep_plan = original | {"layers.*.mlp.gate_proj": "colwise"}
                    config = DistributedConfig(tp_size=4, **{plan_name: {key: "colwise_rep"}})
                    tensor_parallel.resolve_parallel_plans(self.model, config)
                    self.assertEqual(getattr(self.model, plan_name)[key], "colwise_rep")
                    setattr(self.model, plan_name, original)

    def test_head_model_overrides_need_the_model_prefix(self):
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(self.config)
        config = DistributedConfig(tp_size=4, ep_size=4, ep_plan={"layers.*.mlp.gate": "ep_router"})
        with self.assertRaisesRegex(ValueError, "including any 'model.' prefix"):
            tensor_parallel.resolve_parallel_plans(model, config)

        config = DistributedConfig(
            tp_size=4,
            ep_size=4,
            tp_plan={"model.layers.*.self_attn.q_proj": "colwise_rep"},
            ep_plan={"model.layers.*.mlp.gate": "ep_router"},
        )
        tp_plan, ep_plan = tensor_parallel.resolve_parallel_plans(model, config)
        expected_tp_plan = {f"model.{k}": v for k, v in DENSE_TP_PLAN.items()} | {"lm_head": "colwise_gather_output"}
        self.assertEqual(tp_plan, expected_tp_plan | config.tp_plan)
        self.assertEqual(ep_plan, {f"model.{k}": v for k, v in EP_PLAN.items()})

    def test_invalid_styles_are_reported(self):
        self.model.ep_plan = None
        self.model._ep_plan = {"layers.*.mlp.experts": "invalid_style", "layers.*.mlp.gate": "another_invalid_style"}
        with self.assertRaises(ValueError) as context:
            tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4, ep_size=4))
        self.assertIn("'invalid_style'", str(context.exception))
        self.assertIn("'another_invalid_style'", str(context.exception))

    def test_resolution_does_not_apply_sharding(self):
        with (
            patch.object(tensor_parallel, "_apply_parallel_plan") as apply,
            patch.object(ALL_PARALLEL_STYLES["grouped_gemm"], "shard_param") as shard,
        ):
            tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4, ep_size=4))
        apply.assert_not_called()
        shard.assert_not_called()

    def test_masked_ep_shards_and_installs_hooks_on_the_tp_mesh(self):
        tp_mesh = object()
        _, ep_plan = tensor_parallel.resolve_parallel_plans(self.model, DistributedConfig(tp_size=4, ep_size=4))
        experts, router = self.model.layers[0].mlp.experts, self.model.layers[0].mlp.gate
        with (
            patch.object(ALL_PARALLEL_STYLES["grouped_gemm"], "validate_param") as validate,
            patch.object(ALL_PARALLEL_STYLES["grouped_gemm"], "shard_param") as shard,
            patch.object(ALL_PARALLEL_STYLES["moe_tp_experts"], "install_forward") as install_experts,
            patch.object(ALL_PARALLEL_STYLES["ep_router"], "install_forward") as install_router,
        ):
            result = tensor_parallel.apply_masked_expert_parallelism(self.model, tp_mesh, ep_plan)
        self.assertIs(result, self.model)
        self.assertEqual(shard.call_count, 2)
        for name in ("gate_up_proj", "down_proj"):
            validate.assert_any_call(experts, name, tp_mesh, parameter_name=f"layers.0.mlp.experts.{name}")
            shard.assert_any_call(experts, name, tp_mesh)
        install_experts.assert_called_once_with(experts, tp_mesh)
        install_router.assert_called_once_with(router, tp_mesh)

    def test_maybe_distribute_model_applies_tp_and_ep_on_the_tp_mesh(self):
        tp_mesh, fsdp_mesh = object(), object()
        meshes = Mock()
        meshes.get_mesh.side_effect = lambda dims: {"tp": tp_mesh, "fsdp": fsdp_mesh}.get(dims, Mock())
        for config, expected_tp_plan, expected_ep_plan in (
            (DistributedConfig(tp_size=4), DENSE_TP_PLAN | EXPERT_TP_PLAN, None),
            (DistributedConfig(tp_size=4, ep_size=4), DENSE_TP_PLAN, EP_PLAN),
            (DistributedConfig(tp_size=2, fsdp_size=2, ep_size=2), DENSE_TP_PLAN, EP_PLAN),
            (DistributedConfig(fsdp_size=4), None, None),
        ):
            with (
                self.subTest(config=config),
                patch("transformers.distributed.mixin.apply_tensor_parallelism", return_value=self.model) as tp,
                patch("transformers.distributed.mixin.apply_masked_expert_parallelism", return_value=self.model) as ep,
                patch(
                    "transformers.distributed.mixin.apply_fully_sharded_data_parallelism", return_value=self.model
                ) as fsdp,
            ):
                result = self.model.maybe_distribute_model(self.model, config, meshes)
                self.assertIs(result, self.model)
                self.assertIs(self.model.config.distributed_config, config)
                if expected_tp_plan is None:
                    tp.assert_not_called()
                else:
                    tp.assert_called_once_with(self.model, tp_mesh, expected_tp_plan)
                if expected_ep_plan is None:
                    ep.assert_not_called()
                else:
                    ep.assert_called_once_with(self.model, tp_mesh, expected_ep_plan)
                if config.fsdp_size > 1:
                    fsdp.assert_called_once_with(self.model, fsdp_mesh)
                else:
                    fsdp.assert_not_called()

    def test_maybe_distribute_model_without_meshes_is_a_no_op(self):
        with patch("transformers.distributed.mixin.resolve_parallel_plans") as resolve:
            self.assertIs(self.model.maybe_distribute_model(self.model, DistributedConfig(), None), self.model)
        resolve.assert_not_called()


@is_tensor_parallel_test
class TestTensorParallelProperties(TestCasePlus):
    def test_tp_plan_property_setter_getter(self):
        """Test that tp_plan property can be set and retrieved correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting empty plan
        model.tp_plan = {}
        self.assertEqual(model.tp_plan, {})

        # Test setting a valid plan
        valid_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        model.tp_plan = valid_plan
        self.assertEqual(model.tp_plan, valid_plan)

        # Test updating the plan
        model.tp_plan.update({"model.layers.*.self_attn.k_proj": "colwise"})
        expected_plan = {"model.layers.*.self_attn.q_proj": "colwise", "model.layers.*.self_attn.k_proj": "colwise"}
        self.assertEqual(model.tp_plan, expected_plan)

        # Test overriding existing entry
        model.tp_plan.update({"model.layers.*.self_attn.q_proj": "rowwise"})
        expected_plan = {
            "model.layers.*.self_attn.q_proj": "rowwise",
            "model.layers.*.self_attn.k_proj": "colwise",
        }
        self.assertEqual(model.tp_plan, expected_plan)

    def test_tp_plan_validation_invalid_style(self):
        """Test that invalid parallel styles are rejected."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        invalid_plan = {
            "layers.*.self_attn.q_proj": "invalid_style",
            "layers.*.self_attn.k_proj": "another_invalid_style",
        }
        with self.assertRaises(ValueError) as context:
            model.tp_plan = invalid_plan

        error_message = str(context.exception)
        for style in invalid_plan.values():
            self.assertIn(repr(style), error_message)
        self.assertIn("Supported styles are", error_message)

    def test_apply_tensor_parallelism_reports_all_invalid_styles(self):
        model = torch.nn.Module()
        model.tp_plan = {
            "first_layer": "invalid_style",
            "second_layer": "another_invalid_style",
        }

        with self.assertRaises(ValueError) as context:
            tensor_parallel.apply_tensor_parallelism(model, tp_mesh=None)

        error_message = str(context.exception)
        self.assertIn("'invalid_style'", error_message)
        self.assertIn("'another_invalid_style'", error_message)

    def test_tp_plan_validation_nonexistent_layer_warning(self):
        """Test that warnings are issued for non-existent layer patterns."""

        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test warning for non-existent layer pattern
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.tp_plan = {"nonexistent.*.layer": "colwise"}

            # Check that a warning was issued
            self.assertTrue(len(w) > 0)
            warning_message = str(w[0].message)
            self.assertIn("Layer pattern 'nonexistent.*.layer' does not match any parameters", warning_message)

    def test_tp_plan_valid_layer_patterns(self):
        """Test that valid layer patterns are accepted without warnings."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test valid layer patterns that should match the model structure
        valid_plans = [
            {"model.layers.*.self_attn.q_proj": "colwise"},
            {"model.layers.*.self_attn.k_proj": "rowwise"},
            {"model.layers.*.mlp.gate_proj": "colwise"},
        ]

        for plan in valid_plans:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.tp_plan = plan

                # Filter out any warnings that are not about layer patterns
                layer_warnings = [
                    warning
                    for warning in w
                    if "Layer pattern" in str(warning.message)
                    and "does not match any parameters" in str(warning.message)
                ]

                # Should not have layer pattern warnings for valid patterns
                self.assertEqual(
                    len(layer_warnings),
                    0,
                    f"Unexpected warning for valid pattern {plan}: {[str(w.message) for w in layer_warnings]}",
                )

        # Verify the final plan was set correctly
        self.assertEqual(model.tp_plan, valid_plans[-1])

    def test_tp_plan_none_handling(self):
        """Test that None values are handled correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting None
        model.tp_plan = None
        self.assertEqual(model.tp_plan, {})

        # Test setting a plan after None
        model.tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        self.assertEqual(model.tp_plan, {"model.layers.*.self_attn.q_proj": "colwise"})

    def test_post_init_keeps_class_level_plans(self):
        """Class-level plans (e.g. `lm_head` on ForCausalLM classes) must survive post_init alongside the base model plan."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        self.assertIn("lm_head", model._tp_plan)
        self.assertIn("model.layers.*.self_attn.q_proj", model._tp_plan)
        self.assertIn("lm_head", model._pp_plan)
        # The merge must not have mutated the class attribute shared by all instances
        self.assertEqual(set(type(model)._tp_plan), {"lm_head"})


@is_tensor_parallel_test
class TestTensorParallelLayer(TestCasePlus):
    class MockDeviceMesh:
        def __init__(self, world_size, rank):
            self.world_size = world_size
            self.rank = rank
            self.shape = (world_size,)
            self.ndim = 1

        def size(self):
            return self.world_size

        def get_local_rank(self):
            return self.rank

    def _get_parameter_placements(self, module, style, mesh=None):
        placements = {}
        mesh = object() if mesh is None else mesh
        with patch.object(
            tensor_parallel, "distribute_tensor", side_effect=lambda tensor, *args, **kwargs: tensor
        ) as distribute:
            for parameter_name in list(module._parameters):
                style.shard_param(module, parameter_name, mesh)
                placements[parameter_name] = distribute.call_args.args[2][0]

        return placements

    def _get_local_shape(self, global_shape, placement, world_size, rank):
        if placement.is_replicate():
            return tuple(global_shape)

        shard_dim = placement.dim
        local_size, _ = placement._local_shard_size_and_offset(global_shape[shard_dim], world_size, rank)
        local_shape = list(global_shape)
        local_shape[shard_dim] = local_size
        return tuple(local_shape)

    def _make_dtensor_shard_op(self, mesh, placement, param_shape, local_shape):
        op = object.__new__(DtensorShardOperation)
        op.device_mesh = mesh
        op.placements = (placement,)
        op.param_ndim = len(param_shape)
        op._axis0_offset = 0
        op._axis0_local_size = local_shape[0]
        return op

    def test_colwise_gather_output_rejects_indivisible_out_features(self):
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(8, 99)
        model.tp_plan = {"lm_head": "colwise_gather_output"}
        device_mesh = self.MockDeviceMesh(world_size=2, rank=0)

        with self.assertRaises(ValueError) as context:
            tensor_parallel.apply_tensor_parallelism(model, device_mesh)

        self.assertIn("lm_head", str(context.exception))
        self.assertIn("divisible", str(context.exception))

    def test_colwise_uneven_local_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(10, 32)))
        module.register_parameter("bias", torch.nn.Parameter(torch.empty(10)))
        placements = self._get_parameter_placements(module, ColwiseParallel())
        expected_local_sizes = (4, 4, 2)

        for rank, expected_size in enumerate(expected_local_sizes):
            weight_shape = self._get_local_shape((10, 32), placements["weight"], world_size=3, rank=rank)
            bias_shape = self._get_local_shape((10,), placements["bias"], world_size=3, rank=rank)

            self.assertEqual(weight_shape, (expected_size, 32))
            self.assertEqual(bias_shape, (expected_size,))

    def test_rowwise_uneven_local_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(32, 10)))
        module.register_parameter("bias", torch.nn.Parameter(torch.empty(10)))
        placements = self._get_parameter_placements(module, RowwiseParallel())
        expected_local_sizes = (4, 4, 2)

        for rank, expected_size in enumerate(expected_local_sizes):
            weight_shape = self._get_local_shape((32, 10), placements["weight"], world_size=3, rank=rank)
            bias_shape = self._get_local_shape((10,), placements["bias"], world_size=3, rank=rank)

            self.assertEqual(weight_shape, (32, expected_size))
            self.assertEqual(bias_shape, (10,))

    def test_embedding_uneven_local_shapes(self):
        rowwise_embedding = torch.nn.Embedding(10, 10)
        rowwise_placement = self._get_parameter_placements(rowwise_embedding, RowwiseParallel())["weight"]

        colwise_embedding = torch.nn.Embedding(10, 10)
        colwise_placement = self._get_parameter_placements(colwise_embedding, ColwiseParallel())["weight"]

        expected_local_sizes = (4, 4, 2)
        for rank, expected_size in enumerate(expected_local_sizes):
            rowwise_shape = self._get_local_shape((10, 10), rowwise_placement, world_size=3, rank=rank)
            colwise_shape = self._get_local_shape((10, 10), colwise_placement, world_size=3, rank=rank)

            self.assertEqual(rowwise_shape, (expected_size, 10))
            self.assertEqual(colwise_shape, (10, expected_size))

    def test_shard_tensor_shape_consistency(self):
        world_size = 4
        cases = {
            "colwise": {
                "module": torch.nn.Linear(32, 16),
                "style": ColwiseParallel(),
                "expected_shapes": {"weight": (4, 32), "bias": (4,)},
            },
            "colwise_gather_output": {
                "module": torch.nn.Linear(32, 16),
                "style": ALL_PARALLEL_STYLES["colwise_gather_output"],
                "expected_shapes": {"weight": (4, 32), "bias": (4,)},
            },
            "rowwise": {
                "module": torch.nn.Linear(32, 16),
                "style": RowwiseParallel(),
                "expected_shapes": {"weight": (16, 8), "bias": (16,)},
            },
            "embedding_rowwise": {
                "module": torch.nn.Embedding(32, 16),
                "style": ALL_PARALLEL_STYLES["embedding_rowwise"],
                "expected_shapes": {"weight": (8, 16)},
            },
            "embedding_colwise": {
                "module": torch.nn.Embedding(32, 16),
                "style": ColwiseParallel(),
                "expected_shapes": {"weight": (32, 4)},
            },
        }

        for case_name, case in cases.items():
            module = case["module"]
            placements = self._get_parameter_placements(module, case["style"])

            for parameter_name, expected_shape in case["expected_shapes"].items():
                global_shape = module._parameters[parameter_name].shape
                placement = placements[parameter_name]

                for rank in range(world_size):
                    with self.subTest(case=case_name, parameter=parameter_name, rank=rank):
                        local_shape = self._get_local_shape(global_shape, placement, world_size, rank)
                        self.assertEqual(local_shape, expected_shape)

    def test_packed_colwise_packed_and_unpacked_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(2, 16, 64)))
        placement = self._get_parameter_placements(module, PackedColwiseParallel())["weight"]
        packed = torch.randn(2, 16, 64)
        unpacked_expert = torch.randn(16, 64)

        self.assertEqual(placement.dim, 1)
        self.assertEqual(placement.split_factor, 2)
        for rank in range(2):
            mesh = self.MockDeviceMesh(world_size=2, rank=rank)
            op = self._make_dtensor_shard_op(mesh, placement, param_shape=(2, 16, 64), local_shape=(2, 8, 64))

            self.assertEqual(op.shard_tensor(packed).shape, (2, 8, 64))
            self.assertEqual(op.shard_tensor(unpacked_expert, tensor_idx=0).shape, (8, 64))

    def test_packed_rowwise_packed_and_unpacked_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(16, 64)))
        placement = self._get_parameter_placements(module, PackedRowwiseParallel())["weight"]
        packed = torch.randn(16, 64)
        unpacked = torch.randn(16, 32)

        self.assertEqual(placement.dim, -1)
        self.assertEqual(placement.split_factor, 2)
        for rank in range(2):
            mesh = self.MockDeviceMesh(world_size=2, rank=rank)
            op = self._make_dtensor_shard_op(mesh, placement, param_shape=(16, 64), local_shape=(16, 32))

            self.assertEqual(op.shard_tensor(packed).shape, (16, 32))
            self.assertEqual(op.shard_tensor(unpacked).shape, (16, 16))

    def test_grouped_gemm_updates_local_expert_count(self):
        module = torch.nn.Module()
        module.num_experts = 8
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))
        grouped_gemm = ALL_PARALLEL_STYLES["grouped_gemm"]

        placements = self._get_parameter_placements(module, grouped_gemm, self.MockDeviceMesh(world_size=4, rank=0))

        self.assertEqual(placements["weight"].dim, 0)
        self.assertEqual(module.num_experts, 2)

    def test_sharding_does_not_create_unrelated_module_attributes(self):
        styles = (ColwiseParallel(), RowwiseParallel(), ALL_PARALLEL_STYLES["grouped_gemm"])

        for style in styles:
            with self.subTest(style=type(style).__name__):
                module = torch.nn.Module()
                module.random_attr = 123
                module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))

                self._get_parameter_placements(module, style, self.MockDeviceMesh(world_size=4, rank=0))

                self.assertEqual(module.random_attr, 123)
                self.assertFalse(hasattr(module, "num_experts"))
