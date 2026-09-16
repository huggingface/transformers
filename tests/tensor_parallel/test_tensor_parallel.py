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
import math
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import torch
from parameterized import parameterized
from torch import nn

from transformers import AutoModelForCausalLM
from transformers.distributed import tensor_parallel
from transformers.distributed.sharding_utils import DtensorShardOperation
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    PackedColwiseParallel,
    PackedRowwiseParallel,
    RowwiseParallel,
    _unit_mesh,
)
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test


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


class MockAttention(nn.Module):
    """The four projections `unit_colwise`/`unit_rowwise` are meant to be used on."""

    def __init__(self, num_heads, num_key_value_heads, head_dim, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        # `apply_tensor_parallelism` attaches this for styles that set `needs_config`.
        config = SimpleNamespace(
            head_dim=head_dim,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
        )
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            proj.config = config


@is_tensor_parallel_test
class TestUnitParallel(TestCasePlus):
    """`unit_colwise`/`unit_rowwise` must give every rank whole heads, and never split one."""

    HEAD_DIM = 8
    HIDDEN = 64

    class FakeMesh:
        """Enough of a DeviceMesh for `_unit_mesh` to reshape, with no process groups."""

        def __init__(self, ranks, rank=0):
            self.mesh, self.rank = ranks, rank
            self.device_type, self.mesh_dim_names = "cpu", None
            self.ndim = ranks.ndim

        def size(self):
            return int(self.mesh.numel())

        def get_local_rank(self):
            return self.rank

    def _shard(self, module, name, style, mesh):
        """Shard one projection, returning the placements it asked for."""
        with (
            patch.object(
                tensor_parallel, "DeviceMesh", lambda device_type, mesh, mesh_dim_names=None: self.FakeMesh(mesh)
            ),
            patch.object(
                tensor_parallel, "distribute_tensor", side_effect=lambda tensor, *a, **kw: tensor
            ) as distribute,
        ):
            style.shard_param(module, name, mesh)
            return distribute.call_args.args[2]

    @parameterized.expand(
        [
            (2, 2, 4),  # fewer heads than ranks -> replicated
            (6, 6, 4),  # heads do not divide the world
            (7, 7, 4),  # coprime -> fully replicated
            (8, 8, 4),  # exact fit, no replication
            (12, 12, 8),
            (2, 2, 8),
            (32, 8, 8),  # GQA
            (32, 8, 16),  # GQA, more ranks than kv heads
            (8, 1, 4),  # MQA
        ]
    )
    def test_every_rank_owns_whole_heads(self, num_heads, num_key_value_heads, world_size):
        mesh = self.FakeMesh(torch.arange(world_size))
        with patch.object(
            tensor_parallel, "DeviceMesh", lambda device_type, mesh, mesh_dim_names=None: self.FakeMesh(mesh)
        ):
            for units in (num_heads, num_key_value_heads):
                groups, replicas = _unit_mesh(mesh, units).mesh.shape
                self.assertEqual(groups * replicas, world_size, "the mesh must cover every rank")
                self.assertEqual(groups, math.gcd(units, world_size), "replication must be the least possible")
                self.assertEqual(units % groups, 0, "a head may not be split across ranks")
                self.assertGreaterEqual(units // groups, 1, "no rank may be left without a head")

    @parameterized.expand([(2, 2, 4), (6, 6, 4), (8, 8, 4), (32, 8, 8), (32, 8, 16), (8, 1, 4)])
    def test_projections_are_sharded_on_the_head_axis(self, num_heads, num_key_value_heads, world_size):
        attention = MockAttention(num_heads, num_key_value_heads, self.HEAD_DIM, self.HIDDEN)
        mesh = self.FakeMesh(torch.arange(world_size))
        colwise, rowwise = ALL_PARALLEL_STYLES["unit_colwise"], ALL_PARALLEL_STYLES["unit_rowwise"]

        for name, units in (("q_proj", num_heads), ("k_proj", num_key_value_heads), ("v_proj", num_key_value_heads)):
            proj = getattr(attention, name)
            placements = self._shard(proj, "weight", colwise, mesh)
            groups = math.gcd(units, world_size)
            self.assertEqual(placements[0].dim, 0, f"{name} shards its output features")
            self.assertTrue(placements[1].is_replicate(), f"{name} replicates along the second axis")
            self.assertEqual(proj.weight.shape[0] // groups % self.HEAD_DIM, 0, f"{name} split a head")

        placements = self._shard(attention.o_proj, "weight", rowwise, mesh)
        self.assertEqual(placements[0].dim, 1, "o_proj shards its input features")
        self.assertTrue(placements[1].is_replicate())
        self.assertEqual(attention.o_proj.weight.shape[1] // math.gcd(num_heads, world_size) % self.HEAD_DIM, 0)

    @parameterized.expand([(2, 4), (6, 4), (8, 4), (12, 8), (32, 16)])
    def test_colwise_and_rowwise_agree_on_the_mesh(self, num_heads, world_size):
        """o_proj consumes q_proj's output, so the two must land on the same unit mesh."""
        attention = MockAttention(num_heads, num_heads, self.HEAD_DIM, self.HIDDEN)
        mesh = self.FakeMesh(torch.arange(world_size))
        self._shard(attention.q_proj, "weight", ALL_PARALLEL_STYLES["unit_colwise"], mesh)
        self._shard(attention.o_proj, "weight", ALL_PARALLEL_STYLES["unit_rowwise"], mesh)
        self.assertEqual(
            tuple(attention.q_proj._unit_mesh.mesh.shape),
            tuple(attention.o_proj._unit_mesh.mesh.shape),
        )
