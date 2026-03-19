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

import torch

from transformers import AutoModelForCausalLM
from transformers.integrations.tensor_parallel import (
    ColwiseParallel,
    EmbeddingParallel,
    GroupedGemmParallel,
    PackedColwiseParallel,
    PackedRowwiseParallel,
    RowwiseParallel,
    get_packed_weights,
    repack_weights,
)
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test


@is_tensor_parallel_test
class TestTensorParallelUtils(TestCasePlus):
    def test_packed_unpacked_conversion(self):
        WORLD_SIZE = 2
        PACKED_BLOCK_SIZE = 800
        SHARDING_DIM = 2
        NUM_BLOCKS = 2

        original_packed_weights = torch.randn(4, 512, 2 * PACKED_BLOCK_SIZE)
        original_packed_weights.get_dtype = lambda: "F32"  # get_packed_weights expects PySlice object
        empty_param = torch.empty(4, 512, 2 * PACKED_BLOCK_SIZE)

        class MockDeviceMesh:
            def size(self):
                return WORLD_SIZE

        mock_mesh = (
            MockDeviceMesh()
        )  # get_packed_weights only calls `.size()`, do this to avoid doing actual distributed run

        packed_weights_0 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 0, SHARDING_DIM)
        packed_weights_1 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 1, SHARDING_DIM)

        # simulate all gather of sharded weights
        packed_weights = torch.cat([packed_weights_0, packed_weights_1], dim=SHARDING_DIM)
        unpacked_weights = repack_weights(packed_weights, SHARDING_DIM, WORLD_SIZE, NUM_BLOCKS)

        assert torch.allclose(unpacked_weights, original_packed_weights)


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

        # Test invalid parallel style
        with self.assertRaises(ValueError) as context:
            model.tp_plan = {"layers.*.self_attn.q_proj": "invalid_style"}

        self.assertIn("Unsupported tensor parallel style 'invalid_style'", str(context.exception))
        self.assertIn("Supported styles are", str(context.exception))

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


@is_tensor_parallel_test
class TestTensorParallelLayer(TestCasePlus):
    class MockDeviceMesh:
        def __init__(self, world_size, rank):
            self.world_size = world_size
            self.rank = rank
            self.shape = (world_size,)

        def size(self):
            return self.world_size

        def get_local_rank(self):
            return self.rank

    def test_colwise_get_expected_sharded_shape(self):
        world_size = 3
        size = 10  # not divisible by world_size to test edge case
        empty_param_2d = torch.empty(size, 32)
        empty_param_1d = torch.empty((size,))
        step = math.ceil(size / world_size)

        for rank in range(world_size):
            for empty_param in [empty_param_2d, empty_param_1d]:
                device_mesh = self.MockDeviceMesh(world_size=world_size, rank=rank)
                layer = ColwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty_param)

                begin = rank * step
                end = min(begin + step, size)
                ground_truth = (end - begin,) + empty_param.shape[1:]
                expected_shape = layer.get_expected_sharded_shape(empty_param.shape)
                self.assertEqual(
                    expected_shape, ground_truth, f"Rank {rank} expected shape {ground_truth} but got {expected_shape}"
                )

    def test_rowwise_get_expected_sharded_shape(self):
        world_size = 3
        size = 10  # not divisible by world_size to test edge case
        empty_param_2d = torch.empty(32, size)
        empty_param_1d = torch.empty((size,))
        step = math.ceil(size / world_size)

        for rank in range(world_size):
            device_mesh = self.MockDeviceMesh(world_size=world_size, rank=rank)

            # 2D: shards on dim -1 (input features)
            layer = RowwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty_param_2d)
            begin = rank * step
            end = min(begin + step, size)
            ground_truth = empty_param_2d.shape[:-1] + (end - begin,)
            expected_shape = layer.get_expected_sharded_shape(empty_param_2d.shape)
            self.assertEqual(
                expected_shape, ground_truth, f"Rank {rank} expected shape {ground_truth} but got {expected_shape}"
            )

            # 1D bias: NOT sharded
            layer = RowwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty_param_1d)
            self.assertEqual(layer.get_expected_sharded_shape(empty_param_1d.shape), empty_param_1d.shape)

    def test_embedding_get_expected_sharded_shape(self):
        world_size = 3
        size = 10  # not divisible by world_size to test edge case; same size on both dims so step applies to both
        empty_param = torch.empty(size, size)
        step = math.ceil(size / world_size)

        for rank in range(world_size):
            device_mesh = self.MockDeviceMesh(world_size=world_size, rank=rank)
            begin = rank * step
            end = min(begin + step, size)

            # embedding_dim_sharding=0: shards dim 0 (vocab)
            layer = EmbeddingParallel(
                device_mesh=device_mesh, rank=rank, empty_param=empty_param, embedding_dim_sharding=0
            )
            ground_truth = (end - begin,) + empty_param.shape[1:]
            expected_shape = layer.get_expected_sharded_shape(empty_param.shape)
            self.assertEqual(
                expected_shape, ground_truth, f"Rank {rank} expected shape {ground_truth} but got {expected_shape}"
            )

            # embedding_dim_sharding=1: shards dim 1 (embedding dim)
            layer = EmbeddingParallel(
                device_mesh=device_mesh, rank=rank, empty_param=empty_param, embedding_dim_sharding=1
            )
            ground_truth = empty_param.shape[:1] + (end - begin,) + empty_param.shape[2:]
            expected_shape = layer.get_expected_sharded_shape(empty_param.shape)
            self.assertEqual(
                expected_shape, ground_truth, f"Rank {rank} expected shape {ground_truth} but got {expected_shape}"
            )

    def test_grouped_gemm_get_expected_sharded_shape(self):
        world_size = 3
        size = 9  # must be divisible by world_size (GroupedGemm requires it)
        empty_param = torch.empty(size, 16, 32)
        step = math.ceil(size / world_size)

        for rank in range(world_size):
            device_mesh = self.MockDeviceMesh(world_size=world_size, rank=rank)
            layer = GroupedGemmParallel(device_mesh=device_mesh, rank=rank, empty_param=empty_param)
            begin = rank * step
            end = min(begin + step, size)
            ground_truth = (end - begin,) + empty_param.shape[1:]
            expected_shape = layer.get_expected_sharded_shape(empty_param.shape)
            self.assertEqual(
                expected_shape, ground_truth, f"Rank {rank} expected shape {ground_truth} but got {expected_shape}"
            )

    def test_colwise_update_module_attributes(self):
        device_mesh = self.MockDeviceMesh(world_size=4, rank=0)

        # gather_output=False (default): out_features is updated
        module = torch.nn.Linear(32, 16)
        layer = ColwiseParallel(device_mesh=device_mesh, rank=0, empty_param=torch.empty(16, 32))
        layer.update_module_attributes(module)
        self.assertEqual(module.out_features, 4)

        # gather_output=True: out_features is NOT updated
        module = torch.nn.Linear(32, 16)
        layer = ColwiseParallel(device_mesh=device_mesh, rank=0, empty_param=torch.empty(16, 32), gather_output=True)
        layer.update_module_attributes(module)
        self.assertEqual(module.out_features, 16)

    def test_rowwise_update_module_attributes(self):
        device_mesh = self.MockDeviceMesh(world_size=4, rank=0)

        module = torch.nn.Linear(32, 16)
        layer = RowwiseParallel(device_mesh=device_mesh, rank=0, empty_param=torch.empty(16, 32))
        layer.update_module_attributes(module)
        self.assertEqual(module.in_features, 8)

    def test_embedding_update_module_attributes(self):
        device_mesh = self.MockDeviceMesh(world_size=4, rank=0)

        # embedding_dim_sharding=0: num_embeddings is updated
        module = torch.nn.Embedding(32, 16)
        layer = EmbeddingParallel(
            device_mesh=device_mesh, rank=0, empty_param=torch.empty(32, 16), embedding_dim_sharding=0
        )
        layer.update_module_attributes(module)
        self.assertEqual(module.num_embeddings, 8)
        self.assertEqual(module.embedding_dim, 16)

        # embedding_dim_sharding=1: embedding_dim is updated
        module = torch.nn.Embedding(32, 16)
        layer = EmbeddingParallel(
            device_mesh=device_mesh, rank=0, empty_param=torch.empty(32, 16), embedding_dim_sharding=1
        )
        layer.update_module_attributes(module)
        self.assertEqual(module.num_embeddings, 32)
        self.assertEqual(module.embedding_dim, 4)

    def test_grouped_gemm_update_module_attributes(self):
        device_mesh = self.MockDeviceMesh(world_size=4, rank=0)

        # There is no torch module with num_experts attribute, it is more at the Transformers level,
        # so just use a SimpleNamespace to test that the attribute is updated correctly.
        module = SimpleNamespace(num_experts=8)
        layer = GroupedGemmParallel(device_mesh=device_mesh, rank=0, empty_param=torch.empty(8, 16, 32))
        layer.update_module_attributes(module)
        self.assertEqual(module.num_experts, 2)

    def test_update_module_attributes_missing_attribute(self):
        device_mesh = self.MockDeviceMesh(world_size=4, rank=0)
        module = SimpleNamespace(random_attr=123)
        for cls in [ColwiseParallel, RowwiseParallel, GroupedGemmParallel]:
            layer = cls(device_mesh=device_mesh, rank=0, empty_param=torch.empty(16, 32))
            layer.update_module_attributes(module)

        self.assertEqual(
            module.__dict__,
            {"random_attr": 123},
            "update_module_attributes should not modify attributes that don't exist",
        )

    def test_shard_tensor_shape_consistency(self):
        """
        Test that shard_tensor returns tensors of the expected shape for different parallel styles and ranks.
        """
        WORLD_SIZE = 4
        cases = [
            (ColwiseParallel, (16, 32), {}),
            (ColwiseParallel, (16, 32), {"gather_output": True}),
            (ColwiseParallel, (16,), {}),
            (RowwiseParallel, (16, 32), {}),
            (RowwiseParallel, (32,), {}),
            (EmbeddingParallel, (32, 16), {"embedding_dim_sharding": 0}),
            (EmbeddingParallel, (32, 16), {"embedding_dim_sharding": 1}),
        ]
        for cls, shape, kwargs in cases:
            for rank in range(WORLD_SIZE):
                device_mesh = self.MockDeviceMesh(world_size=WORLD_SIZE, rank=rank)
                layer = cls(device_mesh=device_mesh, rank=rank, empty_param=torch.empty(*shape), **kwargs)

                full_tensor = torch.randn(*shape)
                sharded = layer.shard_tensor(full_tensor)
                expected = layer.get_expected_sharded_shape(shape)

                self.assertEqual(tuple(sharded.shape), expected, f"{cls.__name__} rank={rank} shape={shape}")

    def test_packed_colwise_shard_tensor(self):
        WORLD_SIZE = 2
        # 3D empty_param
        empty = torch.empty(2, 16, 64)

        # Packed vs unpacked path is determined by checking the following:
        # input.dim() == get_expected_sharded_shape(empty_param).dim()

        # Packed
        full_packed = torch.randn(2, 16, 64)
        full_packed.get_dtype = lambda: "F32"
        for rank in range(WORLD_SIZE):
            device_mesh = self.MockDeviceMesh(world_size=WORLD_SIZE, rank=rank)
            layer = PackedColwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty)
            sharded = layer.shard_tensor(full_packed)
            expected_shape = (2, 8, 64)  # last dim is packed size, middle dim is sharded
            self.assertEqual(sharded.shape, expected_shape)

        # Unpacked
        full_unpacked = torch.randn(16, 64)
        for rank in range(WORLD_SIZE):
            device_mesh = self.MockDeviceMesh(world_size=WORLD_SIZE, rank=rank)
            layer = PackedColwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty)
            sharded = layer.shard_tensor(full_unpacked)
            expected_shape = (8, 64)  # last dim is not packed, so just sharded
            self.assertEqual(sharded.shape, expected_shape)

    def test_packed_rowwise_shard_tensor(self):
        WORLD_SIZE = 2
        # empty_param last dim = 64 signals the packed size (2 * 32)
        empty = torch.empty(16, 64)

        # Packed vs unpacked path is determined by checking the following:
        # input.shape[-1] < empty_param.shape[-1]

        # Packed
        full_packed = torch.randn(16, 64)
        full_packed.get_dtype = lambda: "F32"
        for rank in range(WORLD_SIZE):
            device_mesh = self.MockDeviceMesh(world_size=WORLD_SIZE, rank=rank)
            layer = PackedRowwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty)
            sharded = layer.shard_tensor(full_packed)
            expected_shape = (16, 32)  # last dim is packed size, sharded
            self.assertEqual(sharded.shape, expected_shape)

        # Unpacked
        full_unpacked = torch.randn(16, 32)
        for rank in range(WORLD_SIZE):
            device_mesh = self.MockDeviceMesh(world_size=WORLD_SIZE, rank=rank)
            layer = PackedRowwiseParallel(device_mesh=device_mesh, rank=rank, empty_param=empty)
            sharded = layer.shard_tensor(full_unpacked)
            expected_shape = (16, 16)  # last dim is not packed, so just sharded
            self.assertEqual(sharded.shape, expected_shape)
