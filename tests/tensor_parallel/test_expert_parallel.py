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

from transformers.distributed import expert_parallel
from transformers.distributed.expert_parallel import EpDispatchExpertsParallel
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test

from ..test_tensor_parallel_mixin import _init_distributed


@is_tensor_parallel_test
class TestExpertParallelLayer(TestCasePlus):
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

    def _get_parameter_placements(self, module, style, mesh):
        placements = {}
        with patch.object(
            expert_parallel, "distribute_tensor", side_effect=lambda tensor, *args, **kwargs: tensor
        ) as distribute:
            for parameter_name in list(module._parameters):
                style.shard_param(module, parameter_name, mesh)
                placements[parameter_name] = distribute.call_args.args[2][0]

        return placements

    def test_expert_parallel_styles_are_registered(self):
        """`expert_parallel.py` registers its styles in `ALL_PARALLEL_STYLES` whichever module is imported first."""
        for style in ("grouped_gemm", "ep_router", "ep_dispatch_experts", "megamoe_router", "megamoe_experts"):
            self.assertIn(style, ALL_PARALLEL_STYLES)

    def test_moved_names_still_importable_from_tensor_parallel(self):
        """The EP styles shipped from `tensor_parallel` in v5.20: the old path keeps working with a FutureWarning."""
        from transformers.distributed import tensor_parallel

        with self.assertWarns(FutureWarning):
            self.assertIs(tensor_parallel.EpDispatchExpertsParallel, EpDispatchExpertsParallel)
        with self.assertWarns(FutureWarning):
            self.assertIs(tensor_parallel.apply_expert_parallelism, expert_parallel.apply_expert_parallelism)
        with self.assertRaises(AttributeError):
            tensor_parallel.does_not_exist

    def test_grouped_gemm_updates_local_expert_count(self):
        module = torch.nn.Module()
        module.num_experts = 8
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))
        grouped_gemm = ALL_PARALLEL_STYLES["grouped_gemm"]

        placements = self._get_parameter_placements(module, grouped_gemm, self.MockDeviceMesh(world_size=4, rank=0))

        self.assertEqual(placements["weight"].dim, 0)
        self.assertEqual(module.num_experts, 2)

    def test_sharding_does_not_create_unrelated_module_attributes(self):
        module = torch.nn.Module()
        module.random_attr = 123
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))

        self._get_parameter_placements(
            module, ALL_PARALLEL_STYLES["grouped_gemm"], self.MockDeviceMesh(world_size=4, rank=0)
        )

        self.assertEqual(module.random_attr, 123)
        self.assertFalse(hasattr(module, "num_experts"))

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
