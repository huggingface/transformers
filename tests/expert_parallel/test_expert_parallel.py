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

import pytest
import torch
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from transformers.distributed import expert_parallel
from transformers.distributed.expert_parallel import (
    AllReduceEP,
    AllToAllEP,
    AutoEP,
    EpDispatchExpertsParallel,
    EPImplementation,
)
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES
from transformers.testing_utils import TestCasePlus, is_expert_parallel_test

from ..distributed_testing_utils import run_distributed


@is_expert_parallel_test
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
        run_distributed(world_size=2, backend="gloo")(_dispatch_hands_experts_the_received_rows)()

    @parameterized.expand(
        [
            ("all_to_all", "train"),
            ("all_to_all", "inference"),
            ("all_reduce", "train"),
            ("all_reduce", "inference"),
            ("auto", "checkpoint"),
            ("auto", "reentrant_checkpoint"),
        ]
    )
    def test_ep_implementation_matches_one_process(self, ep_implementation, mode):
        """Each implementation gives the single-process output and input gradients, also under gradient checkpointing,
        where the reentrant mode runs `"auto"` without gradients first (all-reduce) and recomputes it with them
        (all-to-all)."""
        run_distributed(world_size=2, backend="gloo")(_ep_implementation_matches_one_process)(ep_implementation, mode)

    def test_auto_ep_picks_all_reduce_only_at_inference_on_a_shared_batch(self):
        run_distributed(world_size=2, backend="gloo")(_auto_ep_picks_all_reduce_only_at_inference_on_a_shared_batch)()

    def test_install_forward_checks_ep_implementation(self):
        """An unknown name and `"all_reduce"` without a shared batch are refused; a registered implementation runs."""
        run_distributed(world_size=2, backend="gloo")(_install_forward_checks_ep_implementation)()


class _DummyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2

    def forward(self, tokens, expert_ids, weights):
        assert type(tokens) is torch.Tensor, type(tokens)
        return tokens


# Worker functions for the expert-parallel layer tests, spawned through `run_distributed`.
def _dispatch_hands_experts_the_received_rows(rank):
    hidden_states = torch.arange(32.0).view(4, 8) + 100 * rank
    mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "tp"))
    experts = EpDispatchExpertsParallel().install_forward(_DummyExperts(), mesh["ep"], tp_mesh=mesh["tp"])
    experts(hidden_states, torch.tensor([[0, 3], [1, 2], [2, 1], [3, 0]]), torch.ones(4, 2))


def _install_forward_checks_ep_implementation(rank):
    mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "tp"))
    ep_mesh, tp_mesh = mesh["ep"], mesh["tp"]
    # an unknown name, and `"all_reduce"` on ranks holding different batches
    for ep_implementation in ("fast", "all_reduce"):
        with pytest.raises(ValueError):
            EpDispatchExpertsParallel().install_forward(
                _DummyExperts(), ep_mesh, tp_mesh=tp_mesh, ep_implementation=ep_implementation
            )

    class Custom(EPImplementation):
        def __call__(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states + 1

    with patch.dict(EpDispatchExpertsParallel.ep_implementations, {"custom": Custom}):
        experts = EpDispatchExpertsParallel().install_forward(
            _DummyExperts(), ep_mesh, tp_mesh=tp_mesh, ep_implementation="custom"
        )
    torch.testing.assert_close(experts(torch.zeros(2, 4), None, torch.ones(2, 1)), torch.ones(2, 4))


def _auto_ep_picks_all_reduce_only_at_inference_on_a_shared_batch(rank):
    mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "tp"))
    ep_mesh, single = mesh["ep"], mesh["tp"]
    for tp_mesh, grad, all_reduce in ((ep_mesh, False, True), (ep_mesh, True, False), (single, False, False)):
        with (
            patch.object(AllReduceEP, "__call__") as all_reduce_call,
            patch.object(AllToAllEP, "__call__") as all_to_all_call,
            torch.set_grad_enabled(grad),
        ):
            AutoEP(_DummyExperts(), ep_mesh, tp_mesh)(None, None, None)
        assert (all_reduce_call.called, all_to_all_call.called) == (all_reduce, not all_reduce), (tp_mesh, grad)


class _ScaledExperts(torch.nn.Module):
    """Toy experts: expert `i` multiplies a token by `scales[i]`. The id `num_experts` marks a route to another rank's
    expert (`AllReduceEP` masks them that way) and adds nothing."""

    def __init__(self, scales):
        super().__init__()
        self.num_experts = len(scales)
        # one extra scale of 0 for the id `num_experts`
        self.scales = torch.cat([scales, torch.zeros(1)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # each expert is a scalar, so a token's output is the token times the weighted sum of its experts' scales
        factor = (self.scales[top_k_index] * top_k_weights).sum(dim=-1, keepdim=True)
        return hidden_states * factor


def _ep_implementation_matches_one_process(rank, ep_implementation, mode):
    """`ep_implementation` on 2 ranks gives the output and input gradients of all 4 experts on one process."""
    # 4 experts with distinct scales: rank 0 owns experts 0 and 1, rank 1 owns experts 2 and 3
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # 4 tokens, top-2: every token picks one expert on each rank
    hidden_states = torch.arange(32.0).view(4, 8) / 10
    top_k_index = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]])
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
    grad_output = torch.linspace(-1, 1, 32).view(4, 8)

    def forward(experts, hidden_states, top_k_weights):
        # squared, so the backward needs the forward's output (under checkpointing, the recomputed one)
        return experts(hidden_states, top_k_index, top_k_weights).square()

    # expected: all 4 experts on one process
    expected_inputs = [t.clone().requires_grad_() for t in (hidden_states, top_k_weights)]
    expected = forward(_ScaledExperts(scales), *expected_inputs)
    expected.backward(grad_output)

    # actual: the same batch through this rank's 2 experts, behind `ep_implementation`. Both ranks hold the batch
    # (the EP group is the TP group), so every implementation applies.
    mesh = init_device_mesh("cpu", (2,))
    experts = EpDispatchExpertsParallel().install_forward(
        _ScaledExperts(scales[2 * rank : 2 * rank + 2]), mesh, tp_mesh=mesh, ep_implementation=ep_implementation
    )
    inputs = [t.clone().requires_grad_() for t in (hidden_states, top_k_weights)]
    if mode == "inference":
        with torch.no_grad():
            output = forward(experts, *inputs)
    elif mode == "train":
        output = forward(experts, *inputs)
    elif mode == "checkpoint":
        output = checkpoint(forward, experts, *inputs, use_reentrant=False)
    elif mode == "reentrant_checkpoint":
        output = checkpoint(forward, experts, *inputs, use_reentrant=True)

    torch.testing.assert_close(output, expected)
    if mode != "inference":
        output.backward(grad_output)
        torch.testing.assert_close([t.grad for t in inputs], [t.grad for t in expected_inputs])
