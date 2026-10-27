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
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from transformers.distributed import DistributedConfig, expert_parallel
from transformers.distributed.expert_parallel import (
    AllReduceEP,
    AllToAllEP,
    EpDispatchExpertsParallel,
    EPImplementationInterface,
    _resolve_ep_implementation,
)
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES
from transformers.testing_utils import TestCasePlus, is_expert_parallel_test

from ..test_tensor_parallel_mixin import _init_distributed


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
        _init_distributed(tp=2, backend="gloo")(_dispatch_hands_experts_the_received_rows)()

    def test_ep_implementations_match_one_process(self):
        """Every built-in implementation of `ep_dispatch_experts` gives the single-process output and input gradients,
        and `"auto"` runs the all-reduce on a shared batch at inference and the all-to-all in training."""
        _init_distributed(tp=2, backend="gloo")(_ep_implementations_match_one_process)()

    @parameterized.expand([(False,), (True,)])
    def test_ep_auto_implementation_with_gradient_checkpointing(self, use_reentrant):
        """Gradient checkpointing keeps the `"auto"` choice and the gradients correct in both checkpoint modes."""
        _init_distributed(tp=2, backend="gloo")(_ep_gradient_checkpointing_matches_reference)(use_reentrant)

    def test_ep_implementation_names_a_registered_implementation(self):
        """`ep_implementation` defaults to `"auto"`, names any implementation registered in `ALL_EP_IMPLEMENTATIONS`,
        and refuses `"all_reduce"` where the EP group does not share one batch."""
        self.assertEqual(_resolve_ep_implementation(DistributedConfig(tp_size=2, ep_size=2)), "auto")
        self.assertEqual(_resolve_ep_implementation(DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2)), "auto")
        for name in ("all_reduce", "all_to_all"):
            config = DistributedConfig(tp_size=2, ep_size=2, ep_implementation=name)
            self.assertEqual(_resolve_ep_implementation(config), name)
        with self.assertRaisesRegex(ValueError, "requires `ep_size == tp_size`"):
            _resolve_ep_implementation(
                DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2, ep_implementation="all_reduce")
            )
        with self.assertRaisesRegex(ValueError, "names no registered implementation"):
            _resolve_ep_implementation(DistributedConfig(tp_size=2, ep_size=2, ep_implementation="fast"))
        with patch.dict(EPImplementationInterface._global_mapping, {"custom": AllToAllEP}):
            config = DistributedConfig(tp_size=2, ep_size=2, ep_implementation="custom")
            self.assertEqual(_resolve_ep_implementation(config), "custom")


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


class _ScaledExperts(torch.nn.Module):
    """Experts that scale their tokens by a per-expert factor, masking the out-of-range id `num_experts`."""

    def __init__(self, scales):
        super().__init__()
        self.scales = scales
        self.num_experts = len(scales)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        on_rank = top_k_index < self.num_experts
        scale = self.scales[top_k_index.clamp(max=self.num_experts - 1)] * on_rank
        return (hidden_states.unsqueeze(1) * (scale * top_k_weights).unsqueeze(-1)).sum(dim=1)


_SCALES = torch.tensor([1.0, 2.0, 3.0, 4.0])
_HIDDEN_STATES = torch.arange(32.0).view(4, 8) / 10
_TOP_K_INDEX = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]])
_TOP_K_WEIGHTS = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
_GRAD_OUTPUT = torch.linspace(-1, 1, 32).view(4, 8)


def _ep_experts(rank, ep_implementation):
    """This rank's half of the experts, on a 2-rank EP group that shares one batch (tp == ep)."""
    mesh = init_device_mesh("cpu", (2,))
    experts = _ScaledExperts(_SCALES[2 * rank : 2 * rank + 2])
    return EpDispatchExpertsParallel().install_forward(
        experts, mesh, tp_mesh=mesh, ep_implementation=ep_implementation
    )


def _ep_implementations_match_one_process(rank):
    # the same experts on one process: the expected output and input gradients
    h_ref, w_ref = _HIDDEN_STATES.clone().requires_grad_(), _TOP_K_WEIGHTS.clone().requires_grad_()
    expected = _ScaledExperts(_SCALES)(h_ref, _TOP_K_INDEX, w_ref)
    expected.backward(_GRAD_OUTPUT)

    exchanged = []
    all_to_all_dispatch = AllToAllEP.dispatch
    for ep_implementation, training, exchanges in (
        ("auto", True, True),
        ("auto", False, False),
        ("all_reduce", True, False),
        ("all_to_all", False, True),
    ):
        exchanged.clear()
        with patch.object(AllToAllEP, "dispatch", lambda *args: exchanged.append(True) or all_to_all_dispatch(*args)):
            module = _ep_experts(rank, ep_implementation)
            h, w = _HIDDEN_STATES.clone().requires_grad_(), _TOP_K_WEIGHTS.clone().requires_grad_()
            with torch.set_grad_enabled(training):
                output = module(h, _TOP_K_INDEX, w)
            if training:
                output.backward(_GRAD_OUTPUT)
        assert bool(exchanged) == exchanges, (ep_implementation, training, exchanged)
        torch.testing.assert_close(output, expected.detach())
        if training:
            torch.testing.assert_close(h.grad, h_ref.grad)
            torch.testing.assert_close(w.grad, w_ref.grad)


def _ep_gradient_checkpointing_matches_reference(rank, use_reentrant):
    reference = _ep_experts(rank, "auto")
    h_ref, w_ref = _HIDDEN_STATES.clone().requires_grad_(), _TOP_K_WEIGHTS.clone().requires_grad_()
    reference_output = reference(h_ref, _TOP_K_INDEX, w_ref).square()
    reference_output.backward(_GRAD_OUTPUT)

    all_to_all_calls, all_reduce_calls = [], []
    all_to_all_dispatch, all_reduce_dispatch = AllToAllEP.dispatch, AllReduceEP.dispatch

    def record_all_to_all(self, *args):
        all_to_all_calls.append(True)
        return all_to_all_dispatch(self, *args)

    def record_all_reduce(self, *args):
        all_reduce_calls.append(True)
        return all_reduce_dispatch(self, *args)

    with (
        patch.object(AllToAllEP, "dispatch", record_all_to_all),
        patch.object(AllReduceEP, "dispatch", record_all_reduce),
    ):
        module = _ep_experts(rank, "auto")
        h, w = _HIDDEN_STATES.clone().requires_grad_(), _TOP_K_WEIGHTS.clone().requires_grad_()
        output = checkpoint(
            lambda inputs, indices, weights: module(inputs, indices, weights).square(),
            h,
            _TOP_K_INDEX,
            w,
            use_reentrant=use_reentrant,
        )
        output.backward(_GRAD_OUTPUT)

    # Reentrant checkpointing runs the forward without gradients (all-reduce) and recomputes it with them
    # (all-to-all); the non-reentrant mode keeps gradients on throughout.
    if use_reentrant:
        assert len(all_to_all_calls) == 1
        assert len(all_reduce_calls) == 1
    else:
        assert len(all_to_all_calls) >= 1
        assert not all_reduce_calls
    torch.testing.assert_close(output, reference_output.detach())
    torch.testing.assert_close(h.grad, h_ref.grad)
    torch.testing.assert_close(w.grad, w_ref.grad)
