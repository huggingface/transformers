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
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh

from transformers import FineGrainedFP8Config
from transformers.distributed import DistributedConfig
from transformers.distributed.expert_parallel import (
    AllToAllStrategy,
    EpStrategyInterface,
    _resolve_ep_strategy,
)
from transformers.distributed.tensor_parallel import EpDispatchExpertsParallel
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test

from ..test_tensor_parallel_mixin import _init_distributed


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


def _ep_strategies_match_one_process(rank):
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hidden_states = torch.arange(32.0).view(4, 8) / 10
    top_k_index = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]])
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
    grad_output = torch.linspace(-1, 1, 32).view(4, 8)

    # the same experts on one process: the expected output and input gradients
    h_ref, w_ref = hidden_states.clone().requires_grad_(), top_k_weights.clone().requires_grad_()
    expected = _ScaledExperts(scales)(h_ref, top_k_index, w_ref)
    expected.backward(grad_output)

    # the EP group shares one batch (tp == ep): "auto" runs the all-reduce at inference and the all-to-all in training
    mesh = init_device_mesh("cpu", (2,))
    exchanged = []
    all_to_all_dispatch = AllToAllStrategy.dispatch
    for strategy, training, exchanges in (
        ("auto", True, True),
        ("auto", False, False),
        ("all_reduce", True, False),
        ("all_to_all", False, True),
    ):
        config = DistributedConfig(tp_size=2, ep_size=2, ep_strategy=strategy)
        exchanged.clear()
        with patch.object(
            AllToAllStrategy, "dispatch", lambda *args: exchanged.append(True) or all_to_all_dispatch(*args)
        ):
            module = EpDispatchExpertsParallel().install_forward(
                _ScaledExperts(scales[2 * rank : 2 * rank + 2]), mesh, tp_mesh=mesh, distributed_config=config
            )
            h, w = hidden_states.clone().requires_grad_(), top_k_weights.clone().requires_grad_()
            with torch.set_grad_enabled(training):
                output = module(h, top_k_index, w)
            if training:
                output.backward(grad_output)
        assert bool(exchanged) == exchanges, (strategy, training, exchanged)
        torch.testing.assert_close(output, expected.detach())
        if training:
            torch.testing.assert_close(h.grad, h_ref.grad)
            torch.testing.assert_close(w.grad, w_ref.grad)


class _MegaMoEExperts(torch.nn.Module):
    """Stand-in for Mega MoE experts: record what they get, keep a buffer in `symm_buffers`, return a per-rank output."""

    num_experts = 2

    def forward(self, hidden_states, top_k_index, top_k_weights, process_group, symm_buffers):
        self.received = (hidden_states, top_k_index, top_k_weights, process_group)
        self.buffer = symm_buffers.setdefault("buffer", object())
        return hidden_states * (torch.distributed.get_rank() + 1)


def _megamoe_hands_experts_the_shared_batch(rank):
    hidden_states = torch.arange(32.0).view(4, 8) / 10
    top_k_index = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]])
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
    mesh = init_device_mesh("cpu", (2,))
    config = DistributedConfig(tp_size=2, ep_size=2, ep_strategy="deepgemm_megamoe")
    layers = [
        EpDispatchExpertsParallel().install_forward(_MegaMoEExperts(), mesh, tp_mesh=mesh, distributed_config=config)
        for _ in range(2)
    ]
    for experts in layers:
        output = experts(hidden_states, top_k_index, top_k_weights)
        received_hidden, received_index, received_weights, process_group = experts.received
        # the whole batch with global expert ids and unmasked weights, and the output as the experts returned it
        torch.testing.assert_close(received_hidden, hidden_states)
        torch.testing.assert_close(received_index, top_k_index)
        torch.testing.assert_close(received_weights, top_k_weights)
        assert torch.distributed.get_process_group_ranks(process_group) == [0, 1]
        torch.testing.assert_close(output, hidden_states * (rank + 1))
    # the layers of one EP group share the kernel's buffers
    assert layers[0].buffer is layers[1].buffer


@is_tensor_parallel_test
class TestEpStrategies(TestCasePlus):
    def test_ep_strategies_match_one_process(self):
        """Every built-in strategy of `ep_dispatch_experts` gives the single-process output and input gradients, and
        `"auto"` runs the all-reduce on a shared batch at inference and the all-to-all in training."""
        _init_distributed(tp=2, backend="gloo")(_ep_strategies_match_one_process)()

    def test_ep_strategy_names_a_registered_strategy(self):
        """`ep_strategy` defaults to `"auto"`, names any strategy registered in `ALL_EP_STRATEGIES`, and refuses `"all_reduce"`
        where the EP group does not share one batch."""
        shared, split = DistributedConfig(tp_size=2, ep_size=2), DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2)
        self.assertEqual(_resolve_ep_strategy(shared), "auto")
        self.assertEqual(_resolve_ep_strategy(split), "auto")
        self.assertEqual(
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_strategy="all_reduce")), "all_reduce"
        )
        self.assertEqual(
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_strategy="all_to_all")), "all_to_all"
        )
        with self.assertRaisesRegex(ValueError, "requires `ep_size == tp_size`"):
            _resolve_ep_strategy(DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2, ep_strategy="all_reduce"))
        with self.assertRaisesRegex(ValueError, "names no registered strategy"):
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_strategy="fast"))
        with patch.dict(EpStrategyInterface._global_mapping, {"custom": AllToAllStrategy}):
            self.assertEqual(
                _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_strategy="custom")), "custom"
            )

    def test_megamoe_hands_experts_the_shared_batch(self):
        """`"deepgemm_megamoe"` hands the experts the shared batch, global expert ids, unmasked weights and the EP group,
        returns their output as is, and shares the kernel's buffers between layers."""
        _init_distributed(tp=2, backend="gloo")(_megamoe_hands_experts_the_shared_batch)()

    def test_megamoe_selects_its_strategy(self):
        """Loading with DeepGEMM Mega MoE experts runs `ep_dispatch_experts` with `"deepgemm_megamoe"`, and refuses another
        strategy."""
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
        for ep_strategy in (None, "auto", "deepgemm_megamoe"):
            config = SimpleNamespace(
                _experts_implementation="deepgemm_megamoe",
                distributed_config=DistributedConfig(tp_size=2, ep_size=2, ep_strategy=ep_strategy),
            )
            self.assertEqual(quantizer.update_ep_plan(config).distributed_config.ep_strategy, "deepgemm_megamoe")
        config = SimpleNamespace(
            _experts_implementation="deepgemm_megamoe",
            distributed_config=DistributedConfig(tp_size=2, ep_size=2, ep_strategy="all_to_all"),
        )
        with self.assertRaisesRegex(ValueError, "deepgemm_megamoe"):
            quantizer.update_ep_plan(config)
        config = SimpleNamespace(_experts_implementation="grouped_mm", distributed_config=DistributedConfig())
        self.assertIsNone(quantizer.update_ep_plan(config).distributed_config.ep_strategy)
