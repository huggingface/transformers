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
import contextlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh

from transformers import FineGrainedFP8Config
from transformers.distributed import DistributedConfig, expert_parallel
from transformers.distributed.expert_parallel import (
    AllToAllStrategy,
    AutoStrategy,
    DeepEPStrategy,
    EpStrategyInterface,
    _resolve_ep_strategy,
)
from transformers.distributed.tensor_parallel import EpDispatchExpertsParallel
from transformers.integrations import deepep
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
        config = DistributedConfig(tp_size=2, ep_size=2, ep_implementation=strategy)
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


def _megamoe_hands_each_rank_a_slice(rank):
    mesh = init_device_mesh("cpu", (2,))
    config = DistributedConfig(tp_size=2, ep_size=2, ep_implementation="deepgemm_megamoe")
    layers = [
        EpDispatchExpertsParallel().install_forward(_MegaMoEExperts(), mesh, tp_mesh=mesh, distributed_config=config)
        for _ in range(2)
    ]
    # 3 tokens on 2 ranks: 2 rows each, the last one padded with a skipped slot
    hidden_states = torch.arange(24.0).view(3, 8) / 10
    top_k_index = torch.tensor([[0, 3], [1, 2], [2, 0]])
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    padded_hidden = torch.cat([hidden_states, torch.zeros(1, 8)])
    padded_index = torch.cat([top_k_index, torch.full((1, 2), -1)])
    padded_weights = torch.cat([top_k_weights, torch.zeros(1, 2)])
    rows = slice(2 * rank, 2 * rank + 2)
    expected = torch.cat([padded_hidden[:2] * 1, padded_hidden[2:] * 2])[:3]
    with torch.no_grad():
        for experts in layers:
            output = experts(hidden_states, top_k_index, top_k_weights)
            received_hidden, received_index, received_weights, process_group = experts.received
            # this rank's slice with global expert ids and unmasked weights, and the gathered outputs of every slice
            torch.testing.assert_close(received_hidden, padded_hidden[rows])
            torch.testing.assert_close(received_index, padded_index[rows])
            torch.testing.assert_close(received_weights, padded_weights[rows])
            assert torch.distributed.get_process_group_ranks(process_group) == [0, 1]
            torch.testing.assert_close(output, expected)
    # the layers of one EP group share the kernel's buffers
    assert layers[0].buffer is layers[1].buffer


def _deepep_matches_one_process(rank):
    torch.cuda.set_device(rank)
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    # DeepEP needs a hidden size that is a multiple of 256
    hidden_states = torch.arange(1024.0, device="cuda").view(4, 256) / 1000
    top_k_index = torch.tensor([[0, 3], [1, 2], [2, 0], [3, 1]], device="cuda")
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]], device="cuda")
    grad_output = torch.linspace(-1, 1, 1024, device="cuda").view(4, 256)
    h_ref, w_ref = hidden_states.clone().requires_grad_(), top_k_weights.clone().requires_grad_()
    expected = _ScaledExperts(scales)(h_ref, top_k_index, w_ref)
    expected.backward(grad_output)

    mesh = init_device_mesh("cuda", (2,))
    dispatched = []
    deepep_dispatch = DeepEPStrategy.dispatch
    # the EP group shares one batch (each TP rank sends half of it), or each rank holds its own (rows 2 * rank, +1);
    # in training, "auto" dispatches with DeepEP
    for shared, strategy in ((True, "deepep"), (False, "deepep"), (True, "auto"), (False, "auto")):
        tp_size, fsdp_size = (2, 1) if shared else (1, 2)
        config = DistributedConfig(tp_size=tp_size, fsdp_size=fsdp_size, ep_size=2, ep_implementation=strategy)
        tp_mesh, rows = (mesh, slice(None)) if shared else (None, slice(2 * rank, 2 * rank + 2))
        dispatched.clear()
        with patch.object(DeepEPStrategy, "dispatch", lambda *args: dispatched.append(True) or deepep_dispatch(*args)):
            module = EpDispatchExpertsParallel().install_forward(
                _ScaledExperts(scales[2 * rank : 2 * rank + 2]), mesh, tp_mesh=tp_mesh, distributed_config=config
            )
            h, w = hidden_states[rows].clone().requires_grad_(), top_k_weights[rows].clone().requires_grad_()
            output = module(h, top_k_index[rows], w)
            output.backward(grad_output[rows])
        assert dispatched, (shared, strategy)
        # tokens travel in bfloat16
        torch.testing.assert_close(output, expected[rows].detach(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(h.grad, h_ref.grad[rows], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w.grad, w_ref.grad[rows], rtol=1e-2, atol=1e-2)
    deepep.destroy_deepep_buffers()


def _deepep_captures_and_compiles(rank):
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (2,))
    config = DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2, ep_implementation="deepep")
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    module = EpDispatchExpertsParallel().install_forward(
        _ScaledExperts(scales[2 * rank : 2 * rank + 2]), mesh, distributed_config=config
    )
    generator = torch.Generator(device="cuda").manual_seed(rank)

    def routes():
        weights, index = torch.rand(3, 4, device="cuda", generator=generator).softmax(-1).topk(2)
        return torch.randn(3, 256, device="cuda", generator=generator), index, weights

    with torch.no_grad():
        inputs = routes()
        # the usual warm-up: an eager forward at the captured batch size allocates the buffer
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(*inputs)
        for _ in range(2):
            for static, new in zip(inputs, routes()):
                static.copy_(new)
            graph.replay()
            torch.testing.assert_close(captured, module(*inputs))

        compiled = torch.compile(module, fullgraph=True)
        inputs = routes()
        torch.testing.assert_close(compiled(*inputs), module(*inputs))
    deepep.destroy_deepep_buffers()


@contextlib.contextmanager
def _deepep_environment(
    kernels_available=True,
    cuda_available=True,
    capability=(9, 0),
    nvcc_present=True,
    nvcc_version=(13, 1),
    nccl_libraries=("libnccl.so.2",),
    nccl_version=(2, 32, 3),
    kernel=SimpleNamespace(EPBuffer=object),
):
    """A faked environment for `is_deepep_loadable`, a real toolkit directory holding `bin/nvcc` or not."""
    deepep.is_deepep_loadable.cache_clear()
    with tempfile.TemporaryDirectory() as cuda_home, contextlib.ExitStack() as stack:
        if nvcc_present:
            os.makedirs(os.path.join(cuda_home, "bin"))
            open(os.path.join(cuda_home, "bin", "nvcc"), "w", encoding="utf-8").close()
        for target, name, value in (
            (deepep, "is_kernels_available", kernels_available),
            (torch.cuda, "is_available", cuda_available),
            (torch.cuda, "get_device_capability", capability),
            (deepep, "_get_cuda_home", cuda_home),
            (deepep, "_get_nvcc_version", nvcc_version),
            (deepep, "_get_loaded_nccl_libraries", list(nccl_libraries)),
            (deepep, "_get_nccl_runtime_version", nccl_version),
            (deepep, "lazy_load_kernel", kernel),
        ):
            stack.enter_context(patch.object(target, name, return_value=value))
        yield
    deepep.is_deepep_loadable.cache_clear()


@is_tensor_parallel_test
class TestEpStrategies(TestCasePlus):
    def test_ep_strategies_match_one_process(self):
        """Every built-in strategy of `ep_dispatch_experts` gives the single-process output and input gradients, and
        `"auto"` runs the all-reduce on a shared batch at inference and the all-to-all in training."""
        _init_distributed(tp=2, backend="gloo")(_ep_strategies_match_one_process)()

    def test_ep_strategy_names_a_registered_strategy(self):
        """`ep_implementation` defaults to `"auto"`, names any strategy registered in `ALL_EP_STRATEGIES`, and refuses `"all_reduce"`
        where the EP group does not share one batch."""
        shared, split = DistributedConfig(tp_size=2, ep_size=2), DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2)
        self.assertEqual(_resolve_ep_strategy(shared), "auto")
        self.assertEqual(_resolve_ep_strategy(split), "auto")
        self.assertEqual(
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_implementation="all_reduce")), "all_reduce"
        )
        self.assertEqual(
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_implementation="all_to_all")), "all_to_all"
        )
        with self.assertRaisesRegex(ValueError, "requires `ep_size == tp_size`"):
            _resolve_ep_strategy(DistributedConfig(tp_size=1, fsdp_size=2, ep_size=2, ep_implementation="all_reduce"))
        with self.assertRaisesRegex(ValueError, "names no registered strategy"):
            _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_implementation="fast"))
        with patch.dict(EpStrategyInterface._global_mapping, {"custom": AllToAllStrategy}):
            self.assertEqual(
                _resolve_ep_strategy(DistributedConfig(tp_size=2, ep_size=2, ep_implementation="custom")), "custom"
            )

    def test_megamoe_hands_each_rank_a_slice(self):
        """`"deepgemm_megamoe"` hands each rank's kernel an equal slice of the shared batch, with global expert ids,
        unmasked weights and the EP group, gathers the slices back, and shares the kernel's buffers between layers."""
        _init_distributed(tp=2, backend="gloo")(_megamoe_hands_each_rank_a_slice)()

    def test_megamoe_selects_its_strategy(self):
        """Loading with DeepGEMM Mega MoE experts runs `ep_dispatch_experts` with `"deepgemm_megamoe"`, and refuses another
        strategy."""
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
        for ep_implementation in (None, "auto", "deepgemm_megamoe"):
            config = SimpleNamespace(
                _experts_implementation="deepgemm_megamoe",
                distributed_config=DistributedConfig(tp_size=2, ep_size=2, ep_implementation=ep_implementation),
            )
            self.assertEqual(quantizer.update_ep_plan(config).distributed_config.ep_implementation, "deepgemm_megamoe")
        config = SimpleNamespace(
            _experts_implementation="deepgemm_megamoe",
            distributed_config=DistributedConfig(tp_size=2, ep_size=2, ep_implementation="all_to_all"),
        )
        with self.assertRaisesRegex(ValueError, "deepgemm_megamoe"):
            quantizer.update_ep_plan(config)
        config = SimpleNamespace(_experts_implementation="grouped_mm", distributed_config=DistributedConfig())
        self.assertIsNone(quantizer.update_ep_plan(config).distributed_config.ep_implementation)

    def test_is_deepep_loadable(self):
        """`is_deepep_loadable` is `True` where DeepEP runs, and otherwise `False`, or with `raise_error=True` the
        `ImportError` naming the missing requirement."""
        with _deepep_environment():
            self.assertTrue(deepep.is_deepep_loadable())
        for env, reason in (
            ({"kernels_available": False}, "kernel unavailable"),
            ({"cuda_available": False}, "requires CUDA"),
            ({"capability": (8, 0)}, "compute capability 9.0"),
            ({"nvcc_present": False}, "needs a CUDA toolkit"),
            ({"nvcc_version": (12, 9)}, "toolkit ≥ 13.1"),
            ({"nccl_libraries": ()}, "0 are loaded"),
            ({"nccl_libraries": ("libnccl.so.2", "other/libnccl.so.2")}, "2 are loaded"),
            ({"nccl_version": (2, 30, 7)}, "NCCL ≥ 2.32.3, but PyTorch loaded 2.30.7"),
            ({"kernel": None}, "Failed to load"),
        ):
            with self.subTest(env=env), _deepep_environment(**env):
                self.assertFalse(deepep.is_deepep_loadable())
                with self.assertRaisesRegex(ImportError, reason):
                    deepep.is_deepep_loadable(raise_error=True)

    def test_auto_dispatches_with_deepep_where_it_runs(self):
        """`"auto"` dispatches tokens with DeepEP on an NCCL EP group where DeepEP loads, and with all-to-all
        otherwise; `"deepep"` raises the missing requirement."""
        experts, mesh = _ScaledExperts(torch.ones(2)), SimpleNamespace(get_group=lambda: None, size=lambda: 2)
        for backend, loadable, expected in (
            ("nccl", True, DeepEPStrategy),
            ("nccl", False, None),
            ("gloo", True, None),
        ):
            with (
                self.subTest(backend=backend, loadable=loadable),
                patch.object(expert_parallel.dist, "get_backend", return_value=backend),
                patch.object(deepep, "is_deepep_loadable", side_effect=lambda raise_error=False: loadable),
                patch.object(deepep, "lazy_load_kernel", return_value=SimpleNamespace(EPBuffer=object)),
            ):
                auto = AutoStrategy(experts, mesh)
                self.assertIsInstance(auto.all_to_all, AllToAllStrategy)
                self.assertIs(type(auto.deepep) if expected else auto.deepep, expected)
        with _deepep_environment(nccl_version=(2, 30, 7)), self.assertRaisesRegex(ImportError, "NCCL ≥ 2.32.3"):
            DeepEPStrategy(experts, mesh)

    def _skip_unless_deepep_runs(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("DeepEP needs 2 GPUs")
        if not deepep.is_deepep_loadable():
            self.skipTest("DeepEP does not run here (see `is_deepep_loadable(raise_error=True)`)")

    def test_deepep_matches_one_process(self):
        """`"deepep"` gives the single-process output and input gradients, routing weights included, on a batch the
        EP group shares and on batches of each rank's own, and `"auto"` trains with it."""
        self._skip_unless_deepep_runs()
        _init_distributed(tp=2, backend="nccl")(_deepep_matches_one_process)()

    def test_deepep_captures_and_compiles(self):
        """After an eager warm-up, `"deepep"` captures in a CUDA graph whose replays match eager forwards, and compiles
        with `fullgraph=True`: both run DeepEP without its host sync."""
        self._skip_unless_deepep_runs()
        _init_distributed(tp=2, backend="nccl")(_deepep_captures_and_compiles)()
