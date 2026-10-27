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
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from transformers.distributed import expert_parallel
from transformers.distributed.expert_parallel import (
    AllReduceEP,
    AllToAllEP,
    AutoEP,
    DeepEP,
    EpDispatchExpertsParallel,
    EPImplementation,
    EPImplementationInterface,
)
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES
from transformers.integrations import deepep
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
        """`install_forward` runs the implementation registered under `ep_implementation` and refuses an unknown name."""
        with self.assertRaisesRegex(ValueError, "names no registered implementation"):
            EpDispatchExpertsParallel().install_forward(_DummyExperts(), None, ep_implementation="fast")

        class Custom(EPImplementation):
            def __init__(self, experts, ep_mesh, tp_mesh=None):
                self.experts = experts

            def __call__(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states + 1

        with patch.dict(EPImplementationInterface._global_mapping, {"custom": Custom}):
            experts = EpDispatchExpertsParallel().install_forward(_DummyExperts(), None, ep_implementation="custom")
        torch.testing.assert_close(experts(torch.zeros(2, 4), None, torch.ones(2, 1)), torch.ones(2, 4))

    def test_passthrough_hands_experts_the_shared_batch(self):
        """`"passthrough"` hands experts the shared batch, global expert ids, unmasked weights and the EP group."""
        _init_distributed(tp=2, backend="gloo")(_passthrough_hands_experts_the_shared_batch)()

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
            ("nccl", True, DeepEP),
            ("nccl", False, None),
            ("gloo", True, None),
        ):
            with (
                self.subTest(backend=backend, loadable=loadable),
                patch.object(expert_parallel, "get_backend", return_value=backend),
                patch.object(deepep, "is_deepep_loadable", side_effect=lambda raise_error=False: loadable),
                patch.object(deepep, "lazy_load_kernel", return_value=SimpleNamespace(EPBuffer=object)),
            ):
                auto = AutoEP(experts, mesh)
                self.assertIsInstance(auto.all_to_all, AllToAllEP)
                self.assertIs(type(auto.deepep) if expected else auto.deepep, expected)
        with _deepep_environment(nccl_version=(2, 30, 7)), self.assertRaisesRegex(ImportError, "NCCL ≥ 2.32.3"):
            DeepEP(experts, mesh)

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

    # an implementation that needs a shared batch refuses an EP group whose ranks hold different batches
    try:
        AllReduceEP(_ScaledExperts(_SCALES[:2]), init_device_mesh("cpu", (2,)))
    except ValueError as error:
        assert "same batch" in str(error), error
    else:
        raise AssertionError("AllReduceEP accepted an EP group without a shared batch")

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


class _ExchangingExperts(torch.nn.Module):
    """Experts that exchange tokens themselves: record what they get and return a per-rank output."""

    num_experts = 2

    def forward(self, hidden_states, top_k_index, top_k_weights, process_group):
        self.received = (hidden_states, top_k_index, top_k_weights, process_group)
        return hidden_states * (torch.distributed.get_rank() + 1)


def _passthrough_hands_experts_the_shared_batch(rank):
    mesh = init_device_mesh("cpu", (2,))
    experts = EpDispatchExpertsParallel().install_forward(
        _ExchangingExperts(), mesh, tp_mesh=mesh, ep_implementation="passthrough"
    )
    output = experts(_HIDDEN_STATES, _TOP_K_INDEX, _TOP_K_WEIGHTS)
    received_hidden, received_index, received_weights, process_group = experts.received
    torch.testing.assert_close(received_hidden, _HIDDEN_STATES)
    torch.testing.assert_close(received_index, _TOP_K_INDEX)
    torch.testing.assert_close(received_weights, _TOP_K_WEIGHTS)
    assert torch.distributed.get_process_group_ranks(process_group) == [0, 1]
    torch.testing.assert_close(output, _HIDDEN_STATES * (rank + 1))


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
    deepep_dispatch = DeepEP.dispatch
    # the EP group shares one batch (each TP rank sends half of it), or each rank holds its own (rows 2 * rank, +1);
    # in training, "auto" dispatches with DeepEP
    for shared, ep_implementation in ((True, "deepep"), (False, "deepep"), (True, "auto"), (False, "auto")):
        tp_mesh, rows = (mesh, slice(None)) if shared else (None, slice(2 * rank, 2 * rank + 2))
        dispatched.clear()
        with patch.object(DeepEP, "dispatch", lambda *args: dispatched.append(True) or deepep_dispatch(*args)):
            module = EpDispatchExpertsParallel().install_forward(
                _ScaledExperts(scales[2 * rank : 2 * rank + 2]),
                mesh,
                tp_mesh=tp_mesh,
                ep_implementation=ep_implementation,
            )
            h, w = hidden_states[rows].clone().requires_grad_(), top_k_weights[rows].clone().requires_grad_()
            output = module(h, top_k_index[rows], w)
            output.backward(grad_output[rows])
        assert dispatched, (shared, ep_implementation)
        # tokens travel in bfloat16
        torch.testing.assert_close(output, expected[rows].detach(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(h.grad, h_ref.grad[rows], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w.grad, w_ref.grad[rows], rtol=1e-2, atol=1e-2)
    deepep.destroy_deepep_buffers()


def _deepep_captures_and_compiles(rank):
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (2,))
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    module = EpDispatchExpertsParallel().install_forward(
        _ScaledExperts(scales[2 * rank : 2 * rank + 2]), mesh, ep_implementation="deepep"
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
