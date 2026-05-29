# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Tests for the decomposed MoE tensor / expert parallelism plan.

Weight sharding for MoE experts is declared per-parameter in the config plan and
applied by the param-level pass of ``apply_tensor_parallel``:

  * ``grouped_gemm``          — EP, ``Shard(0)`` on the expert dim
  * ``moe_gate_up_colwise``   — TP, ``_StridedShard(dim=-2)`` (Qwen3/Mixtral layout)
  * ``moe_gate_up_colwise_alt`` — TP, ``_StridedShard(dim=-1)`` (GPT-OSS/Llama4 layout)
  * ``moe_down_rowwise``      — TP, ``Shard(-1)`` on down_proj's input dim

The experts *module* keeps a forward-only ``moe_experts_allreduce`` hook (no baked
``shard_plan``), and the router (gate) uses ``ep_router`` to slice router outputs to
local experts under EP.
"""

import os
import socket
import unittest

from transformers import set_seed
from transformers.testing_utils import is_tensor_parallel_test, require_torch
from transformers.utils import is_torch_available, is_torch_greater_or_equal


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.placement_types import _StridedShard
    from torch.multiprocessing.spawn import ProcessRaisedException

    from transformers.distributed.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        PARAM_ONLY_STYLES,
        CustomParallelStyle,
        _get_parameter_tp_plan,
        apply_tensor_parallel,
    )


# =============================================================================
# Tiny inline MoE model (Qwen3-style expert layout)
# =============================================================================


class TinyMoEExperts(nn.Module):
    """Qwen3-style: gate_up [E, 2*inter, hidden], down [E, hidden, inter]."""

    def __init__(self, num_experts=4, hidden=8, intermediate=16):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, intermediate))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class TinyMoERouter(nn.Module):
    def __init__(self, num_experts=4, hidden=8):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden))

    def forward(self, hidden_states):
        logits = hidden_states @ self.weight.T
        scores, indices = torch.topk(logits, k=2, dim=-1)
        return logits, scores, indices


class TinyMoEBlock(nn.Module):
    def __init__(self, num_experts=4, hidden=8, intermediate=16):
        super().__init__()
        self.gate = TinyMoERouter(num_experts, hidden)
        self.experts = TinyMoEExperts(num_experts, hidden, intermediate)


class _TinyConfig:
    """Minimal stand-in for a model config so ``apply_tensor_parallel`` can introspect it."""

    distributed_config = None
    base_model_sp_plan = None
    tie_word_embeddings = False


class TinyMoEModel(nn.Module):
    def __init__(self, num_experts=4, hidden=8, intermediate=16):
        super().__init__()
        self.layers = nn.ModuleList([TinyMoEBlock(num_experts, hidden, intermediate)])
        self.config = _TinyConfig()


EP_PLAN = {
    "layers.*.gate": "ep_router",
    "layers.*.experts.gate_up_proj": "grouped_gemm",
    "layers.*.experts.down_proj": "grouped_gemm",
    "layers.*.experts": "moe_experts_allreduce",
}

TP_PLAN_DECOMPOSED = {
    "layers.*.experts.gate_up_proj": "moe_gate_up_colwise",
    "layers.*.experts.down_proj": "moe_down_rowwise",
    "layers.*.experts": "moe_experts_allreduce",
}


# =============================================================================
# Distributed harness (CPU + gloo, mirrors tests/test_tensor_parallel_mixin.py)
# =============================================================================


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _global_wrapper(rank, func, tp, port, backend, func_args, func_kwargs):
    os.environ["WORLD_SIZE"] = str(tp)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend=backend, rank=rank, world_size=tp)
    try:
        func(rank, *func_args, **func_kwargs)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _init_distributed(tp: int, max_retries: int = 5, backend: str = "gloo"):
    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                port = _find_free_port()
                spawn_args = (func, tp, port, backend, args, kwargs)
                try:
                    mp.spawn(_global_wrapper, args=spawn_args, nprocs=tp)
                    return
                except ProcessRaisedException as e:
                    if "EADDRINUSE" in str(e) and attempt < max_retries - 1:
                        continue
                    raise

        return wrapper

    return _init_distributed_inner


# --- per-process bodies (must be module-level so mp.spawn can pickle them) -----


def _ep_param_shard_impl(rank, world_size):
    set_seed(0)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = TinyMoEModel(num_experts=4, hidden=8, intermediate=16)
    apply_tensor_parallel(model, mesh, dict(EP_PLAN))

    experts = model.layers[0].experts
    gate_up, down = experts.gate_up_proj, experts.down_proj

    assert isinstance(gate_up, DTensor), f"gate_up_proj not sharded: {type(gate_up)}"
    assert isinstance(down, DTensor), f"down_proj not sharded: {type(down)}"
    # EP shards the expert dimension (dim 0) for both projections.
    assert gate_up.placements[0] == Shard(0), gate_up.placements
    assert down.placements[0] == Shard(0), down.placements
    assert gate_up.to_local().shape[0] == 4 // world_size, gate_up.to_local().shape
    assert down.to_local().shape[0] == 4 // world_size, down.to_local().shape
    # grouped_gemm updates num_experts to the per-rank local count.
    assert experts.num_experts == 4 // world_size, experts.num_experts


def _tp_param_shard_impl(rank, world_size):
    set_seed(0)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = TinyMoEModel(num_experts=4, hidden=8, intermediate=16)
    apply_tensor_parallel(model, mesh, dict(TP_PLAN_DECOMPOSED))

    experts = model.layers[0].experts
    gate_up, down = experts.gate_up_proj, experts.down_proj

    assert isinstance(gate_up, DTensor), f"gate_up_proj not sharded: {type(gate_up)}"
    assert isinstance(down, DTensor), f"down_proj not sharded: {type(down)}"
    # gate_up is packed (gate||up) → interleaved strided shard on the packed (-2) dim.
    assert isinstance(gate_up.placements[0], _StridedShard), gate_up.placements
    assert gate_up.placements[0].split_factor == 2, gate_up.placements
    # gate_up [E=4, 2*inter=32, hidden=8] → dim -2 (=32) sharded.
    assert gate_up.to_local().shape[1] == 32 // world_size, gate_up.to_local().shape
    # down is plain rowwise on the last (input) dim — NOT strided.
    assert not isinstance(down.placements[0], _StridedShard), down.placements
    assert isinstance(down.placements[0], Shard), down.placements
    # down [E=4, hidden=8, inter=16] → dim -1 (=16) sharded.
    assert down.to_local().shape[-1] == 16 // world_size, down.to_local().shape
    # TP does not shard the expert dim, so num_experts is unchanged.
    assert experts.num_experts == 4, experts.num_experts


def _ep_router_impl(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    style = ALL_PARALLEL_STYLES["ep_router"]
    module = TinyMoERouter(num_experts=4, hidden=8)

    # 4 tokens, top_k=2, 4 global experts, ep_size=2 → 2 local experts/rank.
    logits = torch.zeros(4, 4)
    scores = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    indices = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]])

    _, out_scores, out_indices = style.transform_output_post_forward(module, (logits, scores, indices), mesh)

    # Sentinel for dropped (non-local) slots is num_local_experts = 4 // 2 = 2.
    expected_indices = {
        0: torch.tensor([[0, 2], [1, 2], [2, 0], [2, 1]]),  # rank 0 owns experts {0, 1}
        1: torch.tensor([[2, 0], [2, 1], [0, 2], [1, 2]]),  # rank 1 owns experts {2, 3}
    }
    expected_scores = {
        0: torch.tensor([[0.5, 0.0], [0.6, 0.0], [0.0, 0.3], [0.0, 0.8]]),
        1: torch.tensor([[0.0, 0.5], [0.0, 0.4], [0.7, 0.0], [0.2, 0.0]]),
    }
    assert torch.equal(out_indices, expected_indices[rank]), (rank, out_indices)
    assert torch.allclose(out_scores, expected_scores[rank]), (rank, out_scores)


# =============================================================================
# Layer 1 — Plan resolution (no distributed)
# =============================================================================


@require_torch
class TestMoEPlanResolution(unittest.TestCase):
    def test_ep_plan_resolves_param_keys(self):
        # grouped_gemm is declared and resolved at parameter granularity.
        self.assertEqual(
            _get_parameter_tp_plan("layers.0.experts.gate_up_proj", EP_PLAN, is_weight=True), "grouped_gemm"
        )
        self.assertEqual(_get_parameter_tp_plan("layers.3.experts.down_proj", EP_PLAN, is_weight=True), "grouped_gemm")

    def test_module_lookup_resolves_experts_to_forward_only_style(self):
        # The module pass (is_weight=False) resolves the experts *module* to its own
        # forward-only rule, never to a child parameter's grouped_gemm rule.
        self.assertEqual(_get_parameter_tp_plan("layers.0.experts", EP_PLAN, is_weight=False), "moe_experts_allreduce")
        # The gate module resolves to its router rule.
        self.assertEqual(_get_parameter_tp_plan("layers.0.gate", EP_PLAN, is_weight=False), "ep_router")

    def test_is_weight_parent_fallback_only_for_weights(self):
        plan = {"layers.*.self_attn.q_proj": "colwise"}
        # A weight lookup falls back to the parent module rule ...
        self.assertEqual(_get_parameter_tp_plan("layers.0.self_attn.q_proj.weight", plan, is_weight=True), "colwise")
        # ... but a module lookup (is_weight=False) does not.
        self.assertIsNone(_get_parameter_tp_plan("layers.0.self_attn.q_proj.weight", plan, is_weight=False))

    def test_tp_decomposed_plan_resolves_per_param(self):
        self.assertEqual(
            _get_parameter_tp_plan("layers.0.experts.gate_up_proj", TP_PLAN_DECOMPOSED, is_weight=True),
            "moe_gate_up_colwise",
        )
        self.assertEqual(
            _get_parameter_tp_plan("layers.0.experts.down_proj", TP_PLAN_DECOMPOSED, is_weight=True),
            "moe_down_rowwise",
        )


# =============================================================================
# Layer 2 — Placement expectations (documentation tests)
# =============================================================================


@require_torch
@unittest.skipUnless(is_torch_greater_or_equal("2.5"), "TP styles require torch >= 2.5")
class TestMoEPlacementExpectations(unittest.TestCase):
    def test_grouped_gemm_is_shard_zero_on_expert_dim(self):
        style = ALL_PARALLEL_STYLES["grouped_gemm"]
        self.assertEqual(style.placement, Shard(0))
        self.assertTrue(style.shards_expert_dim)

    def test_tp_gate_up_is_strided_shard_on_packed_dim(self):
        style = ALL_PARALLEL_STYLES["moe_gate_up_colwise"]
        self.assertIsInstance(style.placement, _StridedShard)
        self.assertEqual(style.placement.dim, -2)
        self.assertEqual(style.placement.split_factor, 2)
        self.assertFalse(style.shards_expert_dim)

        alt = ALL_PARALLEL_STYLES["moe_gate_up_colwise_alt"]
        self.assertIsInstance(alt.placement, _StridedShard)
        self.assertEqual(alt.placement.dim, -1)
        self.assertEqual(alt.placement.split_factor, 2)

    def test_tp_down_is_rowwise_on_last_dim(self):
        style = ALL_PARALLEL_STYLES["moe_down_rowwise"]
        self.assertEqual(style.placement, Shard(-1))
        self.assertNotIsInstance(style.placement, _StridedShard)
        self.assertFalse(style.shards_expert_dim)


# =============================================================================
# Layer 3 — Distributed integration (multi-process, mp.spawn + gloo)
# =============================================================================


@require_torch
class TestMoEDistributedApply(unittest.TestCase):
    world_size = 2

    def _skip_if_unsupported(self):
        if not is_torch_greater_or_equal("2.5"):
            self.skipTest("MoE TP/EP styles require torch >= 2.5")
        if torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available()):
            self.skipTest("These tests are CPU-only (gloo) and should not run on GPU/XPU")
        if (os.cpu_count() or 1) < self.world_size:
            self.skipTest(f"Requires at least {self.world_size} CPUs")

    @is_tensor_parallel_test
    def test_ep_plan_shards_expert_dim_per_param(self):
        self._skip_if_unsupported()
        _init_distributed(tp=self.world_size)(_ep_param_shard_impl)(self.world_size)

    @is_tensor_parallel_test
    def test_tp_decomposed_plan_shards_per_param(self):
        self._skip_if_unsupported()
        _init_distributed(tp=self.world_size)(_tp_param_shard_impl)(self.world_size)

    @is_tensor_parallel_test
    def test_ep_router_slices_scores_and_remaps_indices(self):
        self._skip_if_unsupported()
        _init_distributed(tp=self.world_size)(_ep_router_impl)(self.world_size)


# =============================================================================
# Layer 4 — Registry regression guard
# =============================================================================


@require_torch
@unittest.skipUnless(is_torch_greater_or_equal("2.5"), "TP styles require torch >= 2.5")
class TestMoERegistryShape(unittest.TestCase):
    def test_moe_experts_allreduce_has_no_baked_shard_plan(self):
        # Guards against re-introducing the bundled shard_plan (sharding now lives in configs).
        # MoEExpertsParallel is forward-comm only: no baked shard plan and no shard_parameters
        # override (it inherits the base no-op).
        style = ALL_PARALLEL_STYLES["moe_experts_allreduce"]
        self.assertFalse(hasattr(style, "_moe_shard_plan"))
        self.assertIs(type(style).shard_parameters, CustomParallelStyle.shard_parameters)

    def test_param_only_styles_registered(self):
        for name in PARAM_ONLY_STYLES:
            self.assertIn(name, ALL_PARALLEL_STYLES)

    def test_ep_router_registered(self):
        self.assertIn("ep_router", ALL_PARALLEL_STYLES)


if __name__ == "__main__":
    unittest.main()
