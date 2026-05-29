# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""SonicMoE integration: fused MoE using CuteDSL kernels from `kernels-community/sonic-moe`.

Provides:
- `sonicmoe_experts_forward` registered as "sonicmoe" — single-GPU / RouterParallel-style EP
  (sentinel-masked routing on a replicated `(T_global, H)` input, outer all-reduce combine).
- `sonicmoe_ep_experts_forward` registered as "sonicmoe_ep" — DP+EP intra-node Expert
  Parallelism. **Takes `(T_local, H)` per rank and returns `(T_local, H)` per rank** —
  the upstream-native contract. The caller (model) is responsible for token-sharding the
  activations to `T_local` rows *before* the experts block; this impl does not adapt a
  `(T_global, H)` replicated input.

Requirements: CUDA, `kernels`, `nvidia-cutlass-dsl`, has_gate=True.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils import logging
from .hub_kernels import lazy_load_kernel
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)

# Map activation function names from HF config to SonicMoE epilogue names
ACT_MAP = {"silu": "swiglu", "gelu": "geglu", "relu": "reglu"}


@dataclass(frozen=True)
class SonicMoE:
    """Entry points exposed by the `kernels-community/sonic-moe` kernel."""

    activation_type_enum: type
    moe_general_routing_inputs: Callable
    moe_ep_general_routing_forward: Callable
    network_profiler_cls: type


@functools.cache
def _load_sonicmoe_kernel() -> SonicMoE:
    """
    Load sonic-moe once and return its entry points.

    Raises `ImportError` if CUDA/hardware requirements are not met, or if the kernel or
    required symbols are not found.
    """

    if not torch.cuda.is_available():
        raise ImportError(
            "sonic-moe kernel requires CUDA, but CUDA is not available. Use a different `experts_implementation`."
        )

    # sonic-moe requires Hopper (SM90) or newer
    major = torch.cuda.get_device_capability()[0]
    if major < 9:
        raise ImportError(
            f"sonic-moe requires a Hopper (SM90+) or newer GPU, but the current device "
            f"has compute capability {major}.x. Use a different `experts_implementation`."
        )

    kernel = lazy_load_kernel("sonic-moe")
    if kernel is None:
        raise ImportError(
            "Failed to load the sonic-moe kernel — check that `kernels-community/sonic-moe` "
            "has a build matching the current torch/CUDA."
        )

    return SonicMoE(
        activation_type_enum=kernel.enums.ActivationType,
        moe_general_routing_inputs=kernel.moe_general_routing_inputs,
        moe_ep_general_routing_forward=kernel.moe_ep_general_routing_forward,
        network_profiler_cls=kernel.NetworkProfiler,
    )


@torch._dynamo.allow_in_graph
def _sonicmoe_wrapper(
    hidden_states: torch.Tensor,
    router_scores: torch.Tensor,
    expert_ids: torch.Tensor,
    token_idx: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    act_name: str,
    num_experts: int,
    concat_layout: bool,
    is_inference_mode_enabled: bool,
) -> torch.Tensor:
    """Module-level shim around `moe_general_routing_inputs` so `allow_in_graph` can wrap it.

    sonicmoe asserts `not torch.compiler.is_compiling()` internally because it dispatches
    CuteDSL kernels, which Dynamo can't trace. `allow_in_graph` keeps the call in the FX
    graph as a single opaque node (no tracing into the body, no graph break) while still
    running the real Python at runtime — autograd through `_UpProjection` / `_DownProjection`
    flows normally. The decorator must be applied at module load time, not inside the compiled
    function — hence this shim plus the `allow_in_graph` decorator above.
    """
    sonicmoe = _load_sonicmoe_kernel()
    activation_type_enum = sonicmoe.activation_type_enum
    activation_type = getattr(
        activation_type_enum, ACT_MAP.get(act_name, "swiglu").upper(), activation_type_enum.SWIGLU
    )
    output, _ = sonicmoe.moe_general_routing_inputs(
        hidden_states,
        router_scores,
        token_idx,
        expert_ids,
        w1,
        b1,
        w2,
        b2,
        E=num_experts,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        stream_id=None,
    )
    return output


def sonicmoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if not self.has_gate:
        raise ValueError("sonicmoe requires gated experts (has_gate=True)")
    if hidden_states.device.type != "cuda":
        raise ValueError("sonicmoe requires CUDA device")

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)

    # Flatten — token_indices must be int32, sorted ascending (required by sonic-moe)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1).int()
    router_scores = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1).int()

    # EP sentinel handling: leave `expert_ids` unclamped — the kernel's metadata stage drops
    # `expert_ids >= num_experts` from the per-expert histogram and masks them out of the
    # scatter indices, so sentinels never enter the grouped GEMM. Their routing weights are
    # already zero (RouterParallel masks them at dispatch), so the per-token reduction
    # contributes nothing for sentinel slots.

    w1 = to_local(self.gate_up_proj)
    w2 = to_local(self.down_proj)
    b1 = to_local(self.gate_up_proj_bias) if self.has_bias else None
    b2 = to_local(self.down_proj_bias) if self.has_bias else None

    # Map activation function
    act_name = getattr(self.config, "hidden_act", "silu").lower()
    # Permute weights as expected by sonic-moe (E=num_experts, H=hidden_size, I=intermediate_size).
    # Non-transposed: gate_up_proj is (E, 2*I, H), down_proj is (E, H, I) -> permute(1, 2, 0).
    # Transposed: gate_up_proj is (E, H, 2*I), down_proj is (E, I, H) -> permute(2, 1, 0).
    perm = (2, 1, 0) if self.is_transposed else (1, 2, 0)
    w1 = w1.permute(*perm)  # (2*I, H, E)
    w2 = w2.permute(*perm)  # (I, H, E)

    return _sonicmoe_wrapper(
        hidden_states=hidden_states,
        router_scores=router_scores,
        expert_ids=expert_ids,
        token_idx=token_idx,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        act_name=act_name,
        num_experts=self.num_experts,
        concat_layout=self.is_concatenated,
        is_inference_mode_enabled=not torch.is_grad_enabled(),
    )


def _maybe_profile_ep_config(
    self: torch.nn.Module, tokens_per_rank: int, hidden_size: int, num_top_k: int, dtype: torch.dtype
):
    """Profile and cache a `RuntimeEPConfig` per `(tokens_per_rank, hidden_size, num_top_k, dtype)`.

    Caches across shapes so prefill/decode (or varying batch sizes) don't re-profile.
    """
    if not hasattr(self, "_sonicmoe_ep_configs"):
        self._sonicmoe_ep_configs = {}
    cache_key = (tokens_per_rank, hidden_size, num_top_k, dtype)
    if cache_key not in self._sonicmoe_ep_configs:
        sonicmoe = _load_sonicmoe_kernel()
        profiler = sonicmoe.network_profiler_cls(T_local=tokens_per_rank, H=hidden_size, K=num_top_k, dtype=dtype)
        self._sonicmoe_ep_configs[cache_key] = profiler.profile()
    return self._sonicmoe_ep_configs[cache_key]


@torch._dynamo.allow_in_graph
def _sonicmoe_ep_wrapper(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    total_num_experts: int,
    act_name: str,
    concat_layout: bool,
    is_inference_mode_enabled: bool,
    process_group,
    ep_config,
) -> torch.Tensor:
    """Module-level shim around `moe_ep_general_routing_forward` so `allow_in_graph` can wrap it.

    Same reasoning as `_sonicmoe_wrapper`: the sonic kernel dispatches CuteDSL and asserts
    `not torch.compiler.is_compiling()` internally, so Dynamo can't trace it. `allow_in_graph`
    keeps the call as an opaque FX node — no tracing into the body, no graph break.
    """
    sonicmoe = _load_sonicmoe_kernel()
    activation_type_enum = sonicmoe.activation_type_enum
    activation_type = getattr(
        activation_type_enum, ACT_MAP.get(act_name, "swiglu").upper(), activation_type_enum.SWIGLU
    )
    return sonicmoe.moe_ep_general_routing_forward(
        hidden_states,
        top_k_index,
        top_k_weights,
        w1,
        b1,
        w2,
        b2,
        total_num_experts,
        group=process_group,
        activation_type=activation_type,
        is_inference_mode_enabled=is_inference_mode_enabled,
        concat_layout=concat_layout,
        ep_config=ep_config,
    )


def sonicmoe_ep_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """sonic-moe DP+EP forward.

    Strictly the **DP+EP** contract sonic-moe is designed for (upstream README: *"we always
    assume DP → EP → DP"*):

      - Input: `(T_local, H)` per rank — the rank-local share of a `T_global = T_local · W`
        global batch. **The caller is responsible for sharding** the activations to `T_local`
        rows *before* calling this impl. Routing tensors must be the matching `T_local` slice.
      - Output: `(T_local, H)` per rank — the rank's local share of the combined per-token
        outputs. The caller is responsible for any downstream gather/reduction if the rest of
        the model needs a `T_global`-shaped tensor.

    The kernel owns dispatch (AG / A2A / rank-dedup over PyTorch SymmetricMemory) + grouped GEMM
    on the rank-local `E_local = num_experts // ep_size` shard + combine (A2A / RS / rank-dedup).
    No round-trip `all_gather`, no padding — every layer that wants to use this impl has to be
    plumbed for DP+EP at the model level. `MoeTensorParalellExperts._prepare_input_fn` supplies
    `process_group` automatically when the module is wrapped for TP;
    `RouterParallel._prepare_output_fn` skips the global→local remap so `top_k_index` arrives as
    **global expert IDs** over `E_global = num_experts * W`.

    Caller-managed `self` attributes:
      - `gate_up_proj`, `down_proj`: weights already E-sharded by `MoeShardedAcrossExperts`.
      - `num_experts`: local count after sharding (`E_local`); `E_global = num_experts * W`.
      - `is_transposed`, `is_concatenated`: same semantics as the non-EP path.
      - `config.hidden_act` (optional): activation name; defaults to silu→swiglu.
    """
    if not self.has_gate:
        raise ValueError("sonicmoe_ep requires gated experts (has_gate=True)")
    if hidden_states.device.type != "cuda":
        raise ValueError("sonicmoe_ep requires CUDA device")
    if process_group is None:
        raise ValueError(
            "sonic-moe DP+EP requires a `process_group` for symm-mem rendezvous. The TP wrapping "
            "(MoeTensorParalellExperts) supplies it automatically; pass it explicitly otherwise."
        )

    num_top_k = top_k_index.size(-1)
    world_size = process_group.size()
    tokens_per_rank, hidden_size = hidden_states.shape
    total_num_experts = self.num_experts * world_size

    # Cast routing tensors — sonic-moe's EP forward takes int32 indices and hidden-dtype scores.
    top_k_index = top_k_index.to(torch.int32)
    top_k_weights = top_k_weights.to(hidden_states.dtype)

    w1 = to_local(self.gate_up_proj)
    w2 = to_local(self.down_proj)
    b1 = to_local(self.gate_up_proj_bias) if self.has_bias else None
    b2 = to_local(self.down_proj_bias) if self.has_bias else None

    # Map activation function
    act_name = getattr(self.config, "hidden_act", "silu").lower()
    # Permute weights into the layout sonic-moe's EP forward expects (E_local=num_experts after
    # sharding, H=hidden_size, I=intermediate_size). w1 → (2I, H, E_local); w2 → (E_local, I, H).
    #   Non-transposed: gate_up_proj (E_local, 2*I, H) -> .permute(1, 2, 0) -> (2*I, H, E_local)
    #                   down_proj    (E_local, H,  I)  -> .permute(0, 2, 1) -> (E_local, I, H)
    #   Transposed:     gate_up_proj (E_local, H, 2*I) -> .permute(2, 1, 0) -> (2*I, H, E_local)
    #                   down_proj    (E_local, I, H)   -> identity          -> (E_local, I, H)
    w1_perm = (2, 1, 0) if self.is_transposed else (1, 2, 0)
    w2_perm = (0, 1, 2) if self.is_transposed else (0, 2, 1)
    w1 = w1.permute(*w1_perm)
    w2 = w2.permute(*w2_perm)

    # Profile dispatch+combine primitives once per (T_local, H, K, dtype) — see `_maybe_profile_ep_config`.
    ep_config = _maybe_profile_ep_config(
        self,
        tokens_per_rank=tokens_per_rank,
        hidden_size=hidden_size,
        num_top_k=num_top_k,
        dtype=hidden_states.dtype,
    )

    return _sonicmoe_ep_wrapper(
        hidden_states=hidden_states,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        total_num_experts=total_num_experts,
        act_name=act_name,
        concat_layout=self.is_concatenated,
        is_inference_mode_enabled=not torch.is_grad_enabled(),
        process_group=process_group,
        ep_config=ep_config,
    )
