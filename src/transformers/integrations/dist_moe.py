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

"""Dist-MoE integration: the fused expert-parallel MoE of [`dist_moe`](https://github.com/meta-pytorch/dist_moe).

Provides:
- `is_dist_moe_available` / `load_dist_moe`: whether Dist-MoE runs here, and the package.
- `dist_moe_experts_forward`: the `"dist_moe"` experts implementation, dispatch, SwiGLU experts and combine in one
  differentiable call, on a context shared by the layers of an EP group.
- `destroy_dist_moe_contexts`: frees the contexts.

Requirements: the `dist_moe` package, CUDA and a GPU of compute capability 10.0 (Blackwell) or newer.
"""

import functools
import importlib.util
import weakref

import torch

from ..utils import logging
from ..utils.import_utils import maybe_import_error


logger = logging.get_logger(__name__)


# ── Loading ────────────────────────────────────────────────────────────────────


@functools.lru_cache
def is_dist_moe_available(raise_error: bool = False) -> bool:
    """Whether Dist-MoE runs in this environment: the `dist_moe` package installed and a CUDA GPU of compute capability
    10.0 or newer. With `raise_error=True` (used by `load_dist_moe`) it raises the `ImportError` explaining what is
    missing instead of returning `False`."""
    if importlib.util.find_spec("dist_moe") is None:
        return maybe_import_error(
            "Dist-MoE is not installed: install it from https://github.com/meta-pytorch/dist_moe.",
            raise_error=raise_error,
        )
    if not torch.cuda.is_available():
        return maybe_import_error("Dist-MoE requires CUDA, but CUDA is not available.", raise_error=raise_error)
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return maybe_import_error(
            f"Dist-MoE requires a GPU of compute capability 10.0 (Blackwell) or newer; the current device is "
            f"SM{major}{minor}.",
            raise_error=raise_error,
        )
    return True


def load_dist_moe():
    """The `dist_moe` package, raising an `ImportError` that says what is missing when Dist-MoE cannot run here."""
    is_dist_moe_available(raise_error=True)
    import dist_moe

    return dist_moe


# ── Contexts ───────────────────────────────────────────────────────────────────


class _Context:
    """A Dist-MoE context and the backwards that still need its saved activations.

    A training context keeps the activations of every layer of a forward in one activation slot, released layer by
    layer by the backward. A forward whose backward never runs (gradients enabled, the graph dropped) would leave the
    slot full, so the next forward that finds no pending backward resets it first.
    """

    def __init__(self, context, key):
        self.context = context
        self.key = key
        self.pending = weakref.WeakSet()  # one token per layer forward whose backward has not run

    def begin_forward(self):
        if not self.pending:
            self.context.reset()


class _BackwardToken:
    pass


class _TrackBackward(torch.autograd.Function):
    """Identity that marks its layer's backward as run: the token dies with the graph, or is dropped on backward."""

    @staticmethod
    def forward(ctx, output, state):
        ctx.state = state
        ctx.token = _BackwardToken()
        state.pending.add(ctx.token)
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad_output):
        ctx.state.pending.discard(ctx.token)
        return grad_output, None


# (EP group name, training) -> its context. Every layer of an EP group shares it, since the layers run one after
# another.
_CONTEXTS: dict[tuple[str, bool], _Context] = {}


def _get_context(group, training, capacity, hidden_dim, intermediate_dim, num_top_k, num_experts, num_layers):
    """The Dist-MoE context of `group`, rebuilt (a collective) when it does not hold `capacity` tokens per rank of these
    shapes. Never built under CUDA graph capture."""
    dist_moe = load_dist_moe()
    state = _CONTEXTS.get((group.group_name, training))
    key = (hidden_dim, intermediate_dim, num_top_k, num_experts)
    if (
        state is not None
        and state.key[0] == key
        and capacity <= state.key[1]
        and (not training or num_layers <= state.key[2])
    ):
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"The Dist-MoE context of this expert-parallel group does not hold {capacity} tokens per rank yet, and it "
            "cannot be allocated during CUDA graph capture: run the forward eagerly at this batch size before "
            "capturing it (the usual warm-up), so the context is allocated outside the capture."
        )
    if state is not None:
        # every rank rebuilds here, since they agreed on the capacity; let the old context's kernels finish first
        capacity = max(capacity, state.key[1])
        num_layers = max(num_layers, state.key[2])
        state.context.close()
        del _CONTEXTS[(group.group_name, training)]
    ep_size = torch.distributed.get_world_size(group)
    config = dist_moe.Config(
        max_num_local_input_tokens=capacity,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        top_k=num_top_k,
        num_experts=num_experts,
        max_moe_layers_per_activation_slot=num_layers,
        # scratch overflow is a device trap: size it for the worst routing, every token on one rank
        device_scratch_capacity_factor=float(ep_size),
        # keep every layer's activations for the backward when the routing is balanced, recompute past that
        activation_slot_capacity_factor=1.0 if training else None,
        num_activation_slots=1 if training else 0,
        inference=not training,
    )
    context = dist_moe.create_context(group=group, config=config, device=torch.cuda.current_device())
    state = _Context(context, (key, capacity, num_layers))
    _CONTEXTS[(group.group_name, training)] = state
    return state


def destroy_dist_moe_contexts():
    """Free the Dist-MoE contexts of every EP group. A collective: every rank calls it, before the process groups are
    destroyed."""
    for state in _CONTEXTS.values():
        state.context.close()
    _CONTEXTS.clear()


# ── Routed experts ─────────────────────────────────────────────────────────────


def dist_moe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group=None,
    capacity: int | None = None,
    num_layers: int = 1,
) -> torch.Tensor:
    """
    The `"dist_moe"` experts implementation: Dist-MoE sends each token to the ranks of its top-k experts (global ids),
    runs their SwiGLU on this module's `gate_up_proj` (`[local experts, 2 * intermediate, hidden]`, `[gate; up]`) and
    `down_proj` (`[local experts, hidden, intermediate]`), and sums the weighted outputs back on the token's rank,
    forward and backward. It exchanges the tokens itself, so it runs under expert parallelism only: the `"dist_moe"`
    EP strategy supplies `process_group`, `capacity` and `num_layers`.

    Every rank of `process_group` must pass the same number of rows, at most `capacity` (the same on all of them): pad
    with expert id -1 and weight 0. `num_layers` bounds the layers whose activations a training forward keeps until
    the backward. Tokens and weights compute in bfloat16.
    """
    if process_group is None:
        raise ValueError(
            '`experts_implementation="dist_moe"` exchanges the tokens between expert-parallel ranks itself, so it needs '
            "expert parallelism: load the model with a `DistributedConfig` whose `ep_size > 1`, which runs it through "
            '`ep_implementation="dist_moe"`.'
        )
    group, gate_up_proj, down_proj = process_group, self.gate_up_proj, self.down_proj
    capacity = hidden_states.size(0) if capacity is None else capacity
    training = torch.is_grad_enabled() and any(
        t.requires_grad for t in (hidden_states, top_k_weights, gate_up_proj, down_proj)
    )
    hidden_dim, intermediate_dim = down_proj.shape[1], down_proj.shape[2]
    num_experts = gate_up_proj.shape[0] * torch.distributed.get_world_size(group)
    state = _get_context(
        group, training, capacity, hidden_dim, intermediate_dim, top_k_index.size(-1), num_experts, num_layers
    )
    if top_k_weights.dtype not in (torch.bfloat16, torch.float32):
        top_k_weights = top_k_weights.float()
    if training:
        state.begin_forward()
    else:
        # the inference context rejects tensors that require gradients, which parameters do even under `no_grad`
        hidden_states, top_k_weights, gate_up_proj, down_proj = (
            t.detach() for t in (hidden_states, top_k_weights, gate_up_proj, down_proj)
        )
    output = load_dist_moe().routed_experts(
        hidden_states.to(torch.bfloat16).contiguous(),
        top_k_index.contiguous(),
        top_k_weights.contiguous(),
        gate_up_proj.to(torch.bfloat16).contiguous(),
        down_proj.to(torch.bfloat16).contiguous(),
        state.context,
    )
    return _TrackBackward.apply(output, state) if training else output
