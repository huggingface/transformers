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

"""nCompass Triton MoE integration: fused bf16 MoE from `nCompass-tech/triton-moe`.

Provides `ncompass_moe_experts_forward`, registered as "ncompass_moe" in the ExpertsInterface.
The kernel is OpenAI/GPT-OSS-style (combined gate_up + down projections *with biases* and a
clamped-SwiGLU `alpha`); its `layers.moe_forward` is the stateless functional entry point that the
`MoE` / `OpenaiExperts` layers both call. Added for benchmarking against the built-in experts impls.

Requirements: CUDA, `kernels`, gated experts with biases (has_gate=True, has_bias=True).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils import logging
from ..utils.import_utils import is_kernels_available
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)

# OpenAI/GPT-OSS clamped-SwiGLU sigmoid-gate coefficient; used when the config doesn't set one.
_DEFAULT_SWIGLU_ALPHA = 1.702


@dataclass(frozen=True)
class NCompassMoE:
    """Entry point exposed by the `nCompass-tech/triton-moe` kernel."""

    moe_forward: Callable


_NCOMPASS_MOE: NCompassMoE | None = None


def _load_ncompass_moe_kernel() -> NCompassMoE:
    """
    Load nCompass triton-moe once and return its functional entry point.

    Cached in the module-global ``_NCOMPASS_MOE`` (populated on first call) instead of via
    ``functools.cache``: Dynamo traces through the lru wrapper and warns on every compiled call,
    whereas a plain global check/return traces cleanly.

    Loaded with ``trust_remote_code=True`` — ``nCompass-tech`` is not the trusted ``kernels-community``
    publisher, so it can't go through ``lazy_load_kernel`` (which only trusts untrusted repos when the
    global ``ALLOW_ALL_KERNELS`` is set). This is an explicit opt-in for a benchmark integration.

    Raises `ImportError` if `kernels`/CUDA are unavailable or the kernel/symbols can't be found.
    """
    global _NCOMPASS_MOE
    if _NCOMPASS_MOE is not None:
        return _NCOMPASS_MOE

    if not is_kernels_available():
        raise ImportError("ncompass_moe requires the `kernels` package. Install it with `pip install kernels`.")
    if not torch.cuda.is_available():
        raise ImportError(
            "nCompass triton-moe kernel requires CUDA, but CUDA is not available. "
            "Use a different `experts_implementation`."
        )

    from kernels import get_kernel

    try:
        kernel = get_kernel("nCompass-tech/triton-moe", trust_remote_code=True, version=1)
    except Exception as e:
        raise ImportError(
            "Failed to load the nCompass triton-moe kernel — check that `nCompass-tech/triton-moe` has a "
            f"build matching the current torch/CUDA and that `kernels` can reach the Hub. Reason: {e}"
        ) from e

    moe_forward = getattr(getattr(kernel, "layers", None), "moe_forward", None)
    if moe_forward is None:
        raise ImportError("nCompass triton-moe kernel is missing `layers.moe_forward`; check the repo revision.")

    _NCOMPASS_MOE = NCompassMoE(moe_forward=moe_forward)
    return _NCOMPASS_MOE


@torch.library.custom_op("ncompass_moe::fused_moe", mutates_args=())
def _ncompass_moe_op(
    hidden_states: torch.Tensor,
    router_idx: torch.Tensor,
    router_wt: torch.Tensor,
    alpha: float,
    gate_up_weights: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weights: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Opaque custom op wrapping `layers.moe_forward`. Unlike a bare `allow_in_graph` shim (as in
    sonicmoe, whose entry point is fake-safe), the nCompass Triton kernel dereferences ``.data_ptr()``,
    which fails under FakeTensor tracing — so torch.compile/export need the ``register_fake`` below to
    learn the output shape without running the kernel."""
    return _load_ncompass_moe_kernel().moe_forward(
        hidden_states, router_idx, router_wt, alpha, gate_up_weights, gate_up_bias, down_weights, down_bias
    )


@_ncompass_moe_op.register_fake
def _(
    hidden_states,
    router_idx,
    router_wt,
    alpha,
    gate_up_weights,
    gate_up_bias,
    down_weights,
    down_bias,
):
    # Down-projection output: one row per token, hidden-dim wide — same shape/dtype as the input.
    return torch.empty_like(hidden_states)


def ncompass_moe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if not self.has_gate:
        raise ValueError("ncompass_moe requires gated experts (a combined gate_up projection).")
    if not self.has_bias:
        raise NotImplementedError(
            "ncompass_moe targets OpenAI/GPT-OSS-style experts, whose gate_up/down projections carry "
            "biases; this module has none. Use grouped_mm / batched_mm instead."
        )
    if hidden_states.device.type != "cuda":
        raise ValueError("ncompass_moe requires a CUDA device.")

    # GPT-OSS clamped-SwiGLU coefficient — from config if present, else the OpenAI default.
    alpha = float(getattr(self.config, "swiglu_alpha", None) or _DEFAULT_SWIGLU_ALPHA)

    # HF-OpenaiExperts-compatible: weights/biases pass through in the model's native layout
    # (the kernel expects the GPT-OSS gate_up/down layout). router indices/weights map directly.
    return _ncompass_moe_op(
        hidden_states,
        top_k_index,
        top_k_weights,
        alpha,
        to_local(self.gate_up_proj),
        to_local(self.gate_up_proj_bias),
        to_local(self.down_proj),
        to_local(self.down_proj_bias),
    )
