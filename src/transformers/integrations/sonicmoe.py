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

Provides `sonicmoe_experts_forward` registered as "sonicmoe" in the ExpertsInterface.
Requirements: CUDA, `kernels`, `nvidia-cutlass-dsl`, has_gate=True.
"""

from __future__ import annotations

import functools

import torch

from ..utils import logging
from .hub_kernels import lazy_load_kernel


logger = logging.get_logger(__name__)

# Map activation function names from HF config to SonicMoE epilogue names
ACT_MAP = {"silu": "swiglu", "gelu": "geglu", "relu": "reglu"}


@functools.cache
def _load_sonic_kernel():
    """
    Load sonic-moe once and return its required symbols.

    Raises:
        ImportError if CUDA/hardware requirements are not met, or if the kernel or
        required symbols are not found.

    Returns:
        Tuple of (ActivationType, moe_general_routing_inputs function) from the sonic-moe kernel.
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

    ActivationType = getattr(getattr(kernel, "enums", None), "ActivationType", None)
    moe_general_routing_inputs = getattr(kernel, "moe_general_routing_inputs", None)

    missing = [
        name
        for name, attr in [
            ("enums.ActivationType", ActivationType),
            ("moe_general_routing_inputs", moe_general_routing_inputs),
        ]
        if attr is None
    ]
    if missing:
        raise ImportError(
            f"sonic-moe kernel is missing required symbols: {', '.join(missing)}. "
            "Make sure you have the `kernels` package and `nvidia-cutlass-dsl` installed."
        )

    return ActivationType, moe_general_routing_inputs


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
    ActivationType, moe_general_routing_inputs = _load_sonic_kernel()
    activation_type = getattr(ActivationType, ACT_MAP.get(act_name, "swiglu").upper(), ActivationType.SWIGLU)
    output, _ = moe_general_routing_inputs(
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

    # Map activation function
    act_name = getattr(self.config, "hidden_act", "silu").lower()
    # Permute weights as expected by sonic-moe (E=num_experts, H=hidden_size, I=intermediate_size).
    # Non-transposed: gate_up_proj is (E, 2*I, H), down_proj is (E, H, I) -> permute(1, 2, 0).
    # Transposed: gate_up_proj is (E, H, 2*I), down_proj is (E, I, H) -> permute(2, 1, 0).
    perm = (2, 1, 0) if self.is_transposed else (1, 2, 0)
    w1 = self.gate_up_proj.permute(*perm)  # (2*I, H, E)
    w2 = self.down_proj.permute(*perm)  # (I, H, E)
    b1 = self.gate_up_proj_bias if self.has_bias else None
    b2 = self.down_proj_bias if self.has_bias else None

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
