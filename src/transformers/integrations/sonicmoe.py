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
Requirements: CUDA, `kernels`, `nvidia-cutlass-dsl`, has_gate=True, is_transposed=False.
"""

import torch

from ..utils import logging
from .hub_kernels import lazy_load_kernel


logger = logging.get_logger(__name__)

# Map activation function names from HF config to SonicMoE epilogue names
ACT_MAP = {"silu": "swiglu", "gelu": "geglu", "relu": "reglu"}


def sonicmoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if self.is_transposed:
        raise ValueError("sonicmoe requires non-transposed weights (is_transposed=False)")
    if not self.has_gate:
        raise ValueError("sonicmoe requires gated experts (has_gate=True)")
    if hidden_states.device.type != "cuda":
        raise ValueError("sonicmoe requires CUDA device")

    sonic_moe = lazy_load_kernel("sonic-moe")
    moe_general_routing = getattr(sonic_moe, "moe_general_routing_inputs", None)
    ActivationType = getattr(getattr(sonic_moe, "enums", None), "ActivationType", None)
    if moe_general_routing is None or ActivationType is None:
        raise ImportError(
            "moe_general_routing_inputs function or ActivationType enum not found in kernels-community/sonic-moe. "
            "Make sure you have the `kernels` package and `nvidia-cutlass-dsl` installed."
        )

    device = hidden_states.device
    num_experts = self.num_experts
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    grad_enabled = torch.is_grad_enabled()
    stream_id = torch.cuda.current_stream(device).cuda_stream

    # Flatten — token_indices must be int32, sorted ascending (required by sonic-moe)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1).int()
    router_scores = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1).int()

    # Map activation function
    act_name = getattr(self.config, "hidden_act", "silu").lower()
    activation_type = getattr(ActivationType, ACT_MAP.get(act_name, "swiglu").upper(), ActivationType.SWIGLU)

    # Permute weights to (E, H, I) as expected by sonic-moe (E=num_experts, H=hidden_size, I=intermediate_size)
    w1 = self.gate_up_proj.permute(1, 2, 0)  # (2*I, H, E)
    w2 = self.down_proj.permute(1, 2, 0)  # (I, H, E)
    b1 = self.gate_up_proj_bias if self.has_bias else None
    b2 = self.down_proj_bias if self.has_bias else None

    output, _ = moe_general_routing(
        hidden_states,
        router_scores,
        token_idx,
        expert_ids,
        w1,
        b1,
        w2,
        b2,
        E=num_experts,
        stream_id=stream_id,
        activation_type=activation_type,
        is_inference_mode_enabled=not grad_enabled,
        is_concatenated_gate_up=True,
    )

    return output
