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
"""NVFP4 integration for Llama-4 MoE models.

Handles Llama-4 Scout-style batched expert weights (fused gate_up + down across
num_experts dimension) in NVFP4. Separate from nvfp4.py which targets Qwen-
style per-expert Linear layouts.

Checkpoint key convention (modelopt NVFP4 export for Llama-4):
  feed_forward.experts.gate_up_proj                     uint8  [E, in//2, 2*inter]
  feed_forward.experts.gate_up_proj_weight_scale        fp8    [E, in//16, 2*inter]
  feed_forward.experts.gate_up_proj_weight_scale_2      fp32   scalar (reciprocal)
  feed_forward.experts.down_proj                        uint8  [E, inter//2, hidden]
  feed_forward.experts.down_proj_weight_scale           fp8    [E, inter//16, hidden]
  feed_forward.experts.down_proj_weight_scale_2         fp32   scalar (reciprocal)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nvfp4 import unpack_nvfp4


class NVFP4Llama4Experts(nn.Module):
    """Llama-4 fused MoE experts, quantized to NVFP4.

    Mirrors transformers.models.llama4.modeling_llama4.Llama4TextExperts:
    gate_up_proj (gate and up fused on last dim, 2*intermediate) and down_proj,
    batched across num_experts on dim 0.

    Buffers match the checkpoint key layout directly so HF's PlaceOp loader
    (or the custom streaming loader) can populate by name with no renaming.

    Forward dequantizes per-hit-expert on the fly via unpack_nvfp4.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, device=None):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = F.silu

        # gate_up_proj: [E, in_packed, 2*inter] — checkpoint layout
        self.register_buffer(
            "gate_up_proj",
            torch.empty(
                num_experts, hidden_dim // 2, 2 * intermediate_dim,
                dtype=torch.uint8, device=device,
            ),
        )
        self.register_buffer(
            "gate_up_proj_weight_scale",
            torch.empty(
                num_experts, hidden_dim // 16, 2 * intermediate_dim,
                dtype=torch.float8_e4m3fn, device=device,
            ),
        )
        self.register_buffer(
            "gate_up_proj_weight_scale_2",
            torch.empty((), dtype=torch.float32, device=device),
        )

        # down_proj: [E, inter_packed, hidden]
        self.register_buffer(
            "down_proj",
            torch.empty(
                num_experts, intermediate_dim // 2, hidden_dim,
                dtype=torch.uint8, device=device,
            ),
        )
        self.register_buffer(
            "down_proj_weight_scale",
            torch.empty(
                num_experts, intermediate_dim // 16, hidden_dim,
                dtype=torch.float8_e4m3fn, device=device,
            ),
        )
        self.register_buffer(
            "down_proj_weight_scale_2",
            torch.empty((), dtype=torch.float32, device=device),
        )

    def _init_weights(self, module):
        pass

    def reset_parameters(self):
        pass

    def _dequant_gate_up(self, expert_idx, dtype):
        # Checkpoint is [in_packed, out]; unpack_nvfp4 expects [out, in_packed]
        packed = self.gate_up_proj[expert_idx].T.contiguous()
        scale = self.gate_up_proj_weight_scale[expert_idx].T.contiguous()
        return unpack_nvfp4(packed, scale, self.gate_up_proj_weight_scale_2, dtype)

    def _dequant_down(self, expert_idx, dtype):
        packed = self.down_proj[expert_idx].T.contiguous()
        scale = self.down_proj_weight_scale[expert_idx].T.contiguous()
        return unpack_nvfp4(packed, scale, self.down_proj_weight_scale_2, dtype)

    def forward(self, hidden_states):
        """
        Matches Llama4TextExperts.forward signature — single-arg, pre-routed input.

        Args:
            hidden_states: [num_experts * tokens_per_expert, hidden_dim]
                           Caller (Llama4TextMoe) has already replicated hidden
                           states across experts and pre-scaled by gate scores.
                           Tokens are contiguous per expert: expert i owns
                           hidden_states[i*T : (i+1)*T].
        Returns:
            [num_experts * tokens_per_expert, hidden_dim]
        """
        E = self.num_experts
        hidden_states = hidden_states.view(E, -1, self.hidden_dim)  # [E, T, hidden]
        T = hidden_states.shape[1]

        out = torch.empty(
            E, T, self.hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for e in range(E):
            x = hidden_states[e]  # [T, hidden]
            gate_up_w = self._dequant_gate_up(e, dtype=x.dtype)   # [2*inter, hidden]
            down_w = self._dequant_down(e, dtype=x.dtype)          # [hidden, inter]

            gate_up = F.linear(x, gate_up_w)                       # [T, 2*inter]
            gate, up = gate_up.chunk(2, dim=-1)
            inter = up * self.act_fn(gate)                         # [T, inter]
            out[e] = F.linear(inter, down_w)                       # [T, hidden]

        return out.view(-1, self.hidden_dim)                       # [E*T, hidden]


def replace_llama4_moe_experts_with_nvfp4(model, modules_to_not_convert=None):
    """Replace Llama4TextExperts (bf16 fused) with NVFP4Llama4Experts.

    Walks model, finds every Llama4TextExperts instance, swaps it for an
    NVFP4Llama4Experts of matching dimensions. After this runs, the state_dict
    loader will populate batched NVFP4 buffers by name.

    Returns (model, has_been_replaced).
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
    except ImportError:
        return model, False

    has_been_replaced = False
    for name, child in list(model.named_modules()):
        if not isinstance(child, Llama4TextExperts):
            continue
        if any(pattern in name for pattern in modules_to_not_convert):
            continue

        num_experts = child.num_experts
        # child.gate_up_proj shape: [E, hidden, 2*intermediate] on meta
        hidden_dim = child.gate_up_proj.shape[1]
        intermediate_dim = child.gate_up_proj.shape[2] // 2

        nvfp4_experts = NVFP4Llama4Experts(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nvfp4_experts)
        has_been_replaced = True

    return model, has_been_replaced
