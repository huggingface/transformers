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
"""Runtime for MoE experts stored in NVFP4.

NVFP4 shares the e2m1 value encoding with MXFP4, but scales each group of 16 values by an e4m3 factor
divided by one fp32 factor for the whole tensor, where MXFP4 uses a power of two per group of 32. No
matmul kernel reachable from transformers takes that layout today: cuBLAS only runs FP4 matmuls on
Blackwell (SM100+), and the mxfp4 Triton kernels are hard-wired to e8m0 scales. So the weights are kept
packed in memory — which is what makes these checkpoints loadable at all — and each expert is expanded
one at a time inside the forward pass, never more than one at once.
"""

from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

from .mxfp4 import FP4_VALUES


logger = logging.get_logger(__name__)

# One e4m3 scale per this many values, against MXFP4's 32.
NVFP4_GROUP_SIZE = 16


def attach_packed_nvfp4_proj(
    module, proj: str, packed: torch.Tensor, scales: torch.Tensor, global_scales: torch.Tensor
) -> None:
    """Attach one NVFP4-packed expert projection to `module`, replacing its dense parameter.

    `packed` is `(num_experts, out_dim, in_dim // 2)` uint8, `scales` the e4m3 group scales
    `(num_experts, out_dim, in_dim // 16)`, and `global_scales` the per-projection fp32 factor broadcast
    to `(num_experts, out_dim, 1)`. Buffers rather than parameters, since none of them is a weight
    gradients could flow through, and persistent ones because loading replaces every non-persistent
    buffer with uninitialized memory once the checkpoint is in (they are assumed to be meta placeholders
    that `_init_weights` fills).
    """
    module._parameters.pop(proj, None)
    module.register_buffer(proj, packed)
    module.register_buffer(f"{proj}_scale", scales)
    module.register_buffer(f"{proj}_global_scale", global_scales)


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Expand one NVFP4 projection, `(out_dim, in_dim // 2)` uint8, to a dense `(out_dim, in_dim)` weight.

    Each byte holds two e2m1 values, low nibble first. The group scales are read at the weight dtype and
    divided by the global one while it is still fp32, which is what `compressed_tensors.dequantize` does
    for the shape-`(1,)` global scales these checkpoints ship, so the two load paths agree on the weights
    down to the last bit. (Dividing at the weight dtype instead moves the result by an ulp — PyTorch would
    only narrow the division that way if the global scale were stored as a scalar.)
    """
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=packed.device)
    nibbles = torch.stack(((packed & 0x0F).int(), (packed >> 4).int()), dim=-1)
    unpacked = lut[nibbles.reshape(packed.shape[0], -1)]

    scale = scales.to(dtype) / global_scale
    grouped = unpacked.to(scale.dtype).unflatten(-1, (scales.shape[-1], NVFP4_GROUP_SIZE))
    return (grouped * scale.unsqueeze(-1)).flatten(-2).to(dtype)


def nvfp4_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run fused MoE experts whose weights stayed packed in NVFP4.

    Only the experts a token actually routed to are expanded, one at a time, so the peak cost over the
    packed weights is a single dense expert rather than the whole layer. Gating comes from the module's
    own `_apply_gate`, so the numerics stay those of the model rather than of this format.
    """
    has_gate = getattr(self, "has_gate", True)
    proj = "gate_up_proj" if has_gate else "up_proj"
    if not hasattr(self, f"{proj}_scale"):
        raise ValueError(
            f"`experts_implementation='nvfp4'` needs NVFP4-packed expert weights, which {type(self).__name__} "
            "does not hold. It is selected automatically when loading an NVFP4-packed checkpoint and cannot be "
            "requested for a model that was not loaded from one."
        )

    dtype = hidden_states.dtype
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        # The extra class absorbs the out-of-range ids expert parallelism uses as padding.
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

    for expert_index in expert_hit:
        if expert_index == self.num_experts:
            continue
        top_k_pos, token_index = torch.where(expert_mask[expert_index])
        current_state = hidden_states[token_index]

        weight = dequantize_nvfp4(
            getattr(self, proj)[expert_index],
            getattr(self, f"{proj}_scale")[expert_index],
            getattr(self, f"{proj}_global_scale")[expert_index],
            dtype,
        )
        proj_out = torch.nn.functional.linear(current_state, weight)
        proj_out = self._apply_gate(proj_out) if has_gate else self.act_fn(proj_out)

        weight = dequantize_nvfp4(
            self.down_proj[expert_index],
            self.down_proj_scale[expert_index],
            self.down_proj_global_scale[expert_index],
            dtype,
        )
        proj_out = torch.nn.functional.linear(proj_out, weight)
        proj_out = proj_out * top_k_weights[token_index, top_k_pos, None]
        final_hidden_states.index_add_(0, token_index, proj_out.to(final_hidden_states.dtype))

    return final_hidden_states
