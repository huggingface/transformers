"""Affine 4-bit expert banks (the `MetalExperts` layout) on CUDA, small-batch decode, with the `moe-affine` hub kernel:
the routed experts' gate/up matvecs and the down matvec fused with the routing-weighted sum, straight from the packed
codes (no dequantized copy). The kernel registers its ops with fake implementations, so compiled graphs (and CUDA
graphs) capture them."""

import torch

from ..utils.import_utils import is_torchdynamo_compiling
from .hub_kernels import cuda_kernel


_KERNEL = None  # resolved on the first eager call: loading it (hub, file system) can't run inside a compiled graph


def supported(module, hidden_states) -> bool:
    global _KERNEL
    if _KERNEL is None and not is_torchdynamo_compiling() and hidden_states.device.type == "cuda":
        _KERNEL = cuda_kernel("moe-affine", hidden_states.device) or False
    return (
        bool(_KERNEL)
        and hidden_states.device.type == "cuda"
        and module.bits == 4
        and module.group_size in (32, 64, 128)
        and hidden_states.dtype == torch.bfloat16
        and module.gate_up_proj_scales.dtype == torch.bfloat16
        and not getattr(module, "has_bias", False)
    )


def experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """The routed experts of `hidden_states [T, H]`: `top_k_index` ids `>= num_experts` are sentinels (weight 0)."""
    kernel = _KERNEL
    top_k = top_k_index.shape[-1]
    ids = top_k_index.reshape(-1)
    weights = (top_k_weights.reshape(-1) * (ids < module.num_experts)).float().contiguous()
    ids = ids.clamp(max=module.num_experts - 1).to(torch.int32).contiguous()
    gs = module.group_size
    up = (module.gate_up_proj, module.gate_up_proj_scales, module.gate_up_proj_qbiases)
    down = (module.down_proj, module.down_proj_scales, module.down_proj_qbiases)
    act_features = module.down_proj.shape[-1] * (32 // module.bits)
    if (
        kernel.tc6_supported(hidden_states.shape[-1], 1, gs)
        and kernel.tc6_supported(act_features, top_k, gs)
        and all(t.is_contiguous() for t in up + down)
    ):
        # tensor cores: per block, one coalesced batch stages a row tile of every routed expert in shared memory
        act = module._apply_gate(kernel.qmv4_tc6(hidden_states.contiguous(), up, ids, 1, top_k, gs))
        return kernel.qmv4_tc6(act.contiguous(), down, ids, top_k, 1, gs, weights=weights)
    act = module._apply_gate(kernel.gather_qmv4(hidden_states.contiguous(), *up, ids, top_k, gs))
    return kernel.down_route4(act.contiguous(), *down, ids, weights, top_k, gs)
