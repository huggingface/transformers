"""[scratch] gpt-oss MXFP4 experts off Hopper and B200/B300 (Ampere, Ada, SM120 GPUs such as the RTX PRO 6000) through
vLLM's Marlin MoE kernels, which vLLM itself picks there, with the weight layout vLLM prepares for them
(vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py). triton_kernels' matmul_ogs ran gpt-oss-20b's MoE
block at ~1.2 ms per decode step on an RTX PRO 6000 and an A100. Needs vLLM; off with PROTO_MXFP4_MARLIN=0."""
import os

import torch
import torch.nn.functional as F


PROTO_MXFP4_MARLIN = os.environ.get("PROTO_MXFP4_MARLIN", "1") == "1"


def _round_up(x, m):
    return (x + m - 1) // m * m


class _Layer:  # the attribute vLLM's preparation reads from its layer
    params_dtype = torch.bfloat16


@torch.no_grad()
def prepare_gpt_oss_experts(gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias):
    """Checkpoint tensors (`*_blocks` [E, rows, cols / 32, 16] uint8, `*_scales` [E, rows, cols / 32] uint8, biases)
    -> padded (hidden to 256, intermediate to 128, as vLLM pads for Marlin) and repacked Marlin tensors."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    E, two_i, h_blocks, _ = gu_blocks.shape
    I, H = two_i // 2, h_blocks * 32
    I_p, H_p = _round_up(I, 128), _round_up(H, 256)
    dev = gu_blocks.device
    # gate/up rows stay interleaved (gate0, up0, gate1, ...), as SWIGLUOAI expects: pad the pairs, then the columns
    w13 = torch.zeros(E, I_p, 2, H_p // 2, dtype=torch.uint8, device=dev)
    w13[:, :I, :, : H // 2] = gu_blocks.reshape(E, I, 2, H // 2)
    s13 = torch.full((E, I_p, 2, H_p // 32), 127, dtype=torch.uint8, device=dev)
    s13[:, :I, :, : H // 32] = gu_scales.reshape(E, I, 2, H // 32)
    b13 = torch.zeros(E, I_p, 2, dtype=torch.float32, device=dev)
    b13[:, :I] = gu_bias.float().reshape(E, I, 2)
    w2 = torch.zeros(E, H_p, I_p // 2, dtype=torch.uint8, device=dev)
    w2[:, :H, : I // 2] = d_blocks.reshape(E, H, I // 2)
    s2 = torch.full((E, H_p, I_p // 32), 127, dtype=torch.uint8, device=dev)
    s2[:, :H, : I // 32] = d_scales.reshape(E, H, I // 32)
    b2 = torch.zeros(E, H_p, dtype=torch.float32, device=dev)
    b2[:, :H] = d_bias.float()
    w13, w2, s13, s2, b13, b2 = prepare_moe_mxfp4_layer_for_marlin(
        _Layer(), w13.reshape(E, 2 * I_p, H_p // 2), w2, s13.reshape(E, 2 * I_p, H_p // 32), s2,
        b13.reshape(E, 2 * I_p), b2,
    )  # fmt: skip
    return {"w13": w13, "w2": w2, "s13": s13, "s2": s2, "b13": b13, "b2": b2, "hidden": H, "hidden_padded": H_p}


def marlin_gpt_oss_experts(p, hidden_states, top_k_index, top_k_weights):
    """`hidden_states` [T, H] bf16, routes (`top_k_index`, `top_k_weights` fp32) -> [T, H] bf16."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    x = hidden_states
    if p["hidden_padded"] != p["hidden"]:
        x = F.pad(x, (0, p["hidden_padded"] - p["hidden"]))
    out = fused_marlin_moe(
        x, p["w13"], p["w2"], p["b13"], p["b2"], p["s13"], p["s2"], top_k_weights, top_k_index,
        quant_type_id=scalar_types.float4_e2m1f.id, activation=MoEActivation.SWIGLUOAI,
    )  # fmt: skip
    return out[:, : p["hidden"]]


def maybe_prepare_model(model) -> int:
    """After loading: the MXFP4 gpt-oss experts of `model` get their Marlin tensors (Ampere, Ada and SM120 GPUs, one
    process, vLLM installed), and drop their matmul_ogs ones. Returns the number of modules prepared."""
    from .mxfp4 import Mxfp4GptOssExperts

    if not (PROTO_MXFP4_MARLIN and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (8, 12)):
        return 0
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return 0
    try:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: F401
    except ImportError:
        return 0
    prepared = 0
    for module in model.modules():
        if not isinstance(module, Mxfp4GptOssExperts) or module.gate_up_proj_precision_config is None:
            continue
        raw = {}
        for proj in ("gate_up_proj", "down_proj"):
            weight, scales = getattr(module, proj), getattr(module, f"{proj}_precision_config").weight_scale
            scales = scales.storage.layout.unswizzle_data(scales.storage.data).transpose(-1, -2)
            blocks = weight.storage.layout.unswizzle_data(weight.storage.data).transpose(-1, -2)
            raw[proj] = (blocks.reshape(*scales.shape, -1).contiguous(), scales.contiguous())
        module._proto_marlin = prepare_gpt_oss_experts(
            *raw["gate_up_proj"], module.gate_up_proj_bias, *raw["down_proj"], module.down_proj_bias
        )
        module.gate_up_proj = module.down_proj = None
        prepared += 1
    torch.cuda.empty_cache()
    return prepared
