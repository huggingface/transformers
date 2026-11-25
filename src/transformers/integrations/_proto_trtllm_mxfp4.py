"""[scratch] gpt-oss MXFP4 experts on Blackwell through flashinfer's TRT-LLM routed MoE (bf16 activations), with the
weight layout vLLM prepares for it (vllm/model_executor/layers/fused_moe/oracle/mxfp4.py, TRTLLM backends). Off with
PROTO_MXFP4_TRTLLM=0."""
import os

import torch
import torch.nn.functional as F


def _round_up(x, m):
    return (x + m - 1) // m * m


def _swap_every_two_rows(x, axis):
    shape = list(x.shape)
    axis = axis % len(shape)
    new_shape = shape[:axis] + [shape[axis] // 2, 2] + shape[axis + 1 :]
    return x.reshape(new_shape).flip(axis + 1).reshape(shape)


@torch.no_grad()
def prepare_gpt_oss_experts(gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias, alpha=1.702, limit=7.0):
    """Checkpoint tensors (`*_blocks` [E, rows, cols / 32, 16] uint8, `*_scales` [E, rows, cols / 32] uint8, biases)
    -> padded, swapped and shuffled TRT-LLM tensors."""
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    E, two_i, h_blocks, _ = gu_blocks.shape
    I, H = two_i // 2, h_blocks * 32
    I_p, H_p = _round_up(I, 128), _round_up(H, 256)
    dev = gu_blocks.device
    # gate/up rows come in (gate, up) pairs: pad the pairs, then the hidden columns
    w13 = torch.zeros(E, I_p, 2, H_p // 2, dtype=torch.uint8, device=dev)
    w13[:, :I, :, : H // 2] = gu_blocks.reshape(E, I, 2, H // 2)
    s13 = torch.full((E, I_p, 2, H_p // 32), 127, dtype=torch.uint8, device=dev)
    s13[:, :I, :, : H // 32] = gu_scales.reshape(E, I, 2, H // 32)
    b13 = torch.zeros(E, I_p, 2, dtype=torch.float32, device=dev)
    b13[:, :I] = gu_bias.float().reshape(E, I, 2)
    w13, s13, b13 = w13.reshape(E, 2 * I_p, H_p // 2), s13.reshape(E, 2 * I_p, H_p // 32), b13.reshape(E, 2 * I_p)
    w2 = torch.zeros(E, H_p, I_p // 2, dtype=torch.uint8, device=dev)
    w2[:, :H, : I // 2] = d_blocks.reshape(E, H, I // 2)
    s2 = torch.full((E, H_p, I_p // 32), 127, dtype=torch.uint8, device=dev)
    s2[:, :H, : I // 32] = d_scales.reshape(E, H, I // 32)
    b2 = torch.zeros(E, H_p, dtype=torch.float32, device=dev)
    b2[:, :H] = d_bias.float()
    # TRT-LLM's swiglu takes (up, gate) pairs
    w13, s13, b13 = _swap_every_two_rows(w13, -2), _swap_every_two_rows(s13, -2), _swap_every_two_rows(b13, -1)
    cache, tile = {}, 128
    out = {k: [] for k in ("w13", "s13", "b13", "w2", "s2", "b2")}
    for e in range(E):
        idx = get_w2_permute_indices_with_cache(cache, w13[e], tile)
        out["w13"].append(w13[e][idx.to(dev)].contiguous())
        idx = get_w2_permute_indices_with_cache(cache, s13[e], tile, num_elts_per_sf=16)
        out["s13"].append(nvfp4_block_scale_interleave(s13[e][idx.to(dev)].contiguous()))
        idx = get_w2_permute_indices_with_cache(cache, b13[e].reshape(-1, 1), tile)
        out["b13"].append(b13[e].reshape(-1, 1)[idx.to(dev)].contiguous())
        idx = get_w2_permute_indices_with_cache(cache, w2[e], tile)
        out["w2"].append(w2[e][idx.to(dev)].contiguous())
        idx = get_w2_permute_indices_with_cache(cache, s2[e], tile, num_elts_per_sf=16)
        out["s2"].append(nvfp4_block_scale_interleave(s2[e][idx.to(dev)].contiguous()))
        idx = get_w2_permute_indices_with_cache(cache, b2[e].reshape(-1, 1), tile)
        out["b2"].append(b2[e].reshape(-1, 1)[idx.to(dev)].contiguous())
    return {
        "w13": torch.stack(out["w13"]),
        "s13": torch.stack(out["s13"]).reshape(E, 2 * I_p, H_p // 32).view(torch.float8_e4m3fn),
        "b13": torch.stack(out["b13"]).reshape(E, -1),
        "w2": torch.stack(out["w2"]),
        "s2": torch.stack(out["s2"]).reshape(E, H_p, I_p // 32).view(torch.float8_e4m3fn),
        "b2": torch.stack(out["b2"]).reshape(E, -1),
        "alpha": torch.full((E,), alpha, dtype=torch.float32, device=dev),
        "beta": torch.ones(E, dtype=torch.float32, device=dev),
        "limit": torch.full((E,), limit, dtype=torch.float32, device=dev),
        "num_experts": E, "hidden": H, "hidden_padded": H_p, "intermediate_padded": I_p,
    }


def trtllm_gpt_oss_experts(p, hidden_states, top_k_index, top_k_weights, local_expert_offset=0):
    """`hidden_states` [T, H] bf16, routes (`top_k_index` int32 global ids, `top_k_weights`) -> [T, H] bf16."""
    from flashinfer.fused_moe import RoutingMethodType, trtllm_fp4_block_scale_routed_moe

    x = hidden_states
    if p["hidden_padded"] != p["hidden"]:
        x = F.pad(x, (0, p["hidden_padded"] - p["hidden"]))
    out = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(top_k_index, top_k_weights), routing_bias=None, hidden_states=x, hidden_states_scale=None,
        gemm1_weights=p["w13"], gemm1_weights_scale=p["s13"], gemm1_bias=p["b13"], gemm1_alpha=p["alpha"],
        gemm1_beta=p["beta"], gemm1_clamp_limit=p["limit"], gemm2_weights=p["w2"], gemm2_weights_scale=p["s2"],
        gemm2_bias=p["b2"], output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
        num_experts=p["num_experts"], top_k=top_k_index.shape[-1], n_group=None, topk_group=None,
        intermediate_size=p["intermediate_padded"], local_expert_offset=local_expert_offset,
        local_num_experts=p["num_experts"], routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize, do_finalize=True,
    )  # fmt: skip
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out[:, : p["hidden"]]


PROTO_MXFP4_TRTLLM = os.environ.get("PROTO_MXFP4_TRTLLM", "1") == "1"


def maybe_prepare_model(model) -> int:
    """After loading: the MXFP4 gpt-oss experts of `model` get their TRT-LLM tensors (Blackwell, one process), and
    drop their matmul_ogs ones. Returns the number of modules prepared."""
    from .mxfp4 import Mxfp4GptOssExperts

    if not (PROTO_MXFP4_TRTLLM and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
        return 0
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return 0
    try:
        import flashinfer.fused_moe  # noqa: F401
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
        module._proto_trtllm = prepare_gpt_oss_experts(
            *raw["gate_up_proj"], module.gate_up_proj_bias, *raw["down_proj"], module.down_proj_bias,
            alpha=module.alpha, limit=module.limit,
        )  # fmt: skip
        module.gate_up_proj = module.down_proj = None
        prepared += 1
    torch.cuda.empty_cache()
    return prepared
