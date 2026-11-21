"""Experiment helper: Metal affine quantization of the always-on `nn.Linear`s (attention, lm_head) after loading."""

import torch


def _clipped_affine(weight, group_size, bits, shrinks=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    """Affine quantization with a per-group clipping search: for each group, the [min, max] range shrunk by the factor
    that minimizes the reconstruction error (instead of plain min-max). Same packed layout as `_affine_quantize_tensor`."""
    from transformers.integrations.metal_quantization import _affine_quantize_tensor

    rows, columns = weight.shape
    w = weight.float().reshape(rows, columns // group_size, group_size)
    w_min, w_max = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    levels = 2**bits - 1
    best_err, best = None, None
    for shrink in shrinks:
        lo, hi = w_min * shrink, w_max * shrink
        scale = ((hi - lo) / levels).clamp(min=1e-8)
        q = ((w - lo) / scale).round().clamp(0, levels)
        err = ((q * scale + lo - w) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best = err, (lo, scale)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = (torch.where(better, lo, best[0]), torch.where(better, scale, best[1]))
    lo, scale = best
    # Clip, then quantize with min-max: the clipped group's min/max are exactly (lo, lo + levels * scale)
    clipped = torch.maximum(torch.minimum(w, lo + levels * scale), lo).reshape(rows, columns)
    return _affine_quantize_tensor(clipped, group_size, bits)


def quantizable(name, linear, group_size=32):
    """The `nn.Linear`s 4-bit on Metal: all but the MoE router (and inputs not a multiple of the group)."""
    return type(linear) is torch.nn.Linear and name != "gate" and linear.in_features % group_size == 0


def quantize_dense(model, spec):
    """Replace the bf16 `nn.Linear`s (not the MoE router) with Metal affine `MetalLinear`.
    `spec`: "<bits>[g<group>][c][:<lm_head bits>]", e.g. "4", "4g32", "4c" (clipping search), "8:4"."""
    from transformers.integrations.metal_quantization import MetalLinear, _affine_quantize_tensor

    body, _, head = str(spec).partition(":")
    clip = body.endswith("c")
    body = body.removesuffix("c")
    bits, _, group = body.partition("g")
    bits, group_size, head_bits = int(bits), int(group or 64), int(head or bits)
    count = 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not quantizable(child_name, child, group_size):
                continue
            b = head_bits if child_name == "lm_head" else bits
            q = MetalLinear(
                child.in_features, child.out_features, bias=child.bias is not None, bits=b, group_size=group_size
            )
            quantize = _clipped_affine if clip else _affine_quantize_tensor
            packed, scales, biases = quantize(child.weight.data.float(), group_size, b)
            q.weight = torch.nn.Parameter(packed, requires_grad=False)
            q.scales = torch.nn.Parameter(scales.to(child.weight.dtype), requires_grad=False)
            q.qbiases = torch.nn.Parameter(biases.to(child.weight.dtype), requires_grad=False)
            if child.bias is not None:
                q.bias = child.bias
            setattr(module, child_name, q.to(child.weight.device))
            count += 1
    from transformers.integrations.metal_quantization import fuse_sibling_linears

    print(f"  quantized {count} nn.Linear: {spec}, fused {fuse_sibling_linears(model)} sibling groups", flush=True)
