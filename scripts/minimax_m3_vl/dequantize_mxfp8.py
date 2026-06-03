"""Dequantize an MXFP8-quantized MiniMax M3 VL checkpoint to bf16.

MiniMax M3 VL ships in MXFP8 format with ``weight_block_size = [1, 32]`` —
each 32-element block along the last dim shares one fp8 (E4M3) scale. The
HF ``transformers`` repo currently has MXFP4 (``Mxfp4Config``) but no
matching MXFP8 path, so this first port loads MXFP8 checkpoints by
dequantizing them up-front into bf16 / fp16 and stripping
``quantization_config`` from ``config.json``.

Each ``layer.weight`` in the source checkpoint is paired with
``layer.weight_scale`` (or ``layer.weight_scale_inv``) of shape
``[..., num_blocks]`` (one scale per block of 32 along the last axis).
The dequant rule is::

    out[..., j*32 + k] = weight[..., j*32 + k].to(torch.float) * scales[..., j]

Run on the real checkpoint::

    python scripts/minimax_m3_vl/dequantize_mxfp8.py \\
        --source /raid/.../MiniMaxAI--Minimax-M3-preview/snapshots/<sha> \\
        --out    /raid/.../MiniMax-M3-preview-bf16

The script preserves shards, weight names, and shape; only the storage dtype
changes (E4M3 -> bf16) and the scale tensors are dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BLOCK = 32


def _dequant_block_e4m3(w_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize one tensor blocked over the last dim by ``BLOCK``."""
    w = w_fp8.to(torch.float32)
    last = w.shape[-1]
    if last % BLOCK != 0:
        raise ValueError(f"expected last dim {last} to be a multiple of {BLOCK}")
    n_blocks = last // BLOCK
    # scales shape: [..., n_blocks]; broadcast along the last axis
    if scales.shape[-1] != n_blocks:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match weight {tuple(w.shape)} "
            f"with block={BLOCK}"
        )
    scales = scales.to(torch.float32).unsqueeze(-1)  # [..., n_blocks, 1]
    return (w.view(*w.shape[:-1], n_blocks, BLOCK) * scales).reshape_as(w)


def _is_quant_weight(name: str, ignored: list[str]) -> bool:
    if name.endswith(".weight_scale") or name.endswith(".weight_scale_inv"):
        return False
    if not name.endswith(".weight"):
        return False
    return all(skip not in name for skip in ignored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    args.out.mkdir(parents=True, exist_ok=True)

    cfg_path = args.source / "config.json"
    with cfg_path.open() as fh:
        cfg = json.load(fh)
    qcfg = cfg.pop("quantization_config", {})
    ignored: list[str] = qcfg.get("ignored_layers", [])
    if qcfg.get("quant_method") != "mxfp8":
        raise SystemExit(f"unexpected quant_method={qcfg.get('quant_method')!r}; need 'mxfp8'")

    # Copy non-weight files (tokenizer, generation, code) verbatim.
    for path in args.source.iterdir():
        if path.name.endswith(".safetensors") or path.name == "config.json":
            continue
        if path.is_file():
            shutil.copy2(path, args.out / path.name)
        elif path.is_dir():
            shutil.copytree(path, args.out / path.name, dirs_exist_ok=True)

    # Dequantize each shard.
    shards = sorted(args.source.glob("model-*.safetensors")) or [args.source / "model.safetensors"]
    new_index: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            names = list(f.keys())
            tensors = {n: f.get_tensor(n) for n in names}
        out: dict[str, torch.Tensor] = {}
        for name, t in tensors.items():
            if name.endswith(".weight_scale") or name.endswith(".weight_scale_inv"):
                continue
            if _is_quant_weight(name, ignored):
                scale_name = name + "_scale"
                inv_scale_name = name + "_scale_inv"
                if scale_name in tensors:
                    scale = tensors[scale_name]
                elif inv_scale_name in tensors:
                    scale = 1.0 / tensors[inv_scale_name].to(torch.float32)
                else:
                    raise KeyError(f"no scale tensor for {name}")
                deq = _dequant_block_e4m3(t, scale).to(target_dtype)
                out[name] = deq
            else:
                # ignored / non-quant tensor — keep as-is
                out[name] = t
        out_shard = args.out / shard.name
        save_file(out, out_shard)
        for k in out:
            new_index[k] = out_shard.name
        print(f"dequantized {shard.name} ({len(out)} tensors)")

    # Rewrite the safetensors index if present.
    idx_path = args.source / "model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open() as fh:
            idx = json.load(fh)
        idx["weight_map"] = new_index
        with (args.out / "model.safetensors.index.json").open("w") as fh:
            json.dump(idx, fh, indent=2)

    # Drop quantization_config and stamp the new dtype.
    cfg["torch_dtype"] = args.dtype
    with (args.out / "config.json").open("w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"done — bf16/fp16 checkpoint at {args.out.resolve()}")


if __name__ == "__main__":
    main()
