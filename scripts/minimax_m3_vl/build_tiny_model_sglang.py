"""Build a tiny MiniMax M3 VL HF checkpoint matching the layout sglang expects.

Generates a checkpoint compatible with ``sglang.srt.models.minimax_m3_vl``'s
``MiniMaxM3SparseForConditionalGeneration``, so we can run a forward pass on
both backends and diff outputs without loading the 100B-param real model.

The state_dict produced here uses the M3 VL on-disk naming:
  - language_model.model.embed_tokens.weight
  - language_model.model.layers.{i}.{self_attn,mlp,...}
  - language_model.lm_head.weight
  - vision_tower.vision_model.{embeddings,layers,post_layernorm}
  - vision_tower.multi_modal_projector.{linear_1,linear_2}
  - vision_tower.patch_merge_mlp.{linear_1,linear_2}

It does NOT depend on sglang at build time; sglang loads via
``AutoConfig.from_pretrained(..., trust_remote_code=True)`` and a custom
``load_weights`` that maps these keys onto the FusedMoE-style internal layout.

Run:  ``PYTHONPATH=src python scripts/minimax_m3_vl/build_tiny_model_sglang.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import warnings

warnings.filterwarnings("ignore")


def _expert_weights(num_experts: int, hidden_size: int, intermediate_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(num_experts, intermediate_size, hidden_size, generator=g) * 0.02
    w3 = torch.randn(num_experts, intermediate_size, hidden_size, generator=g) * 0.02
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, generator=g) * 0.02
    return w1, w2, w3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("./tiny-minimax-m3-vl"),
                        help="Tiny transformers checkpoint to re-key")
    parser.add_argument("--out", type=Path, default=Path("./tiny-minimax-m3-vl-sglang"))
    args = parser.parse_args()

    # Load transformers tiny model, then re-key state_dict to the sglang/HF
    # on-disk layout (block_sparse_moe.experts.{i}.{w1,w2,w3} etc.).
    from safetensors.torch import load_file, save_file

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.source)
    state_dict = load_file(args.source / "model.safetensors")

    text_cfg = cfg.text_config
    n_experts = text_cfg.num_local_experts
    inter = text_cfg.intermediate_size
    hidden = text_cfg.hidden_size

    remapped: dict[str, torch.Tensor] = {}

    def add(key: str, tensor: torch.Tensor) -> None:
        if key in remapped:
            raise ValueError(f"duplicate key {key}")
        remapped[key] = tensor.contiguous()

    for key, val in state_dict.items():
        new_key = key
        # Strip the LlavaModel-style ``model.`` prefix.
        if new_key.startswith("model.language_model."):
            new_key = "language_model.model." + new_key[len("model.language_model."):]
        elif new_key.startswith("model.vision_tower."):
            new_key = "vision_tower.vision_model." + new_key[len("model.vision_tower."):]
        elif new_key.startswith("model.multi_modal_projector."):
            new_key = "vision_tower.multi_modal_projector." + new_key[len("model.multi_modal_projector."):]
        elif new_key.startswith("model.patch_merge_mlp."):
            new_key = "vision_tower.patch_merge_mlp." + new_key[len("model.patch_merge_mlp."):]
        elif new_key == "lm_head.weight":
            new_key = "language_model.lm_head.weight"

        # Expand the packed Mixtral-style experts (gate_up_proj/down_proj) to
        # per-expert w1/w2/w3, which the sglang loader expects.
        if ".mlp.experts.gate_up_proj" in new_key:
            base = new_key.replace(".mlp.experts.gate_up_proj", "")
            # val: [num_experts, 2*intermediate, hidden]
            gate, up = val.chunk(2, dim=1)
            for e in range(n_experts):
                add(f"{base}.mlp.experts.{e}.w1.weight", gate[e])
                add(f"{base}.mlp.experts.{e}.w3.weight", up[e])
            continue
        if ".mlp.experts.down_proj" in new_key:
            base = new_key.replace(".mlp.experts.down_proj", "")
            for e in range(n_experts):
                add(f"{base}.mlp.experts.{e}.w2.weight", val[e])
            continue

        # Shared expert MLP -> ``mlp.shared_experts.{gate_proj,down_proj,up_proj}``.
        # In transformers we packed it; expand to gate_proj/up_proj here.
        if ".mlp.shared_experts.gate_up_proj.weight" in new_key:
            base = new_key.replace(".mlp.shared_experts.gate_up_proj.weight", "")
            gate, up = val.chunk(2, dim=0)
            add(f"{base}.mlp.shared_experts.gate_proj.weight", gate)
            add(f"{base}.mlp.shared_experts.up_proj.weight", up)
            continue
        if ".mlp.shared_experts.down_proj.weight" in new_key:
            new_key = new_key  # unchanged
        # Same expansion for the dense MLP layers.
        if ".mlp.gate_up_proj.weight" in new_key:
            base = new_key.replace(".mlp.gate_up_proj.weight", "")
            gate, up = val.chunk(2, dim=0)
            add(f"{base}.mlp.gate_proj.weight", gate)
            add(f"{base}.mlp.up_proj.weight", up)
            continue

        add(new_key, val)

    # Write the re-keyed checkpoint and the same config.json so sglang loads it.
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(remapped, args.out / "model.safetensors")
    cfg.save_pretrained(args.out)

    # Stamp the architectures field expected by sglang's MiniMaxM3 VL entry.
    cfg_path = args.out / "config.json"
    with cfg_path.open() as fh:
        config_json = json.load(fh)
    config_json["architectures"] = ["MiniMaxM3SparseForConditionalGeneration"]
    with cfg_path.open("w") as fh:
        json.dump(config_json, fh, indent=2)

    print(f"sglang-layout tiny checkpoint: {args.out.resolve()}")
    print(f"keys: {len(remapped)} tensors")


if __name__ == "__main__":
    main()
