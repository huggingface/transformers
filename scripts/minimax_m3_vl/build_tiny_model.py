"""Build a tiny MiniMax M3 VL checkpoint for fast iteration.

Saves a HF-format checkpoint at ``--out`` (default ``./tiny-minimax-m3-vl``):
  - ``config.json`` with the composite [text_config, vision_config] schema
  - ``model.safetensors`` (random weights)
  - Empty tokenizer placeholder so AutoProcessor can load when extended later

The same on-disk format is what the sglang tiny model loader will consume.
Run:  ``PYTHONPATH=src python scripts/minimax_m3_vl/build_tiny_model.py``
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import warnings

warnings.filterwarnings("ignore")


def build_tiny_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
        MiniMaxM3VLTextConfig,
        MiniMaxM3VLVisionConfig,
    )

    text_cfg = MiniMaxM3VLTextConfig(
        hidden_size=128,
        intermediate_size=64,
        dense_intermediate_size=256,
        shared_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=512,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=[0, 0, 1, 1],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 32,
            "sparse_num_index_heads": 2,
            "sparse_topk_blocks": 4,
            "sparse_block_size": 8,
            "sparse_disable_index_value": [0, 0, 1, 1],
            "sparse_attention_freq": [0, 0, 1, 1],
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
        rotary_dim=16,
        partial_rotary_factor=0.5,
        max_position_embeddings=512,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
    )
    vis_cfg = MiniMaxM3VLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=56,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
    )
    cfg = MiniMaxM3VLConfig(
        text_config=text_cfg.to_dict(),
        vision_config=vis_cfg.to_dict(),
        projector_hidden_size=128,
        image_token_index=300,
        video_token_index=301,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./tiny-minimax-m3-vl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLForConditionalGeneration,
    )

    torch.manual_seed(args.seed)
    cfg = build_tiny_config()
    model = MiniMaxM3VLForConditionalGeneration(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Tiny MiniMax M3 VL: {n / 1e6:.2f}M params")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    cfg.save_pretrained(args.out)
    print(f"Saved tiny model to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
