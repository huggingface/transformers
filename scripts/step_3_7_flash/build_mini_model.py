"""Build a mini Step-3.7-Flash hub-format checkpoint from the original code.

Weights are created with the original code
(src/transformers/models/step_3_7_flash_original/) and saved in hub checkpoint
key format so the new local code can load them via convert_step3p7_weights_to_hf.

Saves to --out (default ./mini-step-3-7-flash):
  config.json       — in original hub-checkpoint format (moe_layers_enum, not mlp_layer_types)
  pytorch_model.bin — weights in hub key format (model.embed_tokens.*, vision_model.*, ...)

Run:
    PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py [--push-to-hub]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _remote_sd_to_hub_format(state_dict: dict) -> dict:
    """Convert original-code state_dict keys → hub checkpoint key format.

    Original code layout           Hub checkpoint layout
    ─────────────────────────────  ────────────────────────────────
    model.language_model.X.*   →  model.X.*
    model.vision_model.*       →  vision_model.*
    model.vit_large_projector.* → vit_large_projector.*
    lm_head.*                  →  lm_head.*  (unchanged)
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model.language_model."):
            out[k.replace("model.language_model.", "model.", 1)] = v
        elif k.startswith("model.vision_model."):
            out[k.replace("model.vision_model.", "vision_model.", 1)] = v
        elif k.startswith("model.vit_large_projector."):
            out[k.replace("model.vit_large_projector.", "vit_large_projector.", 1)] = v
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",  type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--seed", type=int,  default=0)
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        nargs="?",
        const="itazap/Step-3.7-Flash-Mini-Original",
        help="Push hub-format checkpoint to HF Hub. Pass a repo_id or omit to use default.",
    )
    args = parser.parse_args()

    from transformers.models.step_3_7_flash_original.configuration_step3p7 import (
        Step3p7Config,
        Step3p7TextConfig,
        StepRoboticsVisionEncoderConfig,
    )
    from transformers.models.step_3_7_flash_original.modeling_step3p7 import Step3p7ForConditionalGeneration

    text_cfg = Step3p7TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=32,
        vocab_size=512,
        rms_norm_eps=1e-5,
        sliding_window=64,
        moe_intermediate_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        share_expert_dim=64,
        norm_expert_weight=True,
        moe_layers_enum=(2, 3),
        moe_router_activation="sigmoid",
        moe_router_scaling_factor=1.0,
        use_moe_router_bias=True,
        use_head_wise_attn_gate=True,
        need_fp32_gate=True,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        rope_theta=10000.0,
        max_position_embeddings=512,
        max_seq_len=512,
        attention_other_setting={"num_attention_heads": 4, "num_attention_groups": 2, "head_dim": 32},
        pad_token_id=1,
        use_rope_layers=[True, True, True, True],
        yarn_only_types=["full_attention"],
        swiglu_limits=[None, None, 1.0, 1.0],
        swiglu_limits_shared=[None, None, 1.0, 1.0],
    )
    vis_cfg = StepRoboticsVisionEncoderConfig(
        width=64, layers=2, heads=4,
        image_size=56, patch_size=14, mlp_ratio=4.0,
        hidden_act="quick_gelu",
        use_cls_token=False, use_rope2d=True,
        use_abs_posemb=True, ls_init_value=0.1,
    )
    cfg = Step3p7Config(
        text_config=text_cfg.to_dict(),
        vision_config=vis_cfg.to_dict(),
        projector_bias=False,
        image_token_id=511,  # must be < vocab_size (512)
    )

    torch.manual_seed(args.seed)
    model = Step3p7ForConditionalGeneration(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"mini Step-3.7-Flash (original code): {n / 1e6:.2f}M params")

    hub_sd = _remote_sd_to_hub_format(model.state_dict())

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in ("model.safetensors",):
        p = args.out / stale
        if p.exists():
            p.unlink()
    torch.save(hub_sd, args.out / "pytorch_model.bin")

    # Save config via save_pretrained first (captures all fields), then
    # post-process the JSON to restore original-checkpoint format: swap the
    # expanded mlp_layer_types list back to moe_layers_enum (the raw form that
    # real checkpoints ship with, before convert_config.py normalises it).
    import json
    cfg.save_pretrained(args.out)
    config_path = args.out / "config.json"
    saved = json.loads(config_path.read_text())
    text = saved.get("text_config", saved)
    mlp_layer_types = text.pop("mlp_layer_types", None)
    if mlp_layer_types:
        text["moe_layers_enum"] = [i for i, t in enumerate(mlp_layer_types) if t == "sparse"]
    config_path.write_text(json.dumps(saved, indent=2) + "\n")

    print(f"Saved {len(hub_sd)} keys → {args.out.resolve()}/")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.push_to_hub)
        print(f"Pushed → https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
