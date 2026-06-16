"""Build a mini Step-3.7-Flash checkpoint for fast iteration.

Saves a HF-format checkpoint at ``--out`` (default ``./mini-step-3-7-flash``):
  - ``config.json`` with the composite [text_config, vision_config] schema
  - ``model.safetensors`` (random weights)

Run:  ``PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py``
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch


warnings.filterwarnings("ignore")


def build_mini_config():
    from transformers.models.step_3_7_flash.configuration_step3p7 import (
        Step3p7Config,
        Step3p7TextConfig,
        StepRoboticsVisionEncoderConfig,
    )

    # 4 layers: full / sliding / sliding / sliding
    # Layers 2,3 are MoE; layers 0,1 are dense.
    num_layers = 4
    layer_types = [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]

    text_cfg = Step3p7TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=32,
        vocab_size=512,
        rms_norm_eps=1e-5,
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
        layer_types=layer_types,
        sliding_window=64,
        rope_theta=10000.0,
        max_position_embeddings=512,
        max_seq_len=512,
        # sliding attention uses the same head counts as full attention
        attention_other_setting={
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
        },
        pad_token_id=1,
        # Exercise same if-condition branches as the full model config:
        # use_rope_layers: triggers `self.use_rope = use_rope_layers[layer_idx]` in attention
        use_rope_layers=[True, True, True, True],
        # yarn_only_types: layer 0 ("full_attention") is in list → rope_parameters set;
        # layers 1-3 ("sliding_attention") not in list → rope_parameters = None
        yarn_only_types=["full_attention"],
        # swiglu_limits[2,3]=1.0 exercises clamping in MoE layers (moe_layers_enum=(2,3))
        swiglu_limits=[None, None, 1.0, 1.0],
        # swiglu_limits_shared[2,3]=1.0 exercises shared expert clamping in MoE layers
        swiglu_limits_shared=[None, None, 1.0, 1.0],
    )

    # image_size=56, patch_size=14 → 4×4 patch grid
    # two stride-2 downsamplers reduce 4→2→1 per side
    vis_cfg = StepRoboticsVisionEncoderConfig(
        width=64,
        layers=2,
        heads=4,
        image_size=56,
        patch_size=14,
        mlp_ratio=4.0,
        hidden_act="quick_gelu",
        use_cls_token=False,
        use_rope2d=True,
        use_abs_posemb=True,
        ls_init_value=0.1,
    )

    cfg = Step3p7Config(
        text_config=text_cfg.to_dict(),
        vision_config=vis_cfg.to_dict(),
        understand_projector_stride=2,
        projector_bias=False,
        image_token_id=128001,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers.models.step_3_7_flash.modeling_step3p7 import (
        Step3p7ForConditionalGeneration,
    )

    torch.manual_seed(args.seed)
    cfg = build_mini_config()
    model = Step3p7ForConditionalGeneration(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"mini Step-3.7-Flash: {n / 1e6:.2f}M params")

    # The hub modeling code uses a list for _tied_weights_keys, but this version
    # of transformers expects a dict. Patch all submodules before saving.
    for m in model.modules():
        if isinstance(getattr(m, "_tied_weights_keys", None), list):
            m._tied_weights_keys = {}

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    cfg.save_pretrained(args.out)
    print(f"Saved mini model to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
