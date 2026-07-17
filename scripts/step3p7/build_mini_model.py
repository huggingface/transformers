"""Build a mini Step-3.7-Flash hub-format checkpoint from the original code.

Weights are created with the original code
(src/transformers/models/step_3_7_flash_original/) and saved in hub checkpoint
key format so the new local code can load them via convert_step3p7_weights_to_hf.

Saves to --out (default ./mini-step-3-7-flash):
  config.json       — in original hub-checkpoint format (moe_layers_enum, not mlp_layer_types)
  pytorch_model.bin — weights in hub key format (model.embed_tokens.*, vision_model.*, ...)

Run (fully local; --push-to-hub is optional and off by default):
    PYTHONPATH=src python scripts/step3p7/build_mini_model.py
"""

from __future__ import annotations

import argparse
import re
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


def _module_path_to_hub_format(path: str) -> str:
    """Same prefix stripping as `_remote_sd_to_hub_format`, but for a bare module path (no
    trailing `.weight`/`.bias`), as used in `quantization_config.modules_to_not_convert`."""
    if path.startswith("model.language_model."):
        return "model." + path[len("model.language_model.") :]
    if path.startswith("model.vision_model."):
        return "vision_model." + path[len("model.vision_model.") :]
    if path.startswith("model.vit_large_projector"):
        return "vit_large_projector" + path[len("model.vit_large_projector") :]
    return path


_FP8_BLOCK_SIZE = 128
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_blockwise_fp8(weight: torch.Tensor, block_size: int = _FP8_BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a `(..., rows, cols)` weight to fp8 e4m3 with a per-`block_size`x`block_size`-block
    dequant scale, mirroring the real checkpoint's format (and `MoELinear._dequantize`'s reverse of
    it). `rows`/`cols` must be exact multiples of `block_size` -- true for real checkpoints and
    enforced here so the mini model actually exercises multi-block reshaping instead of the
    degenerate 1-block case.
    """
    *lead, rows, cols = weight.shape
    if rows % block_size or cols % block_size:
        raise ValueError(f"{tuple(weight.shape)} isn't a multiple of block_size={block_size}")
    scale_rows, scale_cols = rows // block_size, cols // block_size
    w = weight.float().reshape(*lead, scale_rows, block_size, scale_cols, block_size)
    amax = w.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_MAX  # dequant multiplier: fp8_value * scale ≈ original value
    fp8 = (w / scale).to(torch.float8_e4m3fn).reshape(*lead, rows, cols)
    scale = scale.reshape(*lead, scale_rows, scale_cols)
    return fp8, scale


_MOE_ROUTED_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.moe\.(up_proj|gate_proj|down_proj)\.weight$")


def _quantize_routed_moe_weights(state_dict: dict, moe_layer_indices: set[int]) -> dict:
    """FP8-quantize only the routed-expert weights (`moe.{up,gate,down}_proj`) on MoE layers --
    the one thing a real FP8 checkpoint actually quantizes for this architecture. Everything else
    (self_attn, share_expert, dense mlp, embeddings, norms, lm_head, vision) stays full precision,
    matching `modules_to_not_convert` on the real checkpoint.
    """
    out = dict(state_dict)
    for key, value in state_dict.items():
        match = _MOE_ROUTED_WEIGHT_RE.match(key)
        if match and int(match.group(1)) in moe_layer_indices:
            fp8, scale = _quantize_blockwise_fp8(value)
            out[key] = fp8
            out[key.replace(".weight", ".weight_scale_inv")] = scale
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        nargs="?",
        const="itazap/Step-3.7-Flash-Mini-Original",
        help="Push hub-format checkpoint to HF Hub. Pass a repo_id or omit to use default.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "FP8-quantize the routed MoE expert weights (the only tensors a real Step-3.7-Flash-FP8 "
            "checkpoint actually quantizes) and add a matching `quantization_config`, so the mini "
            "checkpoint exercises the same FP8 loading/dequant code paths as the real one."
        ),
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help=(
            "Use bigger matmul dimensions (hidden_size=1024, moe_intermediate_size=1280 -- the real "
            "checkpoint's own value) and more layers/experts. Bit-exact equality between original and "
            "ported code is easy at tiny size even under real bugs, since a small blocked-FP8 GEMM "
            "often takes the exact same reduction order as a plain matmul; large enough dimensions "
            "make the FP8 kernel's tiling genuinely diverge in accumulation order from the reference's "
            "dequant-then-matmul, the same way it can on the real checkpoint. Still far smaller than "
            "the real 45-layer/288-expert model, so still fast."
        ),
    )
    args = parser.parse_args()

    from transformers.models.step_3_7_flash_original.configuration_step3p7 import (
        Step3p7Config,
        Step3p7TextConfig,
        StepRoboticsVisionEncoderConfig,
    )
    from transformers.models.step_3_7_flash_original.modeling_step3p7 import Step3p7ForConditionalGeneration

    if args.large:
        num_hidden_layers = 6
        moe_layers_enum = (2, 3, 4, 5)
        text_cfg = Step3p7TextConfig(
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=16,
            num_attention_groups=4,
            head_dim=64,
            vocab_size=512,
            rms_norm_eps=1e-5,
            sliding_window=64,
            # The real checkpoint's own value (10 dequant blocks along this axis) rather than just
            # "a multiple of 128" -- see the `--large` help text for why size matters here.
            moe_intermediate_size=1280,
            moe_num_experts=16,
            moe_top_k=4,
            share_expert_dim=256,
            norm_expert_weight=True,
            moe_layers_enum=moe_layers_enum,
            moe_router_activation="sigmoid",
            moe_router_scaling_factor=1.0,
            use_moe_router_bias=True,
            use_head_wise_attn_gate=True,
            need_fp32_gate=True,
            layer_types=["full_attention"] + ["sliding_attention"] * (num_hidden_layers - 1),
            rope_theta=[5000000.0] + [10000.0] * (num_hidden_layers - 1),
            partial_rotary_factors=[0.5] + [1.0] * (num_hidden_layers - 1),
            max_position_embeddings=512,
            max_seq_len=512,
            attention_other_setting={"num_attention_heads": 24, "num_attention_groups": 4, "head_dim": 64},
            pad_token_id=1,
            use_rope_layers=[True] * num_hidden_layers,
            yarn_only_types=["full_attention"],
            swiglu_limits=[None if i not in moe_layers_enum else 1.0 for i in range(num_hidden_layers)],
            swiglu_limits_shared=[None if i not in moe_layers_enum else 1.0 for i in range(num_hidden_layers)],
        )
    else:
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
            # Multiple of the FP8 block size (128) on purpose: up_proj/gate_proj/down_proj then span
            # 3 dequant blocks along one axis instead of the degenerate 1-block case, so `--quantize`
            # actually exercises multi-block reshaping.
            moe_intermediate_size=384,
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
            # Real checkpoints give one rope_theta/partial_rotary_factor *per decoder layer* (indexed by
            # layer_idx in the original code), not one value for the whole model — distinct per-type
            # values here so a regression that collapses/misreads them is numerically detectable.
            rope_theta=[5000000.0, 10000.0, 10000.0, 10000.0],
            partial_rotary_factors=[0.5, 1.0, 1.0, 1.0],
            max_position_embeddings=512,
            max_seq_len=512,
            # Real checkpoints use a *different* head count for sliding vs. full-attention layers
            # (e.g. 96 vs 64) — mismatched here too (6 vs 4) so a regression that ignores this and
            # reuses `num_attention_heads` for every layer produces a detectable shape/logits mismatch.
            attention_other_setting={"num_attention_heads": 6, "num_attention_groups": 2, "head_dim": 32},
            pad_token_id=1,
            use_rope_layers=[True, True, True, True],
            yarn_only_types=["full_attention"],
            swiglu_limits=[None, None, 1.0, 1.0],
            swiglu_limits_shared=[None, None, 1.0, 1.0],
        )
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
        projector_bias=False,
        image_token_id=511,  # must be < vocab_size (512)
    )

    torch.manual_seed(args.seed)
    model = Step3p7ForConditionalGeneration(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"mini Step-3.7-Flash (original code): {n / 1e6:.2f}M params")

    state_dict = model.state_dict()
    quantization_config = None
    if args.quantize:
        moe_layer_indices = {
            i for i, layer in enumerate(model.model.language_model.layers) if layer.is_moe_layer
        }
        state_dict = _quantize_routed_moe_weights(state_dict, moe_layer_indices)
        modules_to_not_convert = [
            _module_path_to_hub_format(name) for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
        ]
        quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
            "modules_to_not_convert": modules_to_not_convert,
        }
        print(f"Quantized routed MoE weights on layers {sorted(moe_layer_indices)}; "
              f"{len(modules_to_not_convert)} modules kept full precision.")
        # Real FP8 checkpoints ship `"dtype": "bfloat16"` in config.json; the FP8 kernel path
        # requires bf16/fp16 activations, and without this the mini config would default to fp32.
        cfg.dtype = "bfloat16"

    hub_sd = _remote_sd_to_hub_format(state_dict)

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
    if quantization_config is not None:
        saved["quantization_config"] = quantization_config
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
