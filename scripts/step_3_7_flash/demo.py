"""Demo: load mini Step-3.7-Flash and run forward passes with debug tracing.

Three modalities are tested:
  1. Text-only  (baseline for compare_outputs.py)
  2. pixel_values  (image run through vision encoder)
  3. image_embeds  (pre-extracted ViT features, same shape as vision encoder output)

Build the mini checkpoint first:
  PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py

Then run this demo:
  PYTHONPATH=src python scripts/step_3_7_flash/demo.py

Debug traces are written to ./debug_output/ (text-only only, used by compare_outputs.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--debug-path", type=str, default=str(Path(__file__).parent / "debug_output"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import model_addition_debugger_context
    from transformers.models.step_3_7_flash.configuration_step3p7 import Step3p7Config
    from transformers.models.step_3_7_flash.modeling_step3p7 import Step3p7ForConditionalGeneration

    print(f"Loading mini model from {args.model_dir} ...")
    cfg = Step3p7Config.from_pretrained(args.model_dir)
    model = Step3p7ForConditionalGeneration.from_pretrained(args.model_dir, config=cfg).eval()

    vis_cfg = cfg.vision_config
    text_cfg = cfg.text_config
    vocab_size = text_cfg.vocab_size
    image_token_id = cfg.image_token_id

    # Vision encoder dimensions derived from config
    # image_size=56, patch_size=14 → 4×4 = 16 patches per image
    # two stride-2 downsamplers collapse 4×4 → 2×2 → 1×1 → 1 token per image
    num_patches_per_image = (vis_cfg.image_size // vis_cfg.patch_size) ** 2  # 16
    vision_hidden_size = vis_cfg.width  # 64

    torch.manual_seed(args.seed)

    # ── 1. Text-only ─────────────────────────────────────────────────────────
    input_ids = torch.randint(0, vocab_size, (1, 16))

    print(f"[1/3] text-only  input_ids shape: {input_ids.shape}")
    print(f"      Writing debug trace to: {args.debug_path}")
    with model_addition_debugger_context(model, debug_path=args.debug_path, do_prune_layers=False):
        out = model.forward(input_ids=input_ids, use_cache=False)
    print(f"      logits shape: {out.logits.shape}  ✓")

    # ── 2. pixel_values ──────────────────────────────────────────────────────
    # One image, no sub-crop patches.
    # The vision encoder output has 1 token after downsampling, so we place
    # exactly 1 image_token_id in the input sequence.
    seq_len = 16
    ids_with_image = torch.randint(0, vocab_size, (1, seq_len))
    ids_with_image[0, seq_len // 2] = image_token_id  # one image slot

    pixel_values = torch.randn(1, vis_cfg.num_channels, vis_cfg.image_size, vis_cfg.image_size)
    num_patches = [0]  # no sub-crop patches for this image

    print(f"[2/3] pixel_values  pixel_values shape: {pixel_values.shape}")
    out_pv = model.forward(
        input_ids=ids_with_image,
        pixel_values=pixel_values,
        num_patches=num_patches,
        use_cache=False,
    )
    print(f"      logits shape: {out_pv.logits.shape}  ✓")

    # ── 3. image_embeds ──────────────────────────────────────────────────────
    # Pre-extracted ViT features (same shape as vision encoder output),
    # bypassing the vision encoder.  Shape: [num_images, num_patches, width].
    image_embeds = torch.randn(1, num_patches_per_image, vision_hidden_size)

    print(f"[3/3] image_embeds  image_embeds shape: {image_embeds.shape}")
    out_ie = model.forward(
        input_ids=ids_with_image,
        image_embeds=image_embeds,
        use_cache=False,
    )
    print(f"      logits shape: {out_ie.logits.shape}  ✓")

    print("\nAll forward passes OK.")


if __name__ == "__main__":
    main()

#PYTHONPATH=src python utils/modular_model_converter.py --files src/transformers/models/step_3_7_flash/modular_step3p7.py
#PYTHONPATH=src python scripts/step_3_7_flash/demo.py --model-dir ./mini-step-3-7-flash --debug-path ./debug_step_3_7_flash
#PYTHONPATH=src python scripts/step_3_7_flash/compare_outputs.py --original ./scripts/step_3_7_flash/debug_output --modified ./debug_step_3_7_flash
