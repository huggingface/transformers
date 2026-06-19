"""Demo: load mini Step-3.7-Flash and run forward passes with debug tracing.

Three modalities are tested:
  1. Text-only
  2. pixel_values  (image run through vision encoder)
  3. image_embeds  (pre-extracted ViT features, same shape as vision encoder output)

The original-code model is run first to produce a reference; the new code is then
compared bit-exactly against it. Both sets of logits are saved to debug_path.

Build the mini checkpoint first:
  PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py

Then run this demo:
  PYTHONPATH=src python scripts/step_3_7_flash/demo.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# original_code/ and convert_config.py live next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_converted_checkpoint(src: Path) -> Path:
    """Copy checkpoint to a temp dir and run convert_config on config.json.

    The raw checkpoint (as built by build_mini_model.py, or a real original
    checkpoint) may use legacy config keys (moe_layers_enum, unnormalised
    per-layer lists, etc.) that convert_config.py normalises for the new code.
    """
    from convert_config import convert

    tmp = Path(tempfile.mkdtemp(prefix="step3p7_converted_"))
    for f in src.iterdir():
        shutil.copy2(f, tmp / f.name)
    converted = convert(src / "config.json")
    (tmp / "config.json").write_text(json.dumps(converted, indent=2) + "\n")
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--debug-path", type=str, default=str(Path(__file__).parent / "debug_output"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    debug_path = Path(args.debug_path)
    debug_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Original code (baseline) ──────────────────────────────────────────
    from original_code.configuration_step3p7 import Step3p7Config as OrigConfig
    from original_code.modeling_step3p7 import Step3p7ForConditionalGeneration as OrigModel

    print("Loading original-code model ...")
    orig_cfg = OrigConfig.from_pretrained(args.model_dir)
    orig_model = OrigModel.from_pretrained(args.model_dir, config=orig_cfg).eval()
    vis_cfg_orig = orig_cfg.vision_config
    text_cfg_orig = orig_cfg.text_config

    # The original code's from_pretrained corrupts non-persistent buffers (freqs_cache)
    # via _move_missing_keys_from_meta_to_device. Recompute them before running.
    for m in orig_model.modules():
        if hasattr(m, "_compute_2d_freqs") and hasattr(m, "freqs_cache"):
            m.register_buffer("freqs_cache", m._compute_2d_freqs(), persistent=False)

    # The original Step3p7ForConditionalGeneration.forward captures pixel_values /
    # image_embeds as explicit params but never forwards them to self.model().  Call
    # the inner model directly and apply lm_head ourselves so vision paths work.
    def _orig_run(input_ids, pixel_values=None, image_embeds=None, num_patches=None):
        out = orig_model.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_embeds=image_embeds,
            num_patches=num_patches,
            use_cache=False,
        )
        return orig_model.lm_head(out.last_hidden_state)

    torch.manual_seed(args.seed)
    input_ids_ref = torch.randint(0, text_cfg_orig.vocab_size, (1, 16))
    seq_len = 16
    ids_with_image_ref = torch.randint(0, text_cfg_orig.vocab_size, (1, seq_len))
    ids_with_image_ref[0, seq_len // 2] = orig_cfg.image_token_id
    torch.manual_seed(args.seed + 1)
    pixel_values_ref = torch.randn(1, vis_cfg_orig.num_channels, vis_cfg_orig.image_size, vis_cfg_orig.image_size)

    with torch.no_grad():
        ref_text = _orig_run(input_ids_ref)
        ref_pv   = _orig_run(ids_with_image_ref, pixel_values=pixel_values_ref, num_patches=[0])
        # For ie_logits: use the vision-encoder output of the original model as image_embeds.
        # The original code's image_embeds path is broken (2D reshape bug), so instead
        # we verify the new model's invariant: ie(vision_encoder(pv)) == pv_logits.
        ref_vis_out = orig_model.model.vision_model(pixel_values_ref)  # (B, P, C)
    torch.save(
        {"text_logits": ref_text, "pv_logits": ref_pv},
        debug_path / "logits_baseline.pt",
    )
    print(f"Baseline logits saved → {debug_path}/logits_baseline.pt")

    # ── 2. New code ───────────────────────────────────────────────────────────
    from transformers.model_debugging_utils import model_addition_debugger_context
    from transformers.models.step_3_7_flash.configuration_step3p7 import Step3p7Config
    from transformers.models.step_3_7_flash.modeling_step3p7 import Step3p7ForConditionalGeneration

    converted_dir = _make_converted_checkpoint(args.model_dir)
    print(f"\nLoading new-code model from {args.model_dir} (converted config → {converted_dir}) ...")
    cfg = Step3p7Config.from_pretrained(converted_dir)
    model = Step3p7ForConditionalGeneration.from_pretrained(converted_dir, config=cfg).eval()

    with torch.no_grad():
        new_text = model.forward(input_ids=input_ids_ref, use_cache=False).logits
        new_pv   = model.forward(input_ids=ids_with_image_ref, pixel_values=pixel_values_ref, num_patches=[0], use_cache=False).logits
        # ie path: pass raw vision-encoder output as image_embeds — must equal pv_logits
        new_ie   = model.forward(input_ids=ids_with_image_ref, image_embeds=ref_vis_out, use_cache=False).logits

    torch.save(
        {"text_logits": new_text, "pv_logits": new_pv, "ie_logits": new_ie},
        debug_path / "logits.pt",
    )
    print(f"New-code logits saved  → {debug_path}/logits.pt")

    # ── 3. Inline comparison ──────────────────────────────────────────────────
    print()
    all_pass = True
    for key, ref, got in (
        ("text_logits", ref_text,  new_text),
        ("pv_logits",   ref_pv,    new_pv),
        ("ie_logits",   new_pv,    new_ie),   # invariant: ie(vis_out) == pv
    ):
        diff = (ref - got).abs()
        mx = diff.max().item()
        mean = diff.mean().item()
        ok = mx == 0.0
        all_pass = all_pass and ok
        sym = "✓" if ok else "✗"
        print(f"  {sym} {key:<14}  max|Δ|={mx:.2e}  mean|Δ|={mean:.2e}")

    if all_pass:
        print("\nRESULT: PASS  (bit-exact)")
    else:
        print("\nRESULT: FAIL")
        sys.exit(1)

    # ── 4. Debug trace (text-only) ────────────────────────────────────────────
    print(f"\nWriting debug trace to: {debug_path}")
    with model_addition_debugger_context(model, debug_path=str(debug_path), do_prune_layers=False):
        model.forward(input_ids=input_ids_ref, use_cache=False)
    print("Debug trace written  ✓")


if __name__ == "__main__":
    main()
