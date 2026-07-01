"""Demo: run forward passes on Step-3.7-Flash and check correctness.

Workflow
--------
1. Build mini hub-format weights and push to HF Hub (once, or after changing architecture):
     PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py --push-to-hub

2. Validate — downloads from Hub, converts keys, compares original code vs new code logits:
     PYTHONPATH=src python scripts/step_3_7_flash/demo.py --hub-dir itazap/Step-3.7-Flash-Mini-Original

3. Regression check — iterate on modular changes without rebuilding hub weights:
     PYTHONPATH=src python scripts/step_3_7_flash/demo.py            # first run creates baseline
     PYTHONPATH=src python scripts/step_3_7_flash/demo.py            # subsequent runs compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _resolve_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(model_dir)))


def _convert_hub_checkpoint(hub_dir: Path, out_dir: Path) -> Path:
    """Convert hub-format checkpoint → transformers format, save to out_dir.

    Weight renaming is applied automatically by `from_pretrained` via the
    `step3p7` entry in `conversion_mapping.py` — see `convert_checkpoint`.
    """
    from transformers.models.step_3_7_flash.convert_step3p7_weights_to_hf import convert_checkpoint

    convert_checkpoint(input_dir=str(hub_dir), output_dir=str(out_dir))
    return out_dir


def _load_raw_state_dict(hub_dir: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint's weight file as-is, with no key renaming."""
    for weight_file in ("model.safetensors", "pytorch_model.bin"):
        src = hub_dir / weight_file
        if src.exists():
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(src)
            return torch.load(src, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No weight file found in {hub_dir}")


def _apply_key_mapping(
    state_dict: dict[str, torch.Tensor], key_mapping: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Rename `state_dict` keys via ordered regex (pattern -> replacement); first match wins per key."""
    import re

    renamed = {}
    for key, value in state_dict.items():
        for pattern, replacement in key_mapping.items():
            if re.search(pattern, key):
                key = re.sub(pattern, replacement, key, count=1)
                break
        renamed[key] = value
    return renamed


def _run_original_forward(
    hub_dir: Path,
    input_ids: torch.Tensor,
    ids_with_image: torch.Tensor,
    pixel_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forward pass with step_3_7_flash_original on hub-format weights.

    Step3p7ForConditionalGeneration.forward in the original code does not
    forward pixel_values to self.model, so we call self.model directly and
    apply lm_head manually for the image path.
    """
    from transformers.models.step_3_7_flash_original.configuration_step3p7 import (
        Step3p7Config as OrigConfig,
    )
    from transformers.models.step_3_7_flash_original.modeling_step3p7 import (
        Step3p7ForConditionalGeneration as OrigModel,
    )

    print(f"Loading original-code model from {hub_dir} ...")
    orig_cfg = OrigConfig.from_pretrained(hub_dir)
    # Load the state dict directly rather than via `from_pretrained`: this vendor
    # reference copy shares both its class name and `config.model_type` ("step3p7")
    # with the new-code model, so `from_pretrained`'s automatic weight-conversion
    # lookup (keyed on those same identifiers) would additionally apply the new
    # code's key-renaming mapping on top of this class's own `_checkpoint_conversion_mapping`,
    # double-renaming keys. Apply only this class's own mapping instead.
    orig_model = OrigModel(orig_cfg)
    state_dict = _load_raw_state_dict(hub_dir)
    state_dict = _apply_key_mapping(state_dict, OrigModel._checkpoint_conversion_mapping)
    missing, unexpected = orig_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [original-code] missing={len(missing)} unexpected={len(unexpected)} keys")
    orig_model.eval()

    # Non-persistent buffers (e.g. RoPE freqs_cache) are not saved in the
    # checkpoint and may be left as uninitialized memory when from_pretrained
    # uses a meta-device init path.  Recompute them by re-running __init__
    # on every EncoderRope2D submodule.
    from transformers.models.step_3_7_flash_original.modeling_step3p7 import EncoderRope2D
    for module in orig_model.modules():
        if isinstance(module, EncoderRope2D):
            cache = module._compute_2d_freqs()
            module.register_buffer("freqs_cache", cache, persistent=False)

    with torch.no_grad():
        orig_text = orig_model(input_ids=input_ids, use_cache=False).logits
        out = orig_model.model(
            input_ids=ids_with_image,
            pixel_values=pixel_values,
            num_patches=[0],
            use_cache=False,
        )
        orig_pv = orig_model.lm_head(out.last_hidden_state)

    return {"text_logits": orig_text, "pv_logits": orig_pv}


def _compare(label: str, ref: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> bool:
    print(f"\n{label}")
    all_pass = True
    for key in ref:
        if key not in got:
            print(f"  - {key:<14}  (not in both, skipped)")
            continue
        r, g = ref[key], got[key]
        if r.isnan().any() or g.isnan().any():
            print(f"  ~ {key:<14}  NaN in output (skipped)")
            continue
        diff = (r - g).abs()
        mx = diff.max().item()
        ok = mx == 0.0
        all_pass = all_pass and ok
        print(f"  {'✓' if ok else '✗'} {key:<14}  max|Δ|={mx:.2e}  mean|Δ|={diff.mean().item():.2e}")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default=str(Path(__file__).parent / "debug_output" / "converted"),
        help="Local path or HF repo ID (transformers format). Defaults to the converted/ dir produced by --hub-dir.",
    )
    parser.add_argument(
        "--hub-dir", default=None,
        help="Hub-format checkpoint dir. Converts keys and compares orig vs new code.",
    )
    parser.add_argument("--debug-path", default=str(Path(__file__).parent / "debug_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-baseline", action="store_true")
    args = parser.parse_args()

    debug_path = Path(args.debug_path)
    debug_path.mkdir(parents=True, exist_ok=True)

    if args.hub_dir:
        hub_dir = _resolve_model_dir(args.hub_dir)
        model_dir = _convert_hub_checkpoint(hub_dir, debug_path / "converted")
    else:
        hub_dir = None
        model_dir = _resolve_model_dir(args.model_dir)

    from transformers.models.step_3_7_flash.configuration_step3p7 import Step3p7Config
    from transformers.models.step_3_7_flash.modeling_step3p7 import Step3p7ForConditionalGeneration

    print(f"Loading model from {model_dir} ...")
    cfg = Step3p7Config.from_pretrained(model_dir)
    model = Step3p7ForConditionalGeneration.from_pretrained(model_dir, config=cfg).eval()

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 16))
    ids_with_image = torch.randint(0, cfg.text_config.vocab_size, (1, 16))
    ids_with_image[0, 8] = cfg.image_token_id
    torch.manual_seed(args.seed + 1)
    pixel_values = torch.randn(1, cfg.vision_config.num_channels, cfg.vision_config.image_size, cfg.vision_config.image_size)

    with torch.no_grad():
        new_text = model(input_ids=input_ids, use_cache=False).logits
        new_pv   = model(input_ids=ids_with_image, pixel_values=pixel_values, num_local_patches=[0], use_cache=False).logits

    current = {"text_logits": new_text, "pv_logits": new_pv}
    torch.save(current, debug_path / "logits.pt")
    print(f"Logits saved → {debug_path}/logits.pt")

    if hub_dir is not None:
        orig = _run_original_forward(hub_dir, input_ids, ids_with_image, pixel_values)
        ok = _compare("Original-code vs new-code:", orig, current)
        print("\nRESULT:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)
        return

    baseline_path = debug_path / "logits_baseline.pt"
    if args.reset_baseline:
        baseline_path.unlink(missing_ok=True)
        print(f"Baseline removed: {baseline_path}")

    if not baseline_path.exists():
        torch.save(current, baseline_path)
        print(f"Baseline saved → {baseline_path}\nRESULT: PASS  (baseline created; re-run to compare)")
        return

    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    ok = _compare("Baseline comparison:", baseline, current)
    print("\nRESULT:", "PASS  (bit-exact)" if ok else "FAIL  (run with --reset-baseline to update)")
    if not ok:
        sys.exit(1)

    from transformers.model_debugging_utils import model_addition_debugger_context
    print(f"\nWriting debug trace → {debug_path}")
    with model_addition_debugger_context(model, debug_path=str(debug_path), do_prune_layers=False):
        model(input_ids=input_ids, use_cache=False)
    print("Debug trace written  ✓")


if __name__ == "__main__":
    main()
