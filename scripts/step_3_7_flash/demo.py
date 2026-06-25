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
    """Convert hub-format checkpoint → transformers format, save to out_dir."""
    import json
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from convert_config import convert as convert_cfg
    from transformers.models.step_3_7_flash.convert_step3p7_weights_to_hf import convert_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cfg = convert_cfg(hub_dir / "config.json")
    (out_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2) + "\n")
    num_hidden_layers = (raw_cfg.get("text_config") or raw_cfg).get("num_hidden_layers")

    for weight_file in ("pytorch_model.bin", "model.safetensors"):
        src = hub_dir / weight_file
        if src.exists():
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                hub_sd = load_file(src)
            else:
                hub_sd = torch.load(src, map_location="cpu", weights_only=True)
            break
    else:
        raise FileNotFoundError(f"No weight file found in {hub_dir}")

    new_sd = convert_state_dict(hub_sd, num_hidden_layers=num_hidden_layers)
    for stale in ("model.safetensors", "pytorch_model.bin"):
        (out_dir / stale).unlink(missing_ok=True)
    torch.save(new_sd, out_dir / "pytorch_model.bin")
    print(f"Converted {len(hub_sd)} → {len(new_sd)} keys → {out_dir}/")
    return out_dir


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
    orig_model = OrigModel.from_pretrained(hub_dir, config=orig_cfg).eval()

    with torch.no_grad():
        orig_text = orig_model(input_ids=input_ids, use_cache=False).logits
        out = orig_model.model(
            input_ids=ids_with_image,
            pixel_values=pixel_values,
            num_local_patches=[0],
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
