"""Compare logits from two runs of the Step-3.7-Flash demo.

Usage:
    PYTHONPATH=src python scripts/step_3_7_flash/compare_outputs.py \
        --original ./scripts/step_3_7_flash/debug_output \
        --modified ./debug_step_3_7_flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--modified", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    orig_path = args.original / "logits.pt"
    mod_path  = args.modified / "logits.pt"

    for p in (orig_path, mod_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(2)

    orig = torch.load(orig_path, map_location="cpu", weights_only=True)
    mod  = torch.load(mod_path,  map_location="cpu", weights_only=True)

    all_pass = True
    for key in ("text_logits", "pv_logits", "ie_logits"):
        r, g = orig[key], mod[key]
        diff  = (r - g).abs()
        mx    = diff.max().item()
        mean  = diff.mean().item()
        ok    = mx <= args.atol
        if not ok:
            all_pass = False
        sym = "✓" if ok else "✗"
        print(f"  {sym} {key:<14}  max|Δ|={mx:.2e}  mean|Δ|={mean:.2e}  "
              f"ref={r.norm():.4f}  got={g.norm():.4f}")

    print(f"\n  atol = {args.atol:.1e}")
    if all_pass:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
