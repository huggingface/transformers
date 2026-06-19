"""Compare logits from two runs of the Step-3.7-Flash demo.

Defaults compare the original-code baseline produced by demo.py against the
new-code output from the same run.  Pass explicit paths to compare any two runs.

Usage:
    # default: baseline vs new-code from the last demo.py run
    PYTHONPATH=src python scripts/step_3_7_flash/compare_outputs.py

    # explicit paths
    PYTHONPATH=src python scripts/step_3_7_flash/compare_outputs.py \
        --original ./scripts/step_3_7_flash/debug_output \
        --modified ./some_other_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=_HERE / "debug_output",
                        help="directory containing logits_baseline.pt (default: debug_output/)")
    parser.add_argument("--modified", type=Path, default=_HERE / "debug_output",
                        help="directory containing logits.pt (default: debug_output/)")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    # When both paths point at the same directory, compare baseline vs new-code.
    # When they differ, compare logits.pt from each directory.
    same_dir = args.original.resolve() == args.modified.resolve()
    orig_path = args.original / ("logits_baseline.pt" if same_dir else "logits.pt")
    mod_path  = args.modified / "logits.pt"

    for p in (orig_path, mod_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(2)

    orig = torch.load(orig_path, map_location="cpu", weights_only=True)
    mod  = torch.load(mod_path,  map_location="cpu", weights_only=True)

    all_pass = True
    for key in ("text_logits", "pv_logits", "ie_logits"):
        if key not in orig or key not in mod:
            print(f"  - {key:<14}  (not in both files, skipped)")
            continue
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
