"""Compare debug outputs from two runs of the Step-3.7-Flash demo.

Usage:
    PYTHONPATH=src python scripts/step_3_7_flash/compare_outputs.py \
        --original ./debug_baseline \
        --modified ./debug_step_3_7_flash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FILENAME = "Step3p7ForConditionalGeneration_debug_tree_FULL_TENSORS.json"


def load_tensors(directory: Path) -> dict:
    path = directory / FILENAME
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        return json.load(f)


def _collect_leaves(obj, prefix=""):
    """Recursively collect all leaf numeric values keyed by their JSON path."""
    if isinstance(obj, (int, float)):
        yield prefix, obj
    elif isinstance(obj, list):
        # Could be a flat list of numbers (a tensor) or a nested structure
        if obj and isinstance(obj[0], (int, float)):
            yield prefix, obj
        else:
            for i, v in enumerate(obj):
                yield from _collect_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _collect_leaves(v, f"{prefix}.{k}" if prefix else k)


def compare(original: dict, modified: dict, atol: float = 1e-4):
    mismatches = []
    missing = []

    for key, orig_val in _collect_leaves(original):
        # Find matching key in modified
        parts = key.split(".")
        mod_val = modified
        try:
            for part in parts:
                if "[" in part:
                    name, rest = part.split("[", 1)
                    idx = int(rest.rstrip("]"))
                    if name:
                        mod_val = mod_val[name]
                    mod_val = mod_val[idx]
                else:
                    mod_val = mod_val[part]
        except (KeyError, IndexError, TypeError):
            missing.append(key)
            continue

        # Compare lists of numbers vs single numbers
        if isinstance(orig_val, list) and isinstance(mod_val, list):
            if len(orig_val) != len(mod_val):
                mismatches.append((key, "shape mismatch", len(orig_val), len(mod_val)))
                continue
            abs_diffs = [abs(a - b) for a, b in zip(orig_val, mod_val)]
            rel_diffs = [d / (abs(a) + 1e-8) for d, a in zip(abs_diffs, orig_val)]
            max_abs = max(abs_diffs) if abs_diffs else 0.0
            max_rel = max(rel_diffs) if rel_diffs else 0.0
            if max_abs > atol:
                mismatches.append((key, "value mismatch", max_abs, max_rel))
        elif isinstance(orig_val, (int, float)) and isinstance(mod_val, (int, float)):
            abs_diff = abs(orig_val - mod_val)
            rel_diff = abs_diff / (abs(orig_val) + 1e-8)
            if abs_diff > atol:
                mismatches.append((key, "value mismatch", abs_diff, rel_diff))

    return mismatches, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--modified", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    original = load_tensors(args.original)
    modified = load_tensors(args.modified)

    mismatches, missing = compare(original, modified, atol=args.atol)

    if missing:
        print(f"\nMISSING keys in modified ({len(missing)}):")
        for k in missing[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if mismatches:
        print(f"\nMISMATCHED tensors (abs diff > {args.atol}) — {len(mismatches)} total:")
        header = f"{'Key':<60} {'Type':<16} {'AbsDiff':>12} {'RelDiff':>12}"
        print(header)
        print("-" * len(header))
        for key, kind, abs_diff, rel_diff in mismatches[:50]:
            if kind == "shape mismatch":
                print(f"{key:<60} {'shape':>16} {'orig=' + str(abs_diff):>12} {'mod=' + str(rel_diff):>12}")
            else:
                print(f"{key:<60} {kind:<16} {abs_diff:>12.6f} {rel_diff:>12.6f}")
        if len(mismatches) > 50:
            print(f"  ... and {len(mismatches) - 50} more mismatches")
        print("\nRESULT: FAIL")
        sys.exit(1)
    else:
        print(f"All tensors match within atol={args.atol}.")
        print("RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
