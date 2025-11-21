#!/usr/bin/env python3
"""Generate a dry-run report and optionally apply changes for renaming `transformers`/`sarah` -> `sarah`.

Usage:
  python rename_plan/apply_patch.py --dry-run
  python rename_plan/apply_patch.py --apply --mode noncode
    python rename_plan/apply_patch.py --apply --mode full --pkg sarah

The script intentionally requires `--apply` to make changes. Without it, it only writes previews into `rename_plan/preview/`.
"""
import argparse
import subprocess
from pathlib import Path
import shutil
import re

ROOT = Path('.').resolve()
PREVIEW_DIR = ROOT / 'rename_plan' / 'preview'

NONCODE_SUFFIXES = {'.md', '.rst', '.txt', '.yml', '.yaml', '.json', '.cfg'}


def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True)


def gather_matches(term):
    # Use git grep if available, else fallback to walking files.
    try:
        out = run(f"git grep -n '{term}' || true")
        lines = [l for l in out.splitlines() if l.strip()]
    except Exception:
        lines = []
        for p in ROOT.rglob('*'):
            if p.is_file():
                try:
                    s = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if term in s:
                    for i, line in enumerate(s.splitlines(), start=1):
                        if term in line:
                            lines.append(f"{p}:{i}:{line.strip()}")
    return lines


def is_noncode_file(path: Path):
    return path.suffix.lower() in NONCODE_SUFFIXES


def propose_noncode_change_text(s: str, pkg: str):
    s2 = s.replace('transformers', pkg).replace('Sarah', pkg).replace('sarah', pkg)
    return s2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--mode', choices=['noncode', 'full'], default='noncode')
    parser.add_argument('--pkg', default='sarah')
    args = parser.parse_args()

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    t_matches = gather_matches('transformers')
    s_matches = gather_matches('sarah')

    (PREVIEW_DIR / 'transformers_matches.txt').write_text('\n'.join(t_matches), encoding='utf-8')
    (PREVIEW_DIR / 'sarah_matches.txt').write_text('\n'.join(s_matches), encoding='utf-8')

    print(f"Found {len(t_matches)} matches for 'transformers' and {len(s_matches)} for 'sarah'.")

    # Build proposed changes for non-code files first
    proposed_changes = []
    for line in t_matches + s_matches:
        try:
            path_str, lineno, content = line.split(':', 2)
        except ValueError:
            continue
        p = Path(path_str)
        if not p.exists():
            continue
        if is_noncode_file(p):
            orig = p.read_text(encoding='utf-8', errors='ignore')
            new = propose_noncode_change_text(orig, args.pkg)
            if orig != new:
                proposed_changes.append((p, orig, new))

    # Write preview diffs
    patch_lines = []
    for p, orig, new in proposed_changes:
        patch_lines.append(f"--- {p}\n+++ {p}\n")
        # Simple context: show first 5 changed lines
        orig_lines = orig.splitlines()
        new_lines = new.splitlines()
        for i, (o, n) in enumerate(zip(orig_lines, new_lines)):
            if o != n:
                patch_lines.append(f"- {o}\n+ {n}\n")
                if len(patch_lines) > 2000:
                    break
    (PREVIEW_DIR / 'proposed_changes.patch').write_text(''.join(patch_lines), encoding='utf-8')

    print(f"Wrote previews to {PREVIEW_DIR}.")

    if args.apply:
        if args.mode == 'noncode':
            # Apply only non-code replacements
            for p, orig, new in proposed_changes:
                print(f"Applying change to {p}")
                p.write_text(new, encoding='utf-8')
            print("Applied non-code changes. Review with `git status` and `git diff`.")
        else:
            # Full mode: move package and replace imports across codebase
            print("FULL mode: This will move src/transformers -> src/{pkg} and modify imports.")
            confirm = input("Type 'yes' to proceed: ")
            if confirm.strip().lower() != 'yes':
                print("Aborted by user.")
                return
            # Move package
            src_old = ROOT / 'src' / 'transformers'
            src_new = ROOT / 'src' / args.pkg
            if not src_old.exists():
                print(f"Package folder {src_old} not found. Aborting.")
                return
            shutil.move(str(src_old), str(src_new))
            # Create shim
            shim_dir = ROOT / 'src' / 'transformers'
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim_file = shim_dir / '__init__.py'
            shim_file.write_text(f"# Compatibility shim\nfrom {args.pkg} import *\n", encoding='utf-8')

            # Replace imports across repository (naive replace, review after)
            for p in ROOT.rglob('*.py'):
                if 'rename_plan' in p.parts:
                    continue
                s = p.read_text(encoding='utf-8', errors='ignore')
                s_new = re.sub(r"\bfrom\s+transformers(\S*)\s+import", lambda m: f"from {args.pkg}{m.group(1)} import", s)
                s_new = re.sub(r"\bimport\s+transformers\b", f"import {args.pkg}", s_new)
                if s != s_new:
                    p.write_text(s_new, encoding='utf-8')
            print("Full rename applied. Review changes, run tests, and commit on your branch.")

    else:
        print("Dry-run complete. Inspect files under rename_plan/preview/ before applying.")


if __name__ == '__main__':
    main()
