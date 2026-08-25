# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Repoint modular base classes at the model that introduced the variant.

    python utils/blocks_fix.py --dry-run          # what would change
    python utils/blocks_fix.py                    # apply, gated per model
    python utils/blocks_fix.py --models qwen3,olmo2

A declaration is only touched when all three hold, so the rewrite cannot change behaviour:

1. the child's tier-1 variant equals the canonical owner's -- same forward shape;
2. the child's tier-2 facets equal the canonical owner's -- no `__init__` override needed;
3. the child's canonicalised `forward` is **byte-identical** to the canonical owner's.

(3) is the one that matters. Variant equality is a lossy summary: across the library only 96 of 374
declarations that pass (1) and (2) also pass (3). For the rest the swap would rewrite real code, so
they are left alone -- "more historic" is not worth a behavioural change.

Every model is verified after regeneration by comparing generated files symbol by symbol, and
reverted if anything moved. See `utils/blocks_verify.py` for why a diff is not used.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_facets import (  # noqa: E402
    MODELS_ROOT,
    build_date_data,
    build_variants,
    forwards_match,
    modular_overrides,
    scan_repo,
    tier2_mismatch,
)
from blocks_verify import collect  # noqa: E402


FIXABLE_KINDS = {"attention", "layer", "mlp", "moe", "norm", "rotary"}


def safe_swaps(models: set[str] | None = None) -> list[tuple]:
    """`(model, class, old base model, old base class, new base model, new base class, kind)`."""
    blocks, _ = scan_repo()
    variants = build_variants(blocks)
    dates = build_date_data()
    exact = {(b.model, b.class_name): b for b in blocks}

    swaps = []
    for override in modular_overrides():
        if models and override.child_model not in models:
            continue
        child = exact.get((override.child_model, override.child_class))
        parent = exact.get((override.parent_model, override.parent_class))
        if child is None or parent is None or child.kind != parent.kind or child.kind not in FIXABLE_KINDS:
            continue
        variant = variants.get(child.tag)
        if variant is None or variant.canonical in (None, override.parent_model, child.model):
            continue
        owner = next((b for b in variant.blocks if b.model == variant.canonical), None)
        if owner is None:
            continue
        # Never point a model at something younger than itself.
        if dates.get(owner.model, "9999-99-99") >= dates.get(child.model, "9999-99-99"):
            continue
        if child.variant != parent.variant or tier2_mismatch(child, owner) or not forwards_match(child, owner):
            continue
        swaps.append(
            (
                child.model,
                child.class_name,
                override.parent_model,
                override.parent_class,
                owner.model,
                owner.class_name,
                child.kind,
            )
        )
    return swaps


def _rewrite(model: str, swaps: list[tuple]) -> str | None:
    """Point each class at its new base in the modular file. Returns the original text."""
    path = MODELS_ROOT / model / f"modular_{model}.py"
    if not path.exists():
        return None
    original = path.read_text(encoding="utf-8")
    text = original
    for _, cls, _, old_base, new_model, new_base, _ in swaps:
        # Only the base in this class's own declaration, not every mention of the name.
        pattern = rf"(class\s+{re.escape(cls)}\s*\()([^)]*\b){re.escape(old_base)}\b"
        replaced, n = re.subn(pattern, rf"\g<1>\g<2>{new_base}", text, count=1)
        if not n:
            continue
        text = replaced
        if not re.search(
            rf"^from \.\.{re.escape(new_model)}\.modeling_\w+ import .*\b{re.escape(new_base)}\b", text, re.MULTILINE
        ):
            anchor = re.search(r"^from \.\.\w+\.modeling_\w+ import", text, re.MULTILINE)
            insert = f"from ..{new_model}.modeling_{new_model} import {new_base}\n"
            text = text[: anchor.start()] + insert + text[anchor.start() :] if anchor else insert + text
    if text == original:
        return None
    path.write_text(text, encoding="utf-8")
    return original


def apply_for_model(model: str, swaps: list[tuple]) -> tuple[str, bool]:
    directory = MODELS_ROOT / model
    before = collect(model)
    original = _rewrite(model, swaps)
    if original is None:
        return f"  {model:22s} no declaration matched; skipped", False

    modular = directory / f"modular_{model}.py"
    subprocess.run(["ruff", "check", "--fix", "-q", str(modular)], capture_output=True)
    result = subprocess.run(
        [sys.executable, "utils/modular_model_converter.py", model], capture_output=True, text=True
    )
    if result.returncode != 0:
        subprocess.run(["git", "checkout", "--", str(directory)], capture_output=True)
        tail = (result.stderr or result.stdout).strip().splitlines()
        return f"  {model:22s} converter failed -> reverted ({tail[-1][:60] if tail else '?'})", False

    after = collect(model)
    problems = []
    for filename in set(before) | set(after):
        old, new = before.get(filename, {}), after.get(filename, {})
        lost = sorted(set(old) - set(new))
        moved = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        if lost or moved:
            problems.append(f"{filename}: lost={lost[:2]} changed={moved[:2]}")
    if problems:
        subprocess.run(["git", "checkout", "--", str(directory)], capture_output=True)
        return f"  {model:22s} SYMBOLS MOVED -> reverted | {problems[0][:80]}", False
    names = ", ".join(f"{c}->{nb}" for _, c, _, _, _, nb, _ in swaps)
    return f"  {model:22s} {len(swaps)} repointed, symbols identical | {names[:80]}", True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="comma-separated subset")
    parser.add_argument("--dry-run", action="store_true", help="list the swaps without touching anything")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many models")
    args = parser.parse_args()

    only = set(args.models.split(",")) if args.models else None
    swaps = safe_swaps(only)
    by_model: dict[str, list[tuple]] = defaultdict(list)
    for swap in swaps:
        by_model[swap[0]].append(swap)
    print(f"{len(swaps)} safe swaps across {len(by_model)} models")

    if args.dry_run:
        for model in sorted(by_model):
            for _, cls, om, ob, nm, nb, kind in by_model[model]:
                print(f"  {kind:9s} {model}.{cls}: {om}.{ob} -> {nm}.{nb}")
        return 0

    applied = 0
    for i, model in enumerate(sorted(by_model)):
        if args.limit and i >= args.limit:
            print(f"  ... stopping at --limit {args.limit}")
            break
        message, ok = apply_for_model(model, by_model[model])
        print(message)
        applied += ok
    print(f"\n{applied}/{min(len(by_model), args.limit or len(by_model))} models updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
