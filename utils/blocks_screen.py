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
Pre-screen a standalone model before converting it to modular.

    python utils/blocks_screen.py                 # rank every candidate
    python utils/blocks_screen.py donut           # one model, per-symbol detail

Every `# Copied from` marker is a claim: "this symbol is a copy of that one". The claim is often
stale -- the source has been refactored since the copy was taken. In `donut` only 8 of 20 symbols
were still identical to `swin`, and inheriting on the marker's word would have changed the model's
`state_dict` layout while leaving the symbol *set* intact, so an AST-symbol gate would not have
noticed. This checks each claim against the current source before anyone writes a modular file.

Three outcomes per marker:

- `identical`  -- the copy still matches under the converter's renaming. Safe to inherit.
- `drifted`    -- the source exists but has changed. Must be re-declared, or the parent is wrong.
- `dangling`   -- the source symbol no longer exists. The marker is a lie and CI never noticed if it
                  is one of the 109 that `check_copies.py` cannot see (its regex anchors at line
                  start, so `# Todo - ... Copied from ...` is an ordinary comment).

The identical ratio is the feasibility signal: a high ratio means a mechanical conversion, a low one
means the claimed parent has moved on and the model is better left alone.

**Calibration, and the limitation that follows from it.** `check_copies.py` passes on 1021 enforced
markers, of which this screen calls 214 `drifted` -- roughly a 21% false-drift rate. It is the
stricter of the two: `check_copies` applies rename patterns properly, honours `# Ignore copy`, and
compares formatted source, whereas this compares unparsed AST after a regex rename. So:

- `identical` is trustworthy, and `ratio == 1.00` is a genuinely safe shortlist.
- `drifted` means "look at this", not "this has drifted".
- `dangling` is exact -- the symbol either exists or it does not -- and is the finding this tool
  exists for. All 30 dangling markers found on first run were invisible to `check_copies`, every one
  of them carrying text before "Copied from"; that prefix is precisely what made them unenforced,
  so they rotted when the source classes were deleted.

For enforced markers, prefer `check_copies.py` as the authority. Use this for the CI-blind ones and
for ranking conversion candidates.
"""

import argparse
import ast
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "src" / "transformers" / "models"

# `# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BlenderbotSmall, BART->BLENDERBOT_SMALL`
# Case-insensitive: 8 markers in the library are spelled `# copied from`, and a stale lowercase
# one is exactly the sort of false claim worth catching.
MARKER = re.compile(
    r"#.*copied from transformers\.models\.(?P<model>\w+)\.\w+\.(?P<symbol>\w+)(?:\s+with\s+(?P<renames>[^#\n]+))?",
    re.IGNORECASE,
)
ENFORCED = re.compile(r"^\s*#\s*Copied from transformers\.")  # check_copies.py is case-sensitive


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            if isinstance(body[0].value.value, str):
                sub.body = body[1:] or [ast.Pass()]
    return node


def _symbols(path: Path) -> dict[str, ast.AST]:
    return {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body if hasattr(n, "name")}


def _all_symbols(model: str) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for path in sorted((MODELS_ROOT / model).glob("modeling_*.py")):
        out.update(_symbols(path))
    return out


def _apply_renames(source: str, renames: str | None) -> str:
    """
    Apply a marker's `with A->B` clauses, plus the lowercase mapping the converter adds implicitly.

    That implicit part matters: `with Bert->BertGeneration` renames only `Bert`, but the converter
    also maps `bert` -> `bert_generation`, which silently rewrites lowercase model names inside
    strings such as `base_model_prefix`.
    """
    for old, new in re.findall(r"(\w+)\s*->\s*(\w+)", renames or ""):
        source = re.sub(rf"\b{re.escape(old)}", new, source)
        if old.lower() != old:
            source = re.sub(rf"\b{re.escape(old.lower())}", new.lower(), source)
    return source


def screen(model: str) -> list[tuple[str, str, str, str]]:
    """`(status, symbol, source model, source symbol)` for every marker in this model."""
    results = []
    mine = _all_symbols(model)
    source_cache: dict[str, dict[str, ast.AST]] = {}

    for path in sorted((MODELS_ROOT / model).glob("modeling_*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        pending = None
        for line in lines:
            match = MARKER.search(line)
            if match:
                pending = match
                continue
            declared = re.match(r"(?:class|def)\s+(\w+)", line)
            if not declared:
                continue
            if pending is None:
                continue
            src_model, src_symbol = pending.group("model"), pending.group("symbol")
            renames, mine_name = pending.group("renames"), declared.group(1)
            pending = None
            if src_model not in source_cache:
                source_cache[src_model] = _all_symbols(src_model)
            theirs = source_cache[src_model].get(src_symbol)
            if theirs is None:
                results.append(("dangling", mine_name, src_model, src_symbol))
                continue
            ours = mine.get(mine_name)
            if ours is None:
                results.append(("dangling", mine_name, src_model, src_symbol))
                continue
            renamed = _apply_renames(ast.unparse(_strip_docstrings(theirs)), renames)
            # Normalise the target's own name too: the marker's rename map need not cover it.
            a = re.sub(rf"\b{re.escape(mine_name)}\b", "X", ast.unparse(_strip_docstrings(ours)))
            b = re.sub(rf"\b{re.escape(mine_name)}\b", "X", renamed)
            results.append(("identical" if a == b else "drifted", mine_name, src_model, src_symbol))
    return results


def candidates() -> list[str]:
    """Models with at least one marker in a file no modular generates."""
    out = []
    for directory in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        for path in directory.glob("modeling_*.py"):
            text = path.read_text(encoding="utf-8")
            if "This file was automatically generated from" in text[:2000]:
                continue
            if MARKER.search(text):
                out.append(directory.name)
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="one model; omit to rank every candidate")
    parser.add_argument("--min-ratio", type=float, default=0.0, help="only list models at or above this ratio")
    args = parser.parse_args()

    if args.model:
        rows = screen(args.model)
        counts = Counter(status for status, *_ in rows)
        print(f"{args.model}: {len(rows)} markers -> {dict(counts)}")
        for status, symbol, src_model, src_symbol in sorted(rows):
            print(f"  {status:10s} {symbol:40s} <- {src_model}.{src_symbol}")
        return 0

    ranked = []
    for model in candidates():
        rows = screen(model)
        if not rows:
            continue
        counts = Counter(status for status, *_ in rows)
        sources = {src for _, _, src, _ in rows}
        ratio = counts["identical"] / len(rows)
        ranked.append((ratio, len(rows), counts, sources, model))

    ranked.sort(key=lambda r: (-r[0], -r[1]))
    print(f"{len(ranked)} candidate models\n")
    print(f"{'ratio':>6} {'mark':>5} {'ident':>6} {'drift':>6} {'dangl':>6}  {'model':24s} sources")
    shown = 0
    for ratio, total, counts, sources, model in ranked:
        if ratio < args.min_ratio:
            continue
        shown += 1
        src = ",".join(sorted(sources)[:3]) + ("..." if len(sources) > 3 else "")
        print(
            f"{ratio:6.2f} {total:5d} {counts['identical']:6d} {counts['drifted']:6d} "
            f"{counts['dangling']:6d}  {model:24s} {src}"
        )
    total_dangling = sum(c["dangling"] for _, _, c, _, _ in ranked)
    print(f"\n  shown {shown}/{len(ranked)}; dangling markers across all candidates: {total_dangling}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
