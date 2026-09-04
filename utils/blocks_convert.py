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
Draft a modular file for a standalone model, from its `# Copied from` markers.

    python utils/blocks_convert.py fnet --dry-run   # what it would inherit vs re-declare
    python utils/blocks_convert.py fnet             # write modular_fnet.py

The draft is mechanical, and deliberately so. `utils/blocks_screen.py` already decides which
markers still hold; every symbol it calls `identical` becomes `class X(Parent): pass`, and every
other top-level symbol is copied out **byte-exact by AST line range**, so the generated file can
only differ from the original where inheritance replaced a verbatim copy.

This exists because the judgement turned out to be unnecessary. Converting `megatron_bert` and
`visual_bert` by hand showed the set of classes that were AST-identical-after-rename was *exactly*
the marker set -- 8 of 8 and 7 of 7. A `ratio == 1.00` screen fully predicted the outcome, so there
is nothing left to decide: inherit the identical set, emit the rest verbatim.

Leading comments are carried with each symbol. That is not cosmetic: an earlier hand conversion
silently dropped four `# Based on ...` divergence notes because extraction started at the decorator
line, and **no gate catches it** -- symbol comparison ignores comments and a runtime check cannot
see them.

A draft is a starting point, not a result. Always:

    python utils/blocks_verify.py snapshot <model>   # before
    python utils/blocks_convert.py <model>
    python utils/modular_model_converter.py <model>
    python utils/blocks_verify.py check <model>      # 0 missing, 0 added
    ...then a runtime bit-identity check, then the model's tests.

Known limits, each of which the gate will surface rather than hide:
- a class that kept its source's name cannot inherit (it collides after renaming);
- an override calling its own `super()` makes the converter merge rather than replace, so call the
  grandparent directly instead;
- markers on a *method* rather than a class are skipped -- inheriting one drags in the parent's
  docstring, which can document arguments the class does not accept.
"""

import argparse
import ast
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_screen import MARKER, screen  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "src" / "transformers" / "models"


def _leading_comments(lines: list[str], start: int) -> int:
    """Walk back over the comment block directly above `start`, returning the real first line."""
    i = start
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        i -= 1
    return i


def _drop_markers(source: str) -> str:
    """
    Strip `# Copied from` lines from a body being carried across verbatim.

    A modular file must never contain a marker. The marker is the mechanism modular *replaces*:
    inside a modular the relationship is expressed by an import and a base class, so a copied-in
    marker is both meaningless and a claim nothing checks. Carrying them over silently seeded ten
    new modular files with legacy markers before this was caught.
    """
    kept = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith("#"):
            kept.append(line)
            continue
        marker = MARKER.search(line)
        if marker is None:
            kept.append(line)
            continue
        # Preserve any real note that happens to share the line (`# Todo - ... Copied from ...`).
        note = (
            line[: marker.start()]
            + line[marker.start() : marker.start() + line[marker.start() :].index("Copied from")]
        )
        if note.strip().strip("#").strip():
            kept.append(note.rstrip() + "\n")
    return "".join(kept)


def _span(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """The 0-based half-open line range of a top-level symbol, decorators and comments included."""
    first = node.lineno - 1
    if getattr(node, "decorator_list", None):
        first = min(first, min(d.lineno for d in node.decorator_list) - 1)
    return _leading_comments(lines, first), node.end_lineno


def draft(model: str) -> tuple[str, dict[str, str], list[str]]:
    """Return `(source, {symbol: parent}, redeclared)` for the model's drafted modular file."""
    inherit: dict[str, tuple[str, str]] = {}
    for status, symbol, src_model, src_symbol in screen(model):
        if status == "identical":
            inherit[symbol] = (src_model, src_symbol)

    paths = sorted((MODELS_ROOT / model).glob("modeling_*.py"))
    if not paths:
        raise SystemExit(f"no modeling files for {model}")
    # A model with several modeling files needs one modular per file; draft the largest only and say so.
    path = max(paths, key=lambda p: len(p.read_text()))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)

    header: list[str] = []
    body: list[str] = []
    redeclared: list[str] = []
    imports_needed: dict[str, set[str]] = {}

    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header.append(_drop_markers("".join(lines[node.lineno - 1 : node.end_lineno])))
            continue
        if name is None:
            # Module docstring, `__all__`, logger, constants: keep verbatim.
            start, end = _span(node, lines)
            body.append(_drop_markers("".join(lines[start:end])))
            continue
        if name in inherit:
            src_model, src_symbol = inherit[name]
            imports_needed.setdefault(src_model, set()).add(src_symbol)
            kind = "class" if isinstance(node, ast.ClassDef) else None
            if kind is None:
                # A copied *function* cannot be subclassed; import it and let the converter inline it.
                redeclared.append(name)
                start, end = _span(node, lines)
                body.append(_drop_markers("".join(lines[start:end])))
                continue
            body.append(f"class {name}({src_symbol}):\n    pass\n")
        else:
            redeclared.append(name)
            start, end = _span(node, lines)
            body.append(_drop_markers("".join(lines[start:end])))

    extra = [
        f"from ..{src}.modeling_{src} import {', '.join(sorted(names))}\n"
        for src, names in sorted(imports_needed.items())
    ]
    source = "".join(header) + "".join(extra) + "\n\n" + "\n\n".join(s.rstrip() + "\n" for s in body)
    return source, {k: f"{v[0]}.{v[1]}" for k, v in inherit.items()}, redeclared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--dry-run", action="store_true", help="report the plan without writing")
    args = parser.parse_args()

    source, inherit, redeclared = draft(args.model)
    print(f"{args.model}: {len(inherit)} symbols inherited, {len(redeclared)} re-declared verbatim")
    for symbol, parent in sorted(inherit.items()):
        print(f"  inherit   {symbol:42s} <- {parent}")
    if args.dry_run:
        print(f"  re-declare: {', '.join(redeclared[:12])}{' ...' if len(redeclared) > 12 else ''}")
        return 0

    out = MODELS_ROOT / args.model / f"modular_{args.model}.py"
    if out.exists():
        print(f"  {out.name} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    out.write_text(source, encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)} ({len(source.splitlines())} lines)")
    print("  now: modular_model_converter.py, then blocks_verify.py check, then the model's tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
