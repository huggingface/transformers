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
Symbol-level gate for modular conversions.

Converting a standalone model to modular must not change what the generated file *means*. This
compares a model's generated files symbol by symbol against a snapshot taken before the change:

    python utils/blocks_verify.py snapshot llama     # before touching anything
    ...author modular_llama.py, run the converter...
    python utils/blocks_verify.py check llama        # must report 0 missing, 0 added

Why this exists rather than reading a diff:

- **Any** overridden method that calls its own `super()` makes the converter **inline the parent's
  body and then append the child's statements** -- it merges, it does not replace. This is not
  limited to `__init__`: an `_init_weights` override written that way ran both branches, called
  `init.zeros_` on a buffer the model does not have, and dragged two unused classes into the file.
  It imported cleanly and ruff was clean. The fix is to call the grandparent directly, e.g.
  `PreTrainedModel.__init__(self, config)` or `PreTrainedModel._init_weights(self, module)`, which
  replaces the body instead of merging.
- Renaming is wider than a marker claims: `with Bert->BertGeneration` renames only `Bert`, but the
  converter also maps `bert` -> `bert_generation`, silently rewriting lowercase model names inside
  strings (checkpoint ids, `base_model_prefix`, error messages).
- A parent method's docstring is inherited unconditionally when the child's override has none, and
  can end up documenting arguments the child does not accept. There is no way to suppress it.
- The converter emits **only** what the modular declares, so an incomplete modular silently deletes
  code -- one conversion attempt dropped 565 of 655 lines and still imported.
- `git diff` is not usable for this in this repo: `diff.external` is set to difftastic, so the output
  is a structural diff that cannot be read as unified output.

Docstrings are ignored, matching the block registry: they are not what we compare on.
Order is ignored too -- the converter legitimately emits imported helpers earlier in the file.

A clean report here is necessary but **not sufficient**. Run a runtime equivalence check too --
identical `state_dict` keys, bit-identical weights under a fixed seed, bit-identical outputs. The
reason is that a stale `# Copied from` marker can name a parent that has since been refactored: in
`donut` only 8 of 20 symbols were still identical to `swin`, and inheriting on the marker's word
would have changed the `state_dict` layout while leaving the symbol *set* untouched. Two further
harmless differences also need a human read: class-attribute declaration order follows the parent,
and a parent's class-level decorator can be replaced but never removed.
"""

import argparse
import ast
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "src" / "transformers" / "models"
SNAPSHOT_DIR = REPO_ROOT / ".blocks_verify"


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


def symbol_table(path: Path) -> dict[str, str]:
    """`{top-level symbol: normalised source}` for one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    table: dict[str, str] = {}
    for index, node in enumerate(tree.body):
        name = getattr(node, "name", None)
        if name is None:
            # Module-level statements are keyed by content so a moved import does not read as a
            # change, while a removed one still does.
            name = f"<stmt:{ast.unparse(node)[:60]}>"
        table[name] = ast.unparse(_strip_docstrings(node))
    return table


def collect(model: str) -> dict[str, dict[str, str]]:
    """`{filename: symbol table}` for every `modeling_*.py` of a model."""
    return {p.name: symbol_table(p) for p in sorted((MODELS_ROOT / model).glob("modeling_*.py"))}


def cmd_snapshot(args: argparse.Namespace) -> int:
    tables = collect(args.model)
    if not tables:
        print(f"no modeling files for {args.model}", file=sys.stderr)
        return 1
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"{args.model}.json"
    path.write_text(json.dumps(tables, indent=1, sort_keys=True), encoding="utf-8")
    total = sum(len(t) for t in tables.values())
    print(f"snapshot: {args.model} -> {path.relative_to(REPO_ROOT)} ({len(tables)} files, {total} symbols)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = SNAPSHOT_DIR / f"{args.model}.json"
    if not path.exists():
        print(f"no snapshot for {args.model}; run `snapshot` before converting", file=sys.stderr)
        return 1
    before = json.loads(path.read_text(encoding="utf-8"))
    after = collect(args.model)

    failed = False
    for filename in sorted(set(before) | set(after)):
        old, new = before.get(filename, {}), after.get(filename, {})
        missing = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        differ = sorted(name for name in set(old) & set(new) if old[name] != new[name])
        status = "OK" if not (missing or added or differ) else "REVIEW"
        if missing or added:
            status = "FAIL"
            failed = True
        print(f"  {filename}: {len(old)} -> {len(new)} symbols  [{status}]")
        for label, names in (("MISSING", missing), ("ADDED", added), ("DIFFER", differ)):
            if names:
                print(f"      {label} ({len(names)}): {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    if failed:
        print("\nFAIL: symbols were lost or invented. The modular is incomplete -- it must declare")
        print("every symbol the generated file had. Revert rather than shipping this.")
        return 1
    any_differ = any(
        before.get(f, {}).get(n) != after.get(f, {}).get(n)
        for f in set(before) | set(after)
        for n in set(before.get(f, {})) & set(after.get(f, {}))
    )
    if any_differ:
        print("\nREVIEW: no symbol lost, but some bodies changed. Justify each one before shipping:")
        print("  - a declaration-order swap or an inherited class decorator is harmless")
        print("  - dead locals or reordered assignments mean a `super().__init__()` merge; call the")
        print("    grandparent directly instead")
        return 2
    print("\nPASS: same symbols, identical sources (docstrings and order ignored).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("snapshot", cmd_snapshot, "record a model's symbols before converting"),
        ("check", cmd_check, "compare a model's symbols against its snapshot"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("model", help="model directory name, e.g. blenderbot_small")
        p.set_defaults(func=func)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
