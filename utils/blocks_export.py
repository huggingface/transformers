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
Per-model block manifests -- "the transformers format".

One `blocks.json` per model directory, naming each of its blocks by *tag*: the tier-1 facet values
that decide whether the block's `forward` can be inherited.

    python utils/blocks_export.py                # write every manifest
    python utils/blocks_export.py --check-only    # exit 1 if any is stale

Deliberately minimal, and **architectural only** -- no config attributes, no dates, no lineage, not
even the model name (the directory already says it). A manifest answers exactly one question: which block variants does this model use?

No class names -- `Qwen3Attention` adds nothing to the `qwen3` directory. No helper hashes -- an
opaque `e22e87f` is not architecture. Every value is a self-describing tag: `no_qk_norm` rather than
`none`, `norm-attn-residual-norm-mlp-residual` rather than `N A R N M R`.

Everything shared lives once in `docs/source/en/model_blocks.md`, keyed by tag: who owns a tag, which
owner is canonical, what the axes are called. Repeating any of that here would both duplicate the
catalog and be *global* data -- adding one model would rewrite every other model's manifest.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_facets import MODELS_ROOT, scan_repo  # noqa: E402


DEST_NAME = "blocks.json"
# Fixed so output is deterministic; kinds absent from a model are simply omitted.
KIND_ORDER = ("attention", "mixer", "layer", "layer_other", "mlp", "moe", "rotary", "norm")
# The complete schema. Doubles as the orphan-sweep guard: a file with any other top-level key was
# not written by us and must never be deleted, so a mistyped --dest-name cannot eat real config.
ALLOWED_KEYS = frozenset(KIND_ORDER)


def manifest_for(blocks: list) -> dict:
    """One model's manifest: the distinct variant tags it uses, per block kind."""
    by_kind: dict[str, set[str]] = defaultdict(set)
    for block in blocks:
        by_kind[block.kind].add(block.variant)

    manifest: dict[str, str | list[str]] = {}
    for kind in KIND_ORDER:
        tags = sorted(by_kind.get(kind, ()))
        if not tags:
            continue
        # A bare string when there is one variant, a list when a model has several towers.
        manifest[kind] = tags[0] if len(tags) == 1 else tags
    return manifest


def render(manifest: dict) -> str:
    """One line per block kind. `json.dumps(indent=2)` would spend four lines on a single tag."""
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in manifest.items())
    return "{\n" + body + "\n}\n"


def _is_ours(path: Path) -> bool:
    """Whether `path` looks like a manifest we generated, i.e. carries no foreign top-level key."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(parsed, dict) and bool(parsed) and set(parsed) <= ALLOWED_KEYS


def export_all(dest_name: str = DEST_NAME, check_only: bool = False) -> list[Path]:
    """Write (or verify) every model's manifest. Returns the paths that were stale."""
    blocks, _ = scan_repo()

    per_model_blocks: dict[str, list] = defaultdict(list)
    for block in blocks:
        per_model_blocks[block.model].append(block)
    stale: list[Path] = []
    for model_dir in sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir()):
        path = model_dir / dest_name
        model = model_dir.name
        if not per_model_blocks.get(model):
            # No blocks: nothing to describe. Remove a manifest we previously wrote, but never a
            # file we did not write -- otherwise a mistyped --dest-name deletes real config.
            if path.exists() and _is_ours(path):
                stale.append(path)
                if not check_only:
                    path.unlink()
            continue
        expected = render(manifest_for(per_model_blocks[model]))
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != expected:
            stale.append(path)
            if not check_only:
                path.write_text(expected, encoding="utf-8")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="report stale manifests without writing")
    parser.add_argument("--dest-name", default=DEST_NAME, help="manifest filename inside each model directory")
    args = parser.parse_args()

    stale = export_all(args.dest_name, args.check_only)
    if args.check_only:
        print(f"{len(stale)} stale {args.dest_name} manifests" + (f": {[str(p) for p in stale[:5]]}" if stale else ""))
        return 1 if stale else 0
    print(f"wrote {len(stale)} {args.dest_name} manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
