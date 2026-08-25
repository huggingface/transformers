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
The block manifest -- "the transformers format".

Every model's block variants, one model per line, in a single file:

    python utils/blocks_export.py                 # write it
    python utils/blocks_export.py --check-only     # exit 1 if stale

One line per model so it stays greppable in both directions: `grep '"qwen3":'` gives that model's
whole architecture at a glance, and `grep sliding_attention` gives every model that slides.

Architecture only. No class names (`Qwen3Attention` adds nothing to the `qwen3` key), no helper
hashes, no dates, no lineage. Every value is self-describing: `no_qk_norm` rather than `none`,
`norm-attn-residual-norm-mlp-residual` rather than `N A R N M R`.

Nothing derived from other models is stored either -- who owns a tag, which owner is canonical, what
the axes are called all live once in `docs/source/en/model_blocks.md`, keyed by the tag. Repeating
them here would duplicate the catalog and, being global, would make adding one model rewrite every
other model's entry.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_facets import REPO_ROOT, scan_repo  # noqa: E402


MANIFEST_PATH = REPO_ROOT / "utils" / "model_blocks.json"
# Fixed so output is deterministic; kinds absent from a model are simply omitted.
KIND_ORDER = ("attention", "mixer", "layer", "layer_other", "mlp", "moe", "rotary", "norm")


def build_manifest(blocks: list) -> dict[str, dict]:
    """`{model: {kind: tag}}` for every model that has at least one block."""
    per_model: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for block in blocks:
        per_model[block.model][block.kind].add(block.variant)

    manifest: dict[str, dict] = {}
    for model in sorted(per_model):
        entry: dict[str, str | list[str]] = {}
        for kind in KIND_ORDER:
            tags = sorted(per_model[model].get(kind, ()))
            if not tags:
                continue
            # A bare string when there is one variant, a list when a model has several towers.
            entry[kind] = tags[0] if len(tags) == 1 else tags
        if entry:
            manifest[model] = entry
    return manifest


def render(manifest: dict[str, dict]) -> str:
    """One line per model. `json.dumps(indent=2)` would spend six lines on six tags."""
    lines = [f"  {json.dumps(model)}: {json.dumps(entry)}" for model, entry in manifest.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def export_all(check_only: bool = False) -> list[Path]:
    """Write (or verify) the manifest. Returns `[path]` if it was stale, `[]` otherwise."""
    expected = render(build_manifest(scan_repo()[0]))
    current = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else None
    if current == expected:
        return []
    if not check_only:
        MANIFEST_PATH.write_text(expected, encoding="utf-8")
    return [MANIFEST_PATH]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="report staleness without writing")
    args = parser.parse_args()

    stale = export_all(args.check_only)
    if args.check_only:
        print(f"{MANIFEST_PATH.name} is " + ("STALE" if stale else "up to date"))
        return 1 if stale else 0
    manifest = build_manifest(scan_repo()[0])
    print(f"wrote {MANIFEST_PATH} ({len(manifest)} models, {MANIFEST_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
