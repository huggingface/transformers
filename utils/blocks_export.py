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

Two sections. `variants` defines each variant once, keyed by the model that introduced it.
`models` then names, per block kind, which of those a model matches -- a pointer, never a repeated
tag:

    "variants": {"attention": {"qwen3": "self_attention|gqa|rope_half|no_extras|qkv_split|uniform_layer|qk_norm"}}
    "models":   {"afmoe": {"attention": "qwen3", "mlp": "llama", "moe": "afmoe"}}

So `afmoe`'s attention *is* qwen3's, said once. One line per model keeps it greppable in both
directions: `grep '"afmoe":'` gives that model's whole architecture, and `grep '"attention": "qwen3"'`
gives every model sharing qwen3's attention. A model that introduced a variant points at itself.

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

from blocks_facets import REPO_ROOT, build_date_data, build_variants, scan_repo  # noqa: E402


MANIFEST_PATH = REPO_ROOT / "utils" / "model_blocks.json"
# Fixed so output is deterministic; kinds absent from a model are simply omitted.
# Ordered roughly by how much of a model's identity the block carries. `router` and `indexer`
# are separate from `moe` and `attention` on purpose: routing is chosen independently of the
# expert stack it feeds, and a sparse-attention indexer owns projections of its own.
KIND_ORDER = (
    "attention",
    "indexer",
    "mixer",
    "layer",
    "layer_other",
    "mlp",
    "moe",
    "router",
    "rotary",
    "norm",
)


def build_manifest(blocks: list) -> dict[str, dict]:
    """`{"variants": {kind: {owner: tag}}, "models": {model: {kind: owner}}}`."""
    variants = build_variants(blocks)
    dates = build_date_data()
    # Each variant is defined once, under the model that introduced it. One model can introduce two
    # variants of the same kind, though -- an encoder layer and a decoder layer (bart, blenderbot), a
    # self- and a cross-attention (llama4, gemma4), an indexer and its scorer (deepseek_v4) -- and a
    # plain `{owner: tag}` map cannot hold both. Writing them both under the same key silently
    # dropped 22 variants and left the models that used them pointing at the wrong tag. So: keep the
    # oldest holder when that name is still free, otherwise fall back to the next-oldest holder, and
    # only when a variant has a single holder disambiguate it with a `#n` suffix.
    owner_of: dict[str, str] = {}
    definitions: dict[str, dict[str, str]] = defaultdict(dict)
    holders_of: dict[str, list[str]] = {
        v.tag: sorted({b.model for b in v.blocks}, key=lambda m: (dates.get(m, "9999-99-99"), m))
        for v in variants.values()
    }
    ordered_variants = sorted(
        (v for v in variants.values() if v.canonical is not None),
        key=lambda v: (v.kind, dates.get(v.canonical, "9999-99-99"), v.variant),
    )
    for variant in ordered_variants:
        taken = definitions[variant.kind]
        owner = next((m for m in holders_of[variant.tag] if m not in taken), None)
        if owner is None:
            base, i = variant.canonical, 2
            while f"{base}#{i}" in taken:
                i += 1
            owner = f"{base}#{i}"
        owner_of[variant.tag] = owner
        taken[owner] = variant.variant

    per_model: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for block in blocks:
        owner = owner_of.get(block.tag)
        if owner:
            per_model[block.model][block.kind].add(owner)

    models: dict[str, dict] = {}
    for model in sorted(per_model):
        entry: dict[str, str | list[str]] = {}
        for kind in KIND_ORDER:
            owners = sorted(per_model[model].get(kind, ()))
            if not owners:
                continue
            # A bare string when there is one variant, a list when a model has several towers.
            entry[kind] = owners[0] if len(owners) == 1 else owners
        if entry:
            models[model] = entry
    # Variants in the order they entered the library, oldest first: the vocabulary reads as the
    # architecture's history rather than as an alphabet.
    ordered = {
        kind: dict(sorted(definitions[kind].items(), key=lambda kv: (dates.get(kv[0], "9999-99-99"), kv[0])))
        for kind in KIND_ORDER
        if definitions.get(kind)
    }
    return {"variants": ordered, "models": models}


def render(manifest: dict) -> str:
    """One line per variant and per model. `json.dumps(indent=2)` would spend six lines on six tags."""
    out = ["{", '  "variants": {']
    kinds = list(manifest["variants"])
    for i, kind in enumerate(kinds):
        out.append(f"    {json.dumps(kind)}: {{")
        items = list(manifest["variants"][kind].items())
        for j, (owner, tag) in enumerate(items):
            out.append(f"      {json.dumps(owner)}: {json.dumps(tag)}" + ("," if j < len(items) - 1 else ""))
        out.append("    }" + ("," if i < len(kinds) - 1 else ""))
    out += ["  },", '  "models": {']
    models = list(manifest["models"].items())
    for i, (model, entry) in enumerate(models):
        out.append(f"    {json.dumps(model)}: {json.dumps(entry)}" + ("," if i < len(models) - 1 else ""))
    out += ["  }", "}"]
    return "\n".join(out) + "\n"


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
    n_variants = sum(len(v) for v in manifest["variants"].values())
    print(
        f"wrote {MANIFEST_PATH} ({len(manifest['models'])} models, "
        f"{n_variants} variants, {MANIFEST_PATH.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
