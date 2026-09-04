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
Block-variant registry for Transformers.

Every attention-bearing block in the library reduces to one of a small number of *variants*: 2000+
block classes collapse to under 200. Two models sharing a variant have an inheritable `forward`,
so one of them should be inheriting from the other. This tool names the variants and finds the
places where that is not happening.

    python utils/blocks_cli.py compile                 # the variant catalog: what already exists
    python utils/blocks_cli.py compile --markdown      # regenerate the catalog doc
    python utils/blocks_cli.py scan llama              # facets of one model, or of a file path
    python utils/blocks_cli.py lint                    # duplicate / wrong-parent / anachronism
    python utils/blocks_cli.py lint --models qwen3,olmo3 --rules R1

This is an internal repo tool: it reads `modular_*.py` files and model cards, neither of which
ships in a release wheel.
"""

import argparse
import itertools
import statistics
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_facets import (  # noqa: E402
    MODELS_ROOT,
    TIER1_AXES,
    Block,
    ancestors,
    build_date_data,
    build_variants,
    copied_from_sources,
    forwards_match,
    generates_modeling,
    measure_axis_costs,
    modular_class_edges,
    modular_overrides,
    modular_parents,
    scan_file,
    scan_repo,
    tier2_mismatch,
)


REPORTED_KINDS = ("attention", "layer", "mlp", "moe", "rotary", "norm", "mixer", "layer_other")

# Median LoC an override of each block actually costs, measured over the modular text models. Used
# to rank findings so the report leads with what is worth fixing.
OVERRIDE_COST = {"attention": 52, "layer": 33, "moe": 40, "mlp": 12, "rotary": 8, "norm": 3}
DEFAULT_MIN_COST = 10


def _table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)), line]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------
# compile
# --------------------------------------------------------------------------------------------------
def cmd_compile(args: argparse.Namespace) -> int:
    blocks, helpers = scan_repo()
    variants = build_variants(blocks)
    dates = build_date_data()

    print(f"{len(blocks)} blocks and {len(helpers)} helper definitions -> {len(variants)} variants\n")
    for kind in REPORTED_KINDS:
        kind_variants = sorted(
            (v for v in variants.values() if v.kind == kind),
            key=lambda v: (dates.get(v.canonical or "", "9999-99-99"), v.canonical or ""),
        )
        if not kind_variants:
            continue
        singletons = sum(1 for v in kind_variants if len(v.owners) == 1)
        print(f"== {kind}: {len(kind_variants)} variants, {singletons} used by a single model")
        print(f"   axes: {' | '.join(TIER1_AXES.get(kind, ('?',)))}")
        rows = [
            (len(v.owners), v.canonical or "?", dates.get(v.canonical or "", "?"), v.variant)
            for v in kind_variants[: args.top]
        ]
        print(_table(rows, ("models", "canonical", "since", "variant")))
        if len(kind_variants) > args.top:
            print(f"   ... {len(kind_variants) - args.top} more variants")
        print()

    helper_groups = defaultdict(lambda: defaultdict(set))
    for helper in helpers:
        helper_groups[helper.name][helper.variant].add(helper.model)
    print("== module-level helpers")
    rows = []
    for name, by_variant in sorted(helper_groups.items(), key=lambda kv: -sum(len(m) for m in kv[1].values())):
        biggest = max(by_variant.values(), key=len)
        oldest = min((dates.get(m, "9999-99-99"), m) for m in biggest)[1]
        rows.append((sum(len(m) for m in by_variant.values()), len(by_variant), oldest, name))
    print(_table(rows, ("definitions", "variants", "canonical", "helper")))

    if args.markdown:
        path = Path(args.markdown)
        path.write_text(_render_markdown(variants, helper_groups, dates), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def _render_markdown(variants, helper_groups, dates) -> str:
    lines = [
        "<!-- Generated by `python utils/blocks_cli.py compile --markdown`. Do not edit by hand. -->",
        "# Block variants",
        "",
        "Every attention-bearing block in the library, grouped by *variant*: the tier-1 facets that",
        "decide whether its `forward` can be inherited. Two models in the same row have the same",
        "`forward`, so one should be inheriting it from the other -- the `canonical` column names the",
        "model that introduced the variant first.",
        "",
    ]
    for kind in REPORTED_KINDS:
        # Oldest first: the table then reads as the order these shapes entered the library.
        kind_variants = sorted(
            (v for v in variants.values() if v.kind == kind),
            key=lambda v: (dates.get(v.canonical or "", "9999-99-99"), v.canonical or ""),
        )
        if not kind_variants:
            continue
        lines += [f"## {kind}", "", f"Axes: `{' | '.join(TIER1_AXES.get(kind, ('?',)))}`", ""]
        lines += ["| models | canonical | since | variant | owners |", "|---|---|---|---|---|"]
        for v in kind_variants:
            owners = ", ".join(v.owners[:12]) + (f" (+{len(v.owners) - 12})" if len(v.owners) > 12 else "")
            lines.append(
                f"| {len(v.owners)} | `{v.canonical}` | {dates.get(v.canonical or '', '?')} | `{v.variant}` | {owners} |"
            )
        lines.append("")
    lines += ["## Module-level helpers", "", "| definitions | variants | canonical | helper |", "|---|---|---|---|"]
    for name, by_variant in sorted(helper_groups.items(), key=lambda kv: -sum(len(m) for m in kv[1].values())):
        biggest = max(by_variant.values(), key=len)
        oldest = min((dates.get(m, "9999-99-99"), m) for m in biggest)[1]
        lines.append(f"| {sum(len(m) for m in by_variant.values())} | {len(by_variant)} | `{oldest}` | `{name}` |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------------------
def cmd_scan(args: argparse.Namespace) -> int:
    target = Path(args.target)
    if target.is_file():
        paths = [(target, target.parent.name)]
    else:
        model_dir = MODELS_ROOT / args.target
        if not model_dir.is_dir():
            print(f"no such model or file: {args.target}", file=sys.stderr)
            return 1
        paths = [(p, args.target) for p in sorted(model_dir.glob("modeling_*.py"))]

    for path, model in paths:
        blocks, helpers = scan_file(path, model)
        print(f"\n=== {path}")
        rows = [
            (b.class_name, b.kind, b.variant, ", ".join(f"{k}={v}" for k, v in sorted(b.tier2.items())) or "-")
            for b in blocks
        ]
        if rows:
            print(_table(rows, ("class", "kind", "variant (tier 1)", "tier 2")))
        for helper in helpers:
            print(f"  helper {helper.name} -> {helper.variant}")
    return 0


# --------------------------------------------------------------------------------------------------
# lint
# --------------------------------------------------------------------------------------------------
def _best_match(block: Block, variant_blocks: list[Block], model_dates) -> Block | None:
    """
    The oldest block sharing `block`'s variant that `block` could actually reuse.

    Returns `None` when *any* model holding this variant is already an ancestor or a descendant:
    the reuse is in place, so there is nothing to report. Checking only the oldest holder was wrong
    -- a model correctly inheriting the variant from `qwen2` would still get flagged because
    `llama` happens to be older and also holds it.
    """
    holders = [b for b in variant_blocks if b.model != block.model]
    if not holders:
        return None
    # Ancestry is checked against *every* holder, before any filtering: if the variant is already
    # reachable through inheritance the reuse is in place, whatever role that ancestor plays.
    related = ancestors(block.model)
    if any(b.model in related or block.model in ancestors(b.model) for b in holders):
        return None
    # You cannot inherit from a model that did not exist yet. Without this the suggestions come out
    # circular -- gpt_oss told to use granite_swa and granite_swa told to use gpt_oss.
    my_date = model_dates.get(block.model, "9999-99-99")
    candidates = [
        b
        for b in holders
        if model_dates.get(b.model, "9999-99-99") < my_date
        # Facets nominated this candidate; only an identical forward confirms it.
        and forwards_match(block, b)
    ]
    if not candidates:
        return None
    # Prefer the candidate whose `__init__` also matches, then the oldest. Sorting by date alone
    # suggested `vit.ViTMLP` (2020) over `blip_2.Blip2MLP` (2023) for instructblipvideo: both have
    # the same forward, but only blip_2 matches the init too, so only blip_2 gives a diff-neutral
    # refactor. Oldest-first still decides among equally close candidates, keeping lineage historic.
    return min(
        candidates,
        key=lambda b: (tier2_mismatch(block, b), model_dates.get(b.model, "9999-99-99"), b.model),
    )


def cmd_lint(args: argparse.Namespace) -> int:
    rules = set(args.rules.split(",")) if args.rules else {"R1", "R2", "R3"}
    only = set(args.models.split(",")) if args.models else None

    blocks, helpers = scan_repo()
    variants = build_variants(blocks)
    parents = modular_parents()
    dates = build_date_data()
    findings = []

    # R1: same variant, no inheritance relation in either direction.
    for variant in variants.values():
        if variant.kind not in OVERRIDE_COST:
            continue
        cost = OVERRIDE_COST[variant.kind]
        if cost < args.min_cost:
            continue
        for block in variant.blocks:
            if only and block.model not in only:
                continue
            match = _best_match(block, variant.blocks, dates)
            if match is None:
                continue
            if "R1" in rules:
                findings.append(
                    (
                        cost,
                        "R1",
                        block,
                        f"{block.model}/{block.class_name} duplicates {match.model}/{match.class_name} "
                        f"({variant.tag}); inherit it instead",
                        block.tier2_delta(match),
                    )
                )

    # R2: you inherit a block from a parent whose variant differs from yours, so you had to rewrite
    # `forward` -- while some other model already has your exact variant. The "wrong modular" case.
    if "R2" in rules:
        by_model_kind = defaultdict(list)
        for block in blocks:
            by_model_kind[(block.model, block.kind)].append(block)
        for (model, kind), model_blocks in sorted(by_model_kind.items()):
            if kind not in OVERRIDE_COST or OVERRIDE_COST[kind] < args.min_cost:
                continue
            if only and model not in only:
                continue
            model_parents = parents.get(model, ())
            if not model_parents:
                continue
            block = model_blocks[0]
            parent_variants = {p: {b.variant for b in by_model_kind.get((p, kind), ())} for p in model_parents}
            # Only interesting when the model inherits from parents that all disagree with it. A
            # parent with no block of this kind has nothing to disagree about, so needs a parent
            # that actually has one.
            if not any(parent_variants.values()):
                continue
            if any(block.variant in found for found in parent_variants.values()):
                continue
            better = [
                b.model
                for b in variants[block.tag].blocks
                if b.model != model
                and b.model not in ancestors(model)
                and model not in ancestors(b.model)
                and dates.get(b.model, "9999-99-99") < dates.get(model, "9999-99-99")
                and forwards_match(block, b)
            ]
            if not better:
                continue
            oldest = min(better, key=lambda m: (dates.get(m, "9999-99-99"), m))
            disagreeing = ", ".join(
                f"{p}={'/'.join(sorted(v)) or 'none'}" for p, v in sorted(parent_variants.items()) if v
            )
            findings.append(
                (
                    OVERRIDE_COST[kind],
                    "R2",
                    block,
                    f"{model}/{block.class_name} is {block.variant} but inherits {kind} from {disagreeing}; "
                    f"{oldest} already has this exact variant",
                    {},
                )
            )

    # R3: the block inherits its variant from a model younger than the one that introduced it, so
    # the lineage records copy order rather than history. Judged on the class's *declared* base --
    # using the model's parent set instead flagged classes already based on the canonical owner,
    # merely because some other parent of that model was younger.
    if "R3" in rules:
        all_overrides = list(modular_overrides())
        declared = {(o.child_model, o.child_class): o for o in all_overrides}
        # A base that supplies several classes is an established relationship, and sourcing one more
        # from it is descent, not drift: Gemma3 taking its MLP from Gemma2 is right even though llama
        # introduced `gated_mlp`. Only a base supplying exactly one class is a one-off reach into an
        # unrelated model -- the "Qwen VLM's text stack defined via GLM via Phi" problem.
        reach = modular_class_edges()
        for variant in variants.values():
            if variant.kind not in OVERRIDE_COST or OVERRIDE_COST[variant.kind] < args.min_cost:
                continue
            canonical = variant.canonical
            canonical_block = next((b for b in variant.blocks if b.model == canonical), None)
            if canonical_block is None:
                continue
            for block in variant.blocks:
                if only and block.model not in only:
                    continue
                if block.model == canonical or not forwards_match(block, canonical_block):
                    continue
                override = declared.get((block.model, block.class_name))
                if override is None or override.parent_model == canonical:
                    continue
                # Only interesting when the base predates nothing: the canonical owner is older, so
                # the same code could have come from further up the real lineage.
                # `<` not `<=`: a base that shares the canonical owner's date is still the wrong
                # parent, and same-day siblings are common (qwen3 and qwen3_moe both 2025-03-31, so
                # reaching into qwen3_moe for the attention qwen3 owns slipped through unreported).
                if dates.get(override.parent_model, "0000") < dates.get(canonical or "", "9999-99-99"):
                    continue
                if reach.get((block.model, override.parent_model), 0) > 1:
                    continue
                # Only worth reporting when the swap is free. Where the canonical owner's `__init__`
                # differs, the older parent is the *wrong* one: nine models take `gated_mlp` from
                # gemma (`bias=False`) rather than llama (`config.mlp_bias`), and switching to llama
                # would need an `__init__` override longer than the inheritance it replaced -- on a
                # config attribute those models do not even define.
                if tier2_mismatch(block, canonical_block):
                    continue
                findings.append(
                    (
                        OVERRIDE_COST[variant.kind] // 2,
                        "R3",
                        block,
                        f"{block.model}/{block.class_name} reaches into "
                        f"{override.parent_model}.{override.parent_class} "
                        f"({dates.get(override.parent_model, '?')}) for {variant.tag} and takes "
                        f"nothing else from it; {canonical} ({dates.get(canonical or '', '?')}) "
                        f"introduced that variant",
                        block.tier2_delta(canonical_block),
                    )
                )

    # Helper-level R1: the highest-count, lowest-risk duplication in the library.
    if "R1" in rules:
        by_name = defaultdict(lambda: defaultdict(list))
        for helper in helpers:
            by_name[helper.name][helper.variant].append(helper)
        for name, by_variant in by_name.items():
            for group in by_variant.values():
                if len({h.model for h in group}) < 2:
                    continue
                oldest = min((dates.get(h.model, "9999-99-99"), h.model) for h in group)[1]
                owners = {h.model for h in group}
                # The reuse is in place if *any* holder of this exact body is already an ancestor:
                # the converter inlined it from there. Testing only the canonical owner reported
                # every Llama descendant for `rotate_half`, whose oldest holder is gpt_neox.
                # A `# Copied from` marker does NOT count -- that is the legacy mechanism being
                # migrated away from, so a marker makes a finding easier to act on, not exempt.
                unrelated = [h for h in group if h.model != oldest and not (ancestors(h.model) & owners)]
                if only:
                    unrelated = [h for h in unrelated if h.model in only]
                if not unrelated:
                    continue
                # Cost is the helper's own length times the number of models re-declaring it, rather
                # than a flat guess: `rotate_half` is four lines, `eager_attention_forward` is thirty.
                body_loc = len(group[0].body.splitlines())
                findings.append(
                    (
                        body_loc * len(unrelated),
                        "R1",
                        None,
                        f"helper {name}() ({body_loc} LoC): {len(unrelated)} models redefine the body owned "
                        f"by {oldest} without inheriting it "
                        f"(e.g. {', '.join(sorted(h.model for h in unrelated)[:4])})",
                        {},
                    )
                )

    if args.fixable:
        findings = [f for f in findings if f[2] is None or generates_modeling(f[2].path)]
    findings.sort(key=lambda f: -f[0])
    shown = findings[: args.limit]
    for cost, rule, block, message, delta in shown:
        location = f"{block.path}:{block.lineno}" if block is not None else ""
        applicable = ""
        if block is not None:
            copied = copied_from_sources().get((block.model, block.class_name))
            if copied:
                # The legacy mechanism already names the source, so this is a known-safe conversion
                # to modular rather than a discovery -- the easiest kind of finding to act on.
                applicable += f"  [legacy `# Copied from {copied}`: convert to modular]"
            if not generates_modeling(block.path):
                applicable += "  [no modular yet: needs one authored, not a base swap]"
        print(f"[{rule}] ~{cost} LoC  {message}{applicable}")
        if delta:
            print(f"          init differs on: {', '.join(f'{k}: {a} vs {b}' for k, (a, b) in sorted(delta.items()))}")
        if location:
            print(f"          {location}")
    by_rule = defaultdict(int)
    for cost, rule, *_ in findings:
        by_rule[rule] += 1
    print(
        f"\n{len(findings)} findings ({', '.join(f'{r}={n}' for r, n in sorted(by_rule.items()))}); "
        f"showing {len(shown)}. Recoverable: ~{sum(f[0] for f in findings)} LoC."
    )
    return 1 if findings and args.strict else 0


# --------------------------------------------------------------------------------------------------
# fit-order
# --------------------------------------------------------------------------------------------------
def _score_ordering(order: tuple[str, ...], variants: list[str], weights: dict, costs: dict, kind: str) -> float:
    """
    Total override LoC the codebase would pay under this axis ordering.

    Each variant picks the existing variant it shares the longest prefix with -- the rule the trie
    and the wizard both use -- and pays for the axes on which they still differ. Ordering the
    expensive axes first makes agreement on them likelier, so less gets paid.
    """
    facets = {v: v.split("|") for v in variants}
    index = {axis: i for i, axis in enumerate(TIER1_AXES[kind])}
    positions = [index[axis] for axis in order]
    total = 0.0
    for mine in variants:
        best_prefix, best_cost = -1, 0.0
        my_facets = facets[mine]
        for theirs in variants:
            if theirs == mine:
                continue
            other = facets[theirs]
            prefix = 0
            for pos in positions:
                if my_facets[pos] != other[pos]:
                    break
                prefix += 1
            cost = sum(costs[(kind, a)] for a, i in index.items() if my_facets[i] != other[i])
            # Longest prefix wins; among equals prefer the cheaper remaining diff.
            if prefix > best_prefix or (prefix == best_prefix and cost < best_cost):
                best_prefix, best_cost = prefix, cost
        if best_prefix >= 0:
            total += best_cost * weights[mine]
    return total


def cmd_fit_order(args: argparse.Namespace) -> int:
    blocks, _ = scan_repo()
    costs, baseline = measure_axis_costs(blocks)

    print("== baseline: overrides whose variant MATCHES the class they inherit")
    for kind, locs in sorted(baseline.items(), key=lambda kv: -len(kv[1])):
        print(f"   {kind:10s} {len(locs):4d} overrides   median {statistics.median(locs):5.0f} LoC")
    print("   (this is the design's premise: same tier-1 variant means a trivial override)\n")

    variant_map = build_variants(blocks)
    for kind in ("attention", "moe", "mlp"):
        axes = TIER1_AXES[kind]
        if len(axes) < 2:
            continue
        kind_variants = [v.variant for v in variant_map.values() if v.kind == kind]
        weights = {v.variant: len(v.owners) for v in variant_map.values() if v.kind == kind}
        print(f"== {kind}: measured cost per axis (LoC of override when only this axis differs)")
        for axis in sorted(axes, key=lambda a: -costs[(kind, a)]):
            print(f"   {axis:14s} {costs[(kind, axis)]:6.0f}")

        orderings = list(itertools.permutations(axes))
        scored = sorted((_score_ordering(o, kind_variants, weights, costs, kind), o) for o in orderings)
        best_score, best = scored[0]
        current_score = _score_ordering(tuple(axes), kind_variants, weights, costs, kind)
        by_cost = tuple(sorted(axes, key=lambda a: -costs[(kind, a)]))
        by_cost_score = _score_ordering(by_cost, kind_variants, weights, costs, kind)
        print(f"   searched {len(orderings)} orderings")
        print(f"   current    {current_score:9.0f}   {' > '.join(axes)}")
        print(f"   by cost    {by_cost_score:9.0f}   {' > '.join(by_cost)}")
        print(f"   best       {best_score:9.0f}   {' > '.join(best)}")
        if best == by_cost:
            print("   -> the exhaustive optimum IS descending measured cost")
        else:
            print(f"   -> optimum differs from descending cost by {by_cost_score - best_score:.0f} LoC")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="build and print the variant catalog")
    p_compile.add_argument("--top", type=int, default=12, help="variants to print per block kind")
    p_compile.add_argument(
        "--markdown", nargs="?", const="docs/source/en/model_blocks.md", help="also write the catalog doc"
    )
    p_compile.set_defaults(func=cmd_compile)

    p_scan = sub.add_parser("scan", help="print the facets of one model or modeling file")
    p_scan.add_argument("target", help="a model name (llama) or a path to a modeling file")
    p_scan.set_defaults(func=cmd_scan)

    p_fit = sub.add_parser("fit-order", help="measure axis costs and search for the best axis order")
    p_fit.set_defaults(func=cmd_fit_order)

    p_lint = sub.add_parser("lint", help="report duplicated, wrongly-parented and anachronistic blocks")
    p_lint.add_argument("--rules", help="comma-separated subset of R1,R2,R3")
    p_lint.add_argument("--models", help="comma-separated models to restrict the report to")
    p_lint.add_argument("--limit", type=int, default=40, help="findings to print")
    p_lint.add_argument(
        "--min-cost", type=int, default=DEFAULT_MIN_COST, help="skip block kinds cheaper than this to override"
    )
    p_lint.add_argument("--strict", action="store_true", help="exit non-zero when there are findings")
    p_lint.add_argument(
        "--fixable",
        action="store_true",
        help="only findings on models whose modeling file the modular actually generates",
    )
    p_lint.set_defaults(func=cmd_lint)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
