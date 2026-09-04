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
Repo-consistency check for the block-variant catalog.

Two jobs:

1. Keep `docs/source/en/model_blocks.md` in sync with the code. This part is fixable and is what
   `--fix_and_overwrite` regenerates.
2. Report duplicated / wrongly-parented / anachronistic blocks for the models touched by the
   current diff. This part is **advisory**: the backlog on `main` is ~1000 findings, so it prints
   and exits 0 unless `--strict` is passed.

    python utils/check_model_blocks.py
    python utils/check_model_blocks.py --fix_and_overwrite
    python utils/check_model_blocks.py --strict          # fail on findings in changed models
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from blocks_cli import _render_markdown, cmd_lint  # noqa: E402
from blocks_export import export_all  # noqa: E402
from blocks_facets import build_date_data, build_variants, scan_repo  # noqa: E402
from blocks_screen import MARKER  # noqa: E402


CHECKER_CONFIG = {
    "name": "model_blocks",
    "label": "Model block variants",
    # The catalog is derived from modeling sources, model configs and the model cards that carry
    # each model's contribution date.
    "cache_globs": [
        "src/transformers/models/**/modeling_*.py",
        "src/transformers/models/**/configuration_*.py",
        "src/transformers/models/**/modular_*.py",
        "docs/source/en/model_doc/**/*.md",
        "utils/model_blocks.json",
    ],
    "check_args": [],
    "fix_args": ["--fix_and_overwrite"],
}

CATALOG_PATH = Path("docs/source/en/model_blocks.md")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _catalog_text() -> str:
    """Render the catalog markdown without writing it."""
    blocks, helpers = scan_repo()
    helper_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for helper in helpers:
        helper_groups[helper.name][helper.variant].add(helper.model)
    return _render_markdown(build_variants(blocks), helper_groups, build_date_data())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix_and_overwrite", action="store_true", help="regenerate the catalog doc")
    parser.add_argument("--strict", action="store_true", help="fail when changed models have findings")
    parser.add_argument("--all", action="store_true", help="lint every model, not just the diff")
    args = parser.parse_args()

    # The manifest is generated and committed, so it needs the same treatment as the catalog.
    stale = export_all(check_only=not args.fix_and_overwrite)
    if args.fix_and_overwrite:
        print("wrote utils/model_blocks.json" if stale else "utils/model_blocks.json already current")
    elif stale:
        print(
            "utils/model_blocks.json is out of date. Run `python utils/check_model_blocks.py --fix_and_overwrite`.",
            file=sys.stderr,
        )
        return 1

    # A modular file must never carry a `# Copied from` marker: modular *replaces* that mechanism,
    # expressing the relationship as an import plus a base class. A marker inside a modular is a
    # claim nothing checks, and the draft generator seeded ten of them before this check existed.
    offenders = []
    for path in sorted((REPO_ROOT / "src" / "transformers" / "models").glob("*/modular_*.py")):
        hits = [i + 1 for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()) if MARKER.search(line)]
        if hits:
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{','.join(map(str, hits[:4]))}")
    if offenders:
        print(
            "modular files must not contain `# Copied from` markers; use an import and a base class:\n  "
            + "\n  ".join(offenders),
            file=sys.stderr,
        )
        return 1

    expected = _catalog_text()
    if args.fix_and_overwrite:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(expected, encoding="utf-8")
        print(f"wrote {CATALOG_PATH}")
    else:
        current = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else ""
        if current != expected:
            print(
                f"{CATALOG_PATH} is out of date. Run `python utils/check_model_blocks.py --fix_and_overwrite`.",
                file=sys.stderr,
            )
            return 1

    models = None
    if not args.all:
        from check_modular_conversion import get_models_in_diff

        changed = get_models_in_diff()
        if not changed:
            print("no models in the diff; skipping the block-reuse report")
            return 0
        models = ",".join(sorted(changed))
        print(f"block-reuse report for models in the diff: {models}\n")

    lint_args = argparse.Namespace(
        rules=None,
        models=models,
        limit=40,
        min_cost=10,
        fixable=False,
        strict=args.strict and not args.fix_and_overwrite,
    )
    return cmd_lint(lint_args)


if __name__ == "__main__":
    raise SystemExit(main())
