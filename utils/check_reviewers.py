# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Check that every model — and every rule in the reviewer file — can still reach a reviewer.

The `Assign PR Reviewers` workflow (`.github/workflows/assign-reviewers.yml`) resolves reviewers from
`.github/scripts/codeowners_for_review_action`, using the shared resolver in
`transformersci.reviewers` (huggingface/transformers-ci). Both failure modes it has had are silent:
a rule whose pattern matches nothing looks fine forever, and a model nobody claims quietly falls to
the `*` catch-all. This check makes both loud.

The resolver is not a dependency of this package. `utils/checkers.py` installs it from
`utils/checkers-requirements.txt` on the first run in an environment, so normally it is simply there. If that
did not happen — offline, or the install opted out — the check reports what to install and passes,
so a contributor is never blocked on a package unrelated to their change. In CI a missing resolver
is an error instead: a check that quietly passes because its own dependency vanished is the
failure this file exists to prevent.

It reports:
  - a model directory that only the `*` catch-all claims;
  - a rule whose pattern matches no tracked file (a renamed or removed path);
  - a `@@modality/<slug>` rule naming a modality that does not exist;
  - a modality used by the doc toctree with no owner;
  - one owner spelled two ways (`@ArthurZucker` and `@arthurzucker` tally separately);
  - a malformed `# Reviewers:` tag.

Collaborator status is not checked here — it needs the API, and the workflow already warns and skips
at request time.

Usage:

```bash
python utils/check_reviewers.py
python utils/check_reviewers.py --strict   # also fail on non-model paths with no owner
```
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


CHECKER_CONFIG = {
    "name": "reviewers",
    "label": "Reviewer assignment",
    # Also reads docs/source/en/_toctree.yml (modalities) and model file headers (`# Reviewers:`).
    "cache_globs": [
        ".github/scripts/codeowners_for_review_action",
        "docs/source/en/_toctree.yml",
        "src/transformers/models/*/mod*.py",
    ],
    "check_args": [],
    "fix_args": None,
    # Pulls the shared resolver from `utils/checkers-requirements.txt`; see `load_resolver`.
    "needs_requirements": True,
}

REPO_ROOT = Path(__file__).parent.parent

# Paths that carry no code to review.
IGNORED_NAMES = {"__pycache__", ".DS_Store", "py.typed"}

HOW_TO_ADD = """
A path gets a reviewer in one of three places, most specific first:

  1. `# Reviewers: @login` in the leading comment block of
     `src/transformers/models/<model>/modular_<model>.py` (or `modeling_<model>.py`). Use this when
     one person owns a single model; the modular converter copies the header into the generated file,
     so the tag survives regeneration.

  2. A path rule in `.github/scripts/codeowners_for_review_action`, e.g.
     `/src/transformers/models/<model>/mod*_<model>* @login`, or a directory such as
     `/src/transformers/<area>/ @login`. Use this for anything that is not a model, and for a model
     whose owner is not the owner of its modality.

  3. The modality table in the same file (`@@modality/<slug> @login`). It covers every model whose
     doc page sits in that section of `docs/source/en/_toctree.yml`, so a new model normally needs
     nothing here — if a model is missed, it is usually missing from the toctree instead
     (see `utils/check_doc_toc.py`).

A rule with a pattern and no owner marks a path as deliberately unowned, e.g. `utils/dummy*`.
"""


INSTALL_RESOLVER = "pip install -r utils/checkers-requirements.txt"


def load_resolver():
    """The shared resolver, or `None` if it is not installed.

    It lives in huggingface/transformers-ci, alongside the workflow that resolves reviewers for
    real, so the two cannot disagree. It is deliberately not in `setup.py`: that package has no
    release on PyPI, and a `git+` URL in the metadata is a direct reference, which PyPI refuses to
    accept when this one is uploaded. It is pinned in `utils/checkers-requirements.txt` instead, which `utils/checkers.py` installs for you.
    """
    try:
        from transformersci.reviewers import resolver
    except ImportError:
        return None
    return resolver


def tracked_files():
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return output.splitlines()


def model_directories(files, models_dir):
    prefix = f"{models_dir}/"
    return sorted({f[len(prefix) :].split("/")[0] for f in files if f.startswith(prefix) and "/" in f[len(prefix) :]})


def check_models_have_owners(resolver, codeowners_lines, files):
    """Model directories that only the `*` catch-all claims, grouped by why."""
    unplaced = []
    by_modality = {}
    for model in model_directories(files, resolver.MODELS_DIR):
        if model in IGNORED_NAMES:
            continue
        probe = f"{resolver.MODELS_DIR}/{model}/modeling_{model}.py"
        if resolver.resolution_source(probe, codeowners_lines) != "catch-all":
            continue
        modality = resolver.modality_of_model(model)
        if modality is None:
            unplaced.append(model)
        else:
            by_modality.setdefault(modality, []).append(model)

    errors = [
        f"model `{model}` has no reviewer: the doc toctree does not place it in a modality" for model in unplaced
    ]
    # One line per modality rather than one per model: a missing owner takes down its whole section.
    errors += [
        f"modality `{modality}` has no owner, leaving {len(models)} model(s) with no reviewer "
        f"(e.g. {', '.join(models[:3])})"
        for modality, models in sorted(by_modality.items())
    ]
    return errors


def check_rules_match_something(resolver, codeowners_lines, files):
    """Rules whose pattern matches no tracked file, i.e. a renamed or removed path."""
    errors = []
    for pattern, _ in resolver.iter_rules(codeowners_lines):
        if pattern.startswith(resolver.MODALITY_PREFIX):
            continue
        regex = re.compile(resolver.pattern_to_regex(pattern))
        if not any(regex.search(f) for f in files):
            errors.append(f"rule `{pattern}` matches no tracked file (renamed or removed?)")
    return errors


def check_modalities(resolver, codeowners_lines):
    """Modality rules naming an unknown modality, and modalities in the toctree with no owner."""
    errors = [
        f"rule `{resolver.MODALITY_PREFIX}{slug}` names an unknown modality; expected one of "
        f"{sorted(set(resolver.MODALITY_SECTIONS.values()))}"
        for slug in resolver.unknown_modality_slugs(codeowners_lines)
    ]
    owners = resolver.modality_owners_table(codeowners_lines)
    for modality in sorted(set(resolver.toctree_modalities().values())):
        if not owners.get(modality):
            errors.append(
                f"modality `{modality}` is used by the doc toctree but has no owner; add "
                f"`{resolver.MODALITY_PREFIX}{modality} @login`"
            )
    return errors


def check_owner_spellings(resolver, codeowners_lines):
    """One owner spelled two ways: the workflow tallies per name, so the two would compete."""
    spellings = {}
    for _, owners in resolver.iter_rules(codeowners_lines):
        for owner in owners:
            spellings.setdefault(owner.casefold(), set()).add(owner)
    return [
        f"owner {sorted(variants)} is spelled {len(variants)} ways; use one spelling everywhere"
        for variants in spellings.values()
        if len(variants) > 1
    ]


def check_reviewer_tags(resolver, files):
    """`# Reviewers:` tags whose value is not a list of @logins."""
    errors = []
    login = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9-]*$")
    for path in files:
        name = Path(path).name
        if not name.startswith(("modeling_", "modular_")):
            continue
        for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                break  # the header ends at the first statement
            tag = resolver.REVIEWERS_TAG_RE.match(line)
            if tag is None:
                continue
            bad = [token for token in tag.group(1).split() if not login.match(token)]
            if bad:
                errors.append(f"{path}: `# Reviewers:` tag has invalid entries {bad}; expected `@login`")
            break
    return errors


def uncovered_paths(resolver, codeowners_lines):
    """Non-model paths under `src/transformers` that only the `*` catch-all claims."""
    uncovered = []
    for entry in sorted((REPO_ROOT / "src" / "transformers").iterdir()):
        if entry.name in IGNORED_NAMES or entry.name == "models" or entry.suffix not in {".py", ""}:
            continue
        relative = entry.relative_to(REPO_ROOT).as_posix()
        probe = f"{relative}/__init__.py" if entry.is_dir() else relative
        if resolver.resolution_source(probe, codeowners_lines) == "catch-all":
            uncovered.append(relative)
    return uncovered


def main(strict=False):
    resolver = load_resolver()
    if resolver is None:
        message = f"the shared resolver is not installed. Install it with:\n  {INSTALL_RESOLVER}"
        if os.environ.get("CI"):
            # CI installs it before running this, so missing here means that step broke -- and a
            # check that quietly passes because its own dependency vanished is the exact failure
            # mode this file exists to prevent.
            raise ValueError(f"Cannot check reviewer assignment: {message}")
        print(f"Skipping the reviewer check: {message}")
        return
    codeowners_lines = (REPO_ROOT / resolver.CODEOWNERS_PATH).read_text(encoding="utf-8").splitlines(keepends=True)
    files = tracked_files()

    errors = (
        check_models_have_owners(resolver, codeowners_lines, files)
        + check_rules_match_something(resolver, codeowners_lines, files)
        + check_modalities(resolver, codeowners_lines)
        + check_owner_spellings(resolver, codeowners_lines)
        + check_reviewer_tags(resolver, files)
    )

    uncovered = uncovered_paths(resolver, codeowners_lines)
    if uncovered and strict:
        report = "\n".join(f"  - {path}" for path in uncovered)
        errors.append(
            f"{len(uncovered)} path(s) under src/transformers have no owner and fall to the `*` catch-all:\n{report}"
        )
    elif uncovered:
        # A one-liner in the passing case: this is a standing gap, not something a PR introduced.
        print(
            f"Note: {len(uncovered)} path(s) under src/transformers have no owner and fall to the `*` "
            "catch-all. List them with `python utils/check_reviewers.py --strict`."
        )

    if errors:
        listed = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Reviewer assignment is incomplete:\n{listed}\n{HOW_TO_ADD}")
    print(f"Reviewer assignment OK: {len(model_directories(files, resolver.MODELS_DIR))} models covered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="also fail on non-model paths with no owner")
    args = parser.parse_args()
    try:
        main(strict=args.strict)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
