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
Keep `## Rules reference` section of docs/source/en/modeling_rules.md in sync
with the rules defined in utils/rules.toml via the installed mlinter package.

Usage (from the root of the repo):

Check everything is up to date (used in ``make check-repo``):

```bash
python utils/check_modeling_rules_doc.py
```

Auto-regenerate if out of date (used in ``make fix-repo``):

```bash
python utils/check_modeling_rules_doc.py --fix_and_overwrite
```
"""

import argparse
from pathlib import Path


CHECKER_CONFIG = {
    "name": "modeling_rules_doc",
    "label": "Modeling rules documentation",
    # Depends on utils/rules.toml plus the installed `mlinter` package output,
    # which cannot be fully expressed as repo cache globs for the checker cache.
    "cache_globs": None,
    "check_args": ["--rules-toml", "utils/rules.toml"],
    "fix_args": ["--rules-toml", "utils/rules.toml", "--fix_and_overwrite"],
}

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "source" / "en" / "modeling_rules.md"
RULES_TOML_PATH = ROOT / "utils" / "rules.toml"

BEGIN_MARKER = "<!-- BEGIN RULES REFERENCE -->"
END_MARKER = "<!-- END RULES REFERENCE -->"


def _require_mlinter():
    try:
        import mlinter
        from mlinter import mlinter as mlinter_impl
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "This script requires the standalone `transformers-mlinter` package. "
            'Install the repo quality dependencies with `pip install -e ".[quality]"` and retry.'
        ) from error

    return mlinter, mlinter_impl


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def generate_rules_reference(rule_specs_path: Path = RULES_TOML_PATH) -> str:
    mlinter, mlinter_impl = _require_mlinter()
    # Reuse mlinter's registry-switching helper so docs rendering reflects the repo-local rule file.
    with mlinter_impl._using_rule_specs(_resolve_path(rule_specs_path)):
        return mlinter.render_rules_reference()


def check_modeling_rules_doc(overwrite: bool = False, rule_specs_path: Path = RULES_TOML_PATH):
    with DOC_PATH.open(encoding="utf-8") as f:
        content = f.read()

    begin_idx = content.find(BEGIN_MARKER)
    end_idx = content.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Could not find {BEGIN_MARKER} and {END_MARKER} markers in {DOC_PATH}. "
            "These markers delimit the auto-generated rules reference section."
        )

    after_begin = begin_idx + len(BEGIN_MARKER)
    expected = "\n\n" + generate_rules_reference(rule_specs_path) + "\n"
    current = content[after_begin:end_idx]

    if current == expected:
        return

    if overwrite:
        new_content = content[:after_begin] + expected + content[end_idx:]
        with DOC_PATH.open("w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated rules reference in {DOC_PATH}")
    else:
        raise ValueError(
            "The rules reference section in docs/source/en/modeling_rules.md is out of sync "
            "with utils/rules.toml. Run `make fix-repo` to regenerate it."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rules-toml",
        type=Path,
        default=RULES_TOML_PATH,
        help="Path to a rules TOML file. Defaults to utils/rules.toml.",
    )
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    try:
        check_modeling_rules_doc(args.fix_and_overwrite, args.rules_toml)
    except ModuleNotFoundError as error:
        raise SystemExit(str(error)) from error
