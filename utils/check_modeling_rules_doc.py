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
Keep `## Rules reference` section ofdocs/source/en/modeling_rules.m in sync
with utils/mlinter/rules.toml.

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
import os
import sys


CHECKER_CONFIG = {
    "name": "modeling_rules_doc",
    "label": "Modeling rules documentation",
    "file_globs": ["utils/mlinter/rules.toml", "docs/source/en/modeling_rules.md"],
    "check_args": [],
    "fix_args": ["--fix_and_overwrite"],
}

ROOT = os.path.dirname(os.path.dirname(__file__))
DOC_PATH = os.path.join(ROOT, "docs", "source", "en", "modeling_rules.md")

BEGIN_MARKER = "<!-- BEGIN RULES REFERENCE -->"
END_MARKER = "<!-- END RULES REFERENCE -->"


sys.path.insert(0, ROOT)
from utils.mlinter.mlinter import TRF_RULE_SPECS, format_rule_details  # noqa: E402


def generate_rules_reference() -> str:
    sections = []
    for rule_id in sorted(TRF_RULE_SPECS):
        sections.append(format_rule_details(rule_id))
    return "\n\n".join(sections) + "\n"


def check_modeling_rules_doc(overwrite: bool = False):
    with open(DOC_PATH, encoding="utf-8") as f:
        content = f.read()

    begin_idx = content.find(BEGIN_MARKER)
    end_idx = content.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1:
        raise ValueError(
            f"Could not find {BEGIN_MARKER} and {END_MARKER} markers in {DOC_PATH}. "
            "These markers delimit the auto-generated rules reference section."
        )

    after_begin = begin_idx + len(BEGIN_MARKER)
    expected = "\n\n" + generate_rules_reference() + "\n"
    current = content[after_begin:end_idx]

    if current == expected:
        return

    if overwrite:
        new_content = content[:after_begin] + expected + content[end_idx:]
        with open(DOC_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated rules reference in {DOC_PATH}")
    else:
        raise ValueError(
            "The rules reference section in docs/source/en/modeling_rules.md is out of sync "
            "with utils/mlinter/rules.toml. Run `make fix-repo` to regenerate it."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_modeling_rules_doc(args.fix_and_overwrite)
