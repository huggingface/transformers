# Copyright 2026 The HuggingFace Inc. team.
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
Keeps the `linguist-generated` block of `.gitattributes` in sync with the generated files (modular outputs and
model `__init__.py`), so GitHub collapses them in PR diffs.

Usage (from the root of the repo):

```bash
python utils/check_gitattributes.py                     # check (used in `make check-repo`)
python utils/check_gitattributes.py --fix_and_overwrite  # regenerate (used in `make fix-repo`)
```
"""

import argparse
from collections import defaultdict
from pathlib import Path


CHECKER_CONFIG = {
    "name": "gitattributes",
    "label": "Generated files in .gitattributes",
    "cache_globs": [".gitattributes", "src/transformers/models/**/*.py"],
    "check_args": [],
    "fix_args": ["--fix_and_overwrite"],
}

GITATTRIBUTES = Path(".gitattributes")
MODELS_DIR = Path("src/transformers/models")
GENERATED_HEADER = "This file was automatically generated from"
MARKER = "# Generated files, collapsed in GitHub PR diffs. Run `make fix-repo` to update this block."


def is_generated(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        return GENERATED_HEADER in f.read(1024)  # the header sits in the first lines


def generated_patterns() -> list[str]:
    # One glob for the `__init__.py`, then one per file kind (`modeling_*.py`, `configuration_*.py`, ...) in each model
    # folder, used only when every file of that kind is generated, so hand-written sources like `modular_*.py` never match.
    patterns = [f"{MODELS_DIR.as_posix()}/**/__init__.py linguist-generated=true"]
    for folder in sorted({path.parent for path in MODELS_DIR.rglob("*.py")}):
        kinds = defaultdict(list)
        for path in sorted(folder.glob("*.py")):
            if path.name != "__init__.py":
                kinds[path.name.split("_")[0]].append(path)
        for kind, files in kinds.items():
            generated = [path for path in files if is_generated(path)]
            if len(generated) == len(files):
                patterns.append(f"{folder.as_posix()}/{kind}_*.py linguist-generated=true")
            else:
                patterns += [f"{path.as_posix()} linguist-generated=true" for path in generated]
        init = folder / "__init__.py"
        if init.exists() and not is_generated(init):
            patterns.append(f"{init.as_posix()} linguist-generated=false")
    return patterns


def expected_content(current: str) -> str:
    manual = current.split(MARKER)[0].rstrip("\n")
    block = "\n".join(generated_patterns())
    return f"{manual}\n\n{MARKER}\n{block}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    current = GITATTRIBUTES.read_text(encoding="utf-8")
    expected = expected_content(current)
    if current != expected:
        if not args.fix_and_overwrite:
            raise ValueError("`.gitattributes` is out of sync with the generated files, run `make fix-repo`.")
        GITATTRIBUTES.write_text(expected, encoding="utf-8")
