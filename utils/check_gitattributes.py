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
GENERATED_KINDS = ("configuration", "feature", "image", "modeling", "processing", "tokenization", "video")
MARKER = "# Generated files, collapsed in GitHub PR diffs. Run `make fix-repo` to update this block."


def is_generated(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        return GENERATED_HEADER in f.read(1024)  # the header sits in the first lines


def generated_patterns() -> list[str]:
    # Repo-wide globs for the file kinds modular generates, so a new modular model is covered without editing this
    # file (the CI security gate blocks PRs touching `.gitattributes`). Hand-written files get a `false` override (the
    # last matching line wins), other generated files a per-file `true`.
    patterns = [f"{MODELS_DIR.as_posix()}/**/{kind}_*.py linguist-generated=true" for kind in GENERATED_KINDS]
    patterns.append(f"{MODELS_DIR.as_posix()}/**/__init__.py linguist-generated=true")
    for path in sorted(MODELS_DIR.rglob("*.py")):
        covered = path.name == "__init__.py" or path.name.split("_")[0] in GENERATED_KINDS
        if is_generated(path) != covered:
            patterns.append(f"{path.as_posix()} linguist-generated={str(not covered).lower()}")
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
