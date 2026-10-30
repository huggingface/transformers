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
MARKER = "# Generated files, collapsed in GitHub PR diffs. Run `make fix-repo` to update this block."


def is_generated(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        return GENERATED_HEADER in f.read(1024)  # the header sits in the first lines


def generated_patterns() -> list[str]:
    # `__init__.py` are all generated, and a model folder whose other files (modular/convert sources aside) are all
    # generated gets a single glob; the negations at the end take back what those globs over-match.
    patterns = [f"{MODELS_DIR.as_posix()}/**/__init__.py"]
    negations = [f"{MODELS_DIR.as_posix()}/**/{glob}" for glob in ("modular_*.py", "convert_*.py")]
    for folder in sorted({path.parent for path in MODELS_DIR.rglob("*.py")}):
        files = [path for path in sorted(folder.glob("*.py")) if path.name != "__init__.py"]
        sources = [path for path in files if path.name.startswith(("modular_", "convert_"))]
        generated = [path for path in files if is_generated(path)]
        if generated and len(generated) + len(sources) == len(files):
            patterns.append(f"{folder.as_posix()}/*.py")
        else:
            patterns += [path.as_posix() for path in generated]
        init = folder / "__init__.py"
        if init.exists() and not is_generated(init):
            negations.append(init.as_posix())
    return [f"{pattern} linguist-generated=true" for pattern in patterns] + [
        f"{pattern} linguist-generated=false" for pattern in negations
    ]


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
