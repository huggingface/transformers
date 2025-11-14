#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Utility that ensures `_init_weights(self, module)` implementations do not use `.data`.

Direct `.data` access breaks the lazy-initialization safeguards handled by `HFParameter`, so the library forbids it.
"""

import ast
import sys
from pathlib import Path


MODELING_ROOT = Path("src/transformers/models")
MODELING_PATTERNS = ("modeling_*.py", "modular_*.py")


def iter_modeling_files():
    for pattern in MODELING_PATTERNS:
        yield from MODELING_ROOT.rglob(pattern)


def function_has_forbidden_data_usage(fn: ast.FunctionDef) -> int | None:
    """
    Returns the first offending line number if `.data` is used, otherwise `None`.
    """

    args = fn.args.args
    if len(args) < 2 or getattr(args[0], "arg", None) != "self" or getattr(args[1], "arg", None) != "module":
        return None

    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "data":
            return node.lineno

    return None


def main() -> int:
    violations: list[str] = []

    for file_path in iter_modeling_files():
        try:
            text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(file_path))
        except Exception as exc:
            violations.append(f"{file_path}: failed to parse ({exc}).")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_init_weights":
                offending_line = function_has_forbidden_data_usage(node)
                if offending_line is not None:
                    violations.append(
                        f"{file_path}:{offending_line}: `_init_weights(self, module)` uses `.data`. "
                        "Use tensor ops directly to remain compatible with HFParameter."
                    )
                    break

    if violations:
        print("Found forbidden `.data` usage inside `_init_weights(self, module)`:\n", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
