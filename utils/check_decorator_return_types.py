# Copyright 2026 The HuggingFace Team.
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
"""AST-based checks for decorators that modify return types.

This script ensures that functions decorated with `can_return_tuple` or
`check_model_inputs`:

1. Have an explicit, non-`None` return annotation.
2. Are not annotated with a union that already includes `tuple`.

The intention is that the decorators themselves are responsible for
adding the `tuple` part of the return type, so the underlying function
should be annotated with just the base return type.

Usage (from the root of the repo):

```bash
python utils/check_decorator_return_types.py
```
"""

from __future__ import annotations

import argparse
import ast
import os
from collections.abc import Iterable
from dataclasses import dataclass


PATH_TO_TRANSFORMERS = "src/transformers"


TARGET_DECORATORS = {"can_return_tuple", "check_model_inputs"}


@dataclass
class Violation:
    file_path: str
    line: int
    function_name: str
    decorator_name: str
    message: str

    def format(self) -> str:
        return (
            f"{self.file_path}:{self.line}: function '{self.function_name}' "
            f"decorated with '@{self.decorator_name}' {self.message}"
        )


def _iter_python_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _decorator_name(node: ast.expr) -> str | None:
    """Return the simple name of a decorator, if it matches a target.

    Handles forms like:
    - @can_return_tuple
    - @utils.can_return_tuple
    - @can_return_tuple(...)
    - @utils.check_model_inputs(...)
    """

    target = node
    if isinstance(target, ast.Call):
        target = target.func

    if isinstance(target, ast.Name):
        name = target.id
    elif isinstance(target, ast.Attribute):
        name = target.attr
    else:
        return None

    if name in TARGET_DECORATORS:
        return name
    return None


def _is_none_annotation(returns: ast.expr | None) -> bool:
    if returns is None:
        return True

    # -> None
    if isinstance(returns, ast.Constant) and returns.value is None:
        return True

    # -> None (as a name)
    if isinstance(returns, ast.Name) and returns.id == "None":
        return True

    return False


def _is_tuple_type(node: ast.AST) -> bool:
    """Return True if the node represents a tuple type.

    We conservatively treat the following as tuple types:
    - `tuple`
    - `tuple[...]`
    - `Tuple[...]` (from typing)
    """

    if isinstance(node, ast.Name) and node.id in {"tuple", "Tuple"}:
        return True

    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Name) and value.id in {"tuple", "Tuple"}:
            return True

    return False


def _iter_union_members(node: ast.AST) -> Iterable[ast.AST]:
    """Yield flattened members of a PEP 604-style union (X | Y | Z).

    For non-union nodes, yields the node itself once.
    """

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        yield from _iter_union_members(node.left)
        yield from _iter_union_members(node.right)
    else:
        yield node


def _has_tuple_in_union(returns: ast.expr) -> bool:
    members = list(_iter_union_members(returns))
    if len(members) <= 1:
        # Not a union
        return False

    return any(_is_tuple_type(member) for member in members)


def _is_delegating_to_super(func_node: ast.AST) -> bool:
    """Return True if the function body starts with a super(...) delegation.

    We ignore functions whose first non-docstring statement is either:
    - `return super(...` (possibly via an attribute like `super().foo(...)`), or
    - `super(...` as a bare expression.
    """

    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False

    body = getattr(func_node, "body", [])
    if not body:
        return False

    # Skip an initial docstring expression if present.
    first_stmt_idx = 0
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        first_stmt_idx = 1

    if first_stmt_idx >= len(body):
        return False

    first_stmt = body[first_stmt_idx]
    if isinstance(first_stmt, ast.Return):
        target = first_stmt.value
    elif isinstance(first_stmt, ast.Expr):
        target = first_stmt.value
    else:
        return False

    if target is None:
        return False

    # Look for a super(...) call anywhere in the expression tree.
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super":
            return True

    return False


def _collect_decorated_functions(tree: ast.AST, file_path: str) -> list[tuple[ast.AST, str]]:
    """Return (function_node, decorator_name) pairs for targeted decorators."""

    functions: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.decorator_list:
            continue
        for deco in node.decorator_list:
            name = _decorator_name(deco)
            if name is not None:
                functions.append((node, name))
                break
    return functions


def _compute_line_offsets(source: str) -> list[int]:
    """Return starting offset in the full string for each line (0-based)."""

    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _make_union_without_tuple(returns: ast.expr) -> str | None:
    """Build a new union annotation string without any tuple-type members.

    Returns the new annotation expression as a string, or None if it cannot
    be constructed (e.g. all members were tuple types).
    """

    members = [m for m in _iter_union_members(returns) if not _is_tuple_type(m)]
    if not members:
        return None

    # We rely on Python's built-in unparser (3.9+).
    pieces = [ast.unparse(m) for m in members]
    return " | ".join(pieces)


def check_decorator_return_types(overwrite: bool = False):
    all_violations: list[Violation] = []
    unfixable_violations: list[Violation] = []

    for file_path in _iter_python_files(PATH_TO_TRANSFORMERS):
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=file_path, type_comments=True)
        except SyntaxError as e:
            print(f"Skipping {file_path} due to SyntaxError: {e}")
            continue

        functions = _collect_decorated_functions(tree, file_path)
        if not functions:
            continue

        fixes: list[tuple[int, int, str]] = []  # (start, end, new_text)

        for func_node, decorator_name in functions:
            # Ignore trivial delegations like `return super(...` or `super(...`.
            if _is_delegating_to_super(func_node):
                continue

            returns = func_node.returns

            # 1. Must have a non-None return annotation.
            if _is_none_annotation(returns):
                v = Violation(
                    file_path=file_path,
                    line=func_node.lineno,
                    function_name=func_node.name,
                    decorator_name=decorator_name,
                    message="must have a non-None return annotation",
                )
                all_violations.append(v)
                unfixable_violations.append(v)
                continue

            # Nothing else to do without an annotation.
            if returns is None:
                continue

            # 2. Annotation must not already be a union including `tuple`.
            if _has_tuple_in_union(returns):
                v = Violation(
                    file_path=file_path,
                    line=func_node.lineno,
                    function_name=func_node.name,
                    decorator_name=decorator_name,
                    message="must not be annotated with a union that includes 'tuple'",
                )
                all_violations.append(v)

                if not overwrite:
                    continue

                new_annotation = _make_union_without_tuple(returns)
                if new_annotation is None:
                    unfixable_violations.append(v)
                    continue

                # Use precise offsets to replace just the annotation.
                if not hasattr(returns, "lineno") or not hasattr(returns, "end_lineno"):
                    unfixable_violations.append(v)
                    continue

                line_offsets = _compute_line_offsets(source)
                try:
                    start = line_offsets[returns.lineno - 1] + returns.col_offset
                    end = line_offsets[returns.end_lineno - 1] + returns.end_col_offset
                except IndexError:
                    unfixable_violations.append(v)
                    continue

                fixes.append((start, end, new_annotation))

        if overwrite and fixes:
            # Apply fixes from the end of the file backwards so offsets stay valid.
            fixes.sort(key=lambda x: x[0], reverse=True)
            new_source = source
            for start, end, text in fixes:
                new_source = new_source[:start] + text + new_source[end:]

            if new_source != source:
                print(f"Updating return annotations in {file_path} to drop 'tuple' from unions.")
                with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new_source)

    if all_violations and not overwrite:
        header = "Found decorator return-type violations:\n\n"
        body = "\n".join(v.format() for v in all_violations)
        footer = "\n\nRun this script with --fix_and_overwrite to auto-fix some violations."
        raise ValueError(header + body + footer)

    if overwrite and unfixable_violations:
        header = "Found decorator return-type violations that could not be auto-fixed:\n\n"
        body = "\n".join(v.format() for v in unfixable_violations)
        footer = "\n\nPlease fix these annotations manually."
        raise ValueError(header + body + footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_decorator_return_types(args.fix_and_overwrite)
