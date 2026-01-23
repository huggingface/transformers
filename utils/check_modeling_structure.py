#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team.
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
Utility that ensures that modeling (and modular) files respect some important conventions we have in Transformers.
"""

import ast
import sys
from pathlib import Path

from rich import print


MODELS_ROOT = Path("src/transformers/models")
MODELING_PATTERNS = ("modeling_*.py", "modular_*.py")


def iter_modeling_files():
    for pattern in MODELING_PATTERNS:
        yield from MODELS_ROOT.rglob(pattern)


def colored_error_message(file_path: str, line_number: int, message: str) -> str:
    return f"[bold red]{file_path}[/bold red]:[bold yellow]L{line_number}[/bold yellow]: {message}"


def full_name(node: ast.AST):
    """
    Return full dotted name from an Attribute or Name node.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return full_name(node.value) + "." + node.attr
    else:
        raise ValueError("Not a Name or Attribute node")


def check_init_weights(node: ast.AST, violations: list[str], file_path: str) -> list[str]:
    """
    Check that `_init_weights` correctly use `init.(...)` patterns to init the weights in-place. This is very important,
    as we rely on the internal flag set on the parameters themselves to check if they need to be re-init or not.
    """
    if isinstance(node, ast.FunctionDef) and node.name == "_init_weights":
        args = node.args.args
        if len(args) < 2 or getattr(args[0], "arg", None) != "self" or getattr(args[1], "arg", None) != "module":
            return violations

        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Attribute):
                is_inplace_ops = sub_node.func.attr.endswith("_")
                # We allow in-place ops on tensors that are not part of the module itself (see e.g. modeling_qwen3_next.py L997)
                is_on_module_weight = isinstance(
                    sub_node.func.value, (ast.Name, ast.Attribute)
                ) and "module." in full_name(sub_node.func.value)
                if is_inplace_ops and is_on_module_weight:
                    error_msg = (
                        "`_init_weights(self, module)` uses an in-place operation on a module's weight. Please use the "
                        "`init` functions primitives instead, usually imported as `from ... import initialization as init`"
                    )
                    violations.append(colored_error_message(file_path, sub_node.lineno, error_msg))

    return violations


def is_self_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a method call on `self`, such as `self.method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method
    )


def is_super_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a call to `super().method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and node.func.attr == method
    )


def check_post_init(node: ast.AST, violations: list[str], file_path: str) -> list[str]:
    """
    Check that `self.post_init()` is correctly called at the end of `__init__` for all `PreTrainedModel`s. This is
    very important as we need to do some processing there.
    """
    # Check if it's a PreTrainedModel class definition
    if isinstance(node, ast.ClassDef) and any(full_name(parent).endswith("PreTrainedModel") for parent in node.bases):
        for sub_node in node.body:
            # Check that we are in __init__
            if isinstance(sub_node, ast.FunctionDef) and sub_node.name == "__init__":
                for statement in ast.walk(sub_node):
                    # This means it's correctly called verbatim
                    if is_self_method_call(statement, method="post_init"):
                        break
                    # This means `super().__init__` is called in a modular, so it is already called in the parent
                    elif "modular_" in str(file_path) and is_super_method_call(statement, method="__init__"):
                        break
                # If we did not break, `post_init` was never called
                else:
                    error_msg = f"`__init__` of {node.name} does not call `self.post_init`"
                    violations.append(colored_error_message(file_path, sub_node.lineno, error_msg))
                break

    return violations


def main():
    violations: list[str] = []

    for file_path in iter_modeling_files():
        try:
            text = file_path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(file_path))
        except Exception as exc:
            violations.append(f"{file_path}: failed to parse ({exc}).")
            continue

        for node in ast.walk(tree):
            violations = check_init_weights(node, violations, file_path)
            violations = check_post_init(node, violations, file_path)

    if len(violations) > 0:
        violations = sorted(violations)
        print("\n".join(violations), file=sys.stderr)
        raise ValueError("Some errors in modelings. Check the above message")


if __name__ == "__main__":
    main()
