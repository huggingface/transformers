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

"""TRF003: forward() should use decorators instead of manual return_dict branching."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _function_argument_names, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def _has_return_dict_branching(function_node: ast.FunctionDef) -> bool:
    """Detect the old 'if not return_dict: return (tuple,)' pattern."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            operand = test.operand
            if isinstance(operand, ast.Name) and operand.id == "return_dict":
                return True
        if isinstance(test, ast.Name) and test.id == "return_dict":
            return True
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "return_dict":
            return True
    return False


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        forward_method = _class_methods(node).get("forward")
        if forward_method is None:
            continue
        if "return_dict" not in _function_argument_names(forward_method):
            continue
        if not _has_return_dict_branching(forward_method):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=forward_method.lineno,
                message=(
                    f"{RULE_ID}: {node.name}.forward uses old return_dict branching pattern. "
                    "Use @can_return_tuple or @capture_output decorator instead."
                ),
            )
        )

    return violations
