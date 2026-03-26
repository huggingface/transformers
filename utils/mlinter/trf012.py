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

"""TRF012: _init_weights must use init primitives, not in-place ops on module weights."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_init_weights"):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        args = node.args.args
        if len(args) < 2 or getattr(args[0], "arg", None) != "self" or getattr(args[1], "arg", None) != "module":
            continue

        for sub_node in ast.walk(node):
            if not (isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Attribute)):
                continue
            is_inplace_ops = sub_node.func.attr.endswith("_")
            is_on_module_weight = isinstance(
                sub_node.func.value, (ast.Name, ast.Attribute)
            ) and "module." in full_name(sub_node.func.value)
            if is_inplace_ops and is_on_module_weight:
                if _has_rule_suppression(source_lines, RULE_ID, sub_node.lineno):
                    continue
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=sub_node.lineno,
                        message=(
                            f"{RULE_ID}: `_init_weights(self, module)` uses an in-place operation on a module's "
                            "weight. Please use the `init` functions primitives instead, usually imported as "
                            "`from ... import initialization as init`"
                        ),
                    )
                )

    return violations
