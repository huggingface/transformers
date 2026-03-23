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

"""TRF007: self.post_init() should remain at the end of __init__."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, is_self_method_call, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        init_method = _class_methods(node).get("__init__")
        if init_method is None:
            continue

        post_init_index = None
        for index, statement in enumerate(init_method.body):
            if isinstance(statement, ast.Expr) and is_self_method_call(statement.value, "post_init"):
                post_init_index = index
                break
        if post_init_index is None:
            continue

        trailing_statements = init_method.body[post_init_index + 1 :]
        has_trailing_self_assignments = any(
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
                for target in (statement.targets if isinstance(statement, ast.Assign) else [statement.target])
            )
            for statement in trailing_statements
        )
        if not has_trailing_self_assignments:
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=init_method.lineno,
                message=f"{RULE_ID}: {node.name} assigns self.* after self.post_init() in __init__.",
            )
        )

    return violations
