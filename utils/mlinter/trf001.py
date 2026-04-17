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

"""TRF001: Class-level config_class on <Model>PreTrainedModel should match <Model>Config naming."""

import ast
from pathlib import Path

from ._helpers import Violation, _get_class_assignments, _simple_name, full_name, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        if not node.name.endswith("PreTrainedModel"):
            continue

        assignments = _get_class_assignments(node)
        config_value = assignments.get("config_class")
        if config_value is None:
            continue
        if not isinstance(config_value, (ast.Name, ast.Attribute)):
            continue

        config_name = _simple_name(full_name(config_value))
        expected = f"{node.name.removesuffix('PreTrainedModel')}Config"
        if config_name != expected:
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(config_value, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name}.config_class is {config_name}, expected {expected}.",
                )
            )

    return violations
