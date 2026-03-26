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

"""TRF010: Direct config definitions must use @strict(accept_kwargs=True)."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _has_rule_suppression,
    _has_strict_decorator,
    _is_direct_pretrained_config_subclass,
)


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("configuration_", "modular_")):
        return []

    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_direct_pretrained_config_subclass(node):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue
        if _has_strict_decorator(node):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(f"{RULE_ID}: {node.name} directly inherits PreTrainedConfig but is missing @strict."),
            )
        )

    return violations
