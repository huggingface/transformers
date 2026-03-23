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

"""TRF009: modeling files should avoid importing implementation code from another model package."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, _known_model_dirs, _model_dir_name


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith("modeling_"):
        return []

    current_model = _model_dir_name(file_path)
    if current_model is None:
        return []

    violations: list[Violation] = []
    known_models = _known_model_dirs()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue

            imported_model = None
            if node.level == 0 and node.module.startswith("transformers.models."):
                remaining = node.module.split("transformers.models.", 1)[1]
                imported_model = remaining.split(".", 1)[0]
            elif node.level >= 2:
                root_name = node.module.split(".", 1)[0]
                if root_name in known_models:
                    imported_model = root_name

            if imported_model is None or imported_model in {current_model, "auto"}:
                continue

            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: {file_path.name} imports implementation code from "
                        f"`{imported_model}`. Keep model logic local to a single modeling file."
                    ),
                )
            )
            continue

        if isinstance(node, ast.Import):
            if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
                continue

            for alias in node.names:
                if not alias.name.startswith("transformers.models."):
                    continue
                remaining = alias.name.split("transformers.models.", 1)[1]
                imported_model = remaining.split(".", 1)[0]
                if imported_model in {current_model, "auto"}:
                    continue
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=node.lineno,
                        message=(
                            f"{RULE_ID}: {file_path.name} imports implementation code from "
                            f"`{imported_model}`. Keep model logic local to a single modeling file."
                        ),
                    )
                )

    return violations
