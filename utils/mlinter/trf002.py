"""TRF002: base_model_prefix should be a non-empty canonical string."""

import ast
from pathlib import Path

from ._helpers import Violation, _get_class_assignments, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        assignments = _get_class_assignments(node)
        prefix_value = assignments.get("base_model_prefix")
        if prefix_value is None:
            continue
        if not (isinstance(prefix_value, ast.Constant) and isinstance(prefix_value.value, str)):
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(prefix_value, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name}.base_model_prefix should be a string literal.",
                )
            )
            continue
        if not prefix_value.value.strip() or any(char.isspace() for char in prefix_value.value):
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(prefix_value, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name}.base_model_prefix should be a non-empty canonical token.",
                )
            )

    return violations
