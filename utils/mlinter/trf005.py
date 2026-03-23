"""TRF005: _no_split_modules should be a list/tuple of non-empty strings."""

import ast
from pathlib import Path

from ._helpers import Violation, _get_class_assignments, _has_rule_suppression

RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        assignments = _get_class_assignments(node)
        value = assignments.get("_no_split_modules")
        if value is None:
            continue
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        if not isinstance(value, (ast.List, ast.Tuple)):
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(value, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name}._no_split_modules should be a list or tuple of strings.",
                )
            )
            continue

        if any(
            not isinstance(element, ast.Constant) or not isinstance(element.value, str) or not element.value
            for element in value.elts
        ):
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(value, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name}._no_split_modules should contain non-empty strings only.",
                )
            )

    return violations
