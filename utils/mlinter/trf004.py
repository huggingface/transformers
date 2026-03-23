"""TRF004: Models must never override tie_weights."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _has_rule_suppression


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        tie_weights = _class_methods(node).get("tie_weights")
        if tie_weights is None:
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=tie_weights.lineno,
                message=(
                    f"{RULE_ID}: {node.name} overrides tie_weights. "
                    "Use _tied_weights_keys class attribute to declare tied weights instead."
                ),
            )
        )

    return violations
