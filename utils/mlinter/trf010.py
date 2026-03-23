"""TRF010: Direct config definitions must use @strict(accept_kwargs=True)."""

import ast
from pathlib import Path

from ._helpers import (
    Violation,
    _has_rule_suppression,
    _has_strict_accept_kwargs_decorator,
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
        if _has_strict_accept_kwargs_decorator(node):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {node.name} directly inherits PreTrainedConfig but is missing "
                    "@strict(accept_kwargs=True)."
                ),
            )
        )

    return violations
