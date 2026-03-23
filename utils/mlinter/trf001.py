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
