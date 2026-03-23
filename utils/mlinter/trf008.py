"""TRF008: Doc decorators should avoid empty add_start_docstrings usage."""

import ast
from pathlib import Path

from ._helpers import Violation, _simple_name, full_name, iter_pretrained_classes


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, (ast.Name, ast.Attribute))
                and _simple_name(full_name(decorator.func)) == "add_start_docstrings"
            ):
                continue
            has_non_empty_string_arg = any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip()
                for arg in decorator.args
            )
            if has_non_empty_string_arg:
                continue

            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=getattr(decorator, "lineno", node.lineno),
                    message=f"{RULE_ID}: {node.name} uses add_start_docstrings without non-empty docstring arguments.",
                )
            )
            break

    return violations
