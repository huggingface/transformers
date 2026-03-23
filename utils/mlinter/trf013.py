"""TRF013: PreTrainedModel __init__ must call self.post_init()."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, full_name, is_self_method_call, is_super_method_call


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        base_names = []
        for parent in node.bases:
            try:
                base_names.append(full_name(parent))
            except ValueError:
                continue

        if not any(base_name.endswith("PreTrainedModel") for base_name in base_names):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        for sub_node in node.body:
            if not (isinstance(sub_node, ast.FunctionDef) and sub_node.name == "__init__"):
                continue

            for statement in ast.walk(sub_node):
                if is_self_method_call(statement, method="post_init"):
                    break
                elif "modular_" in str(file_path) and is_super_method_call(statement, method="__init__"):
                    break
            else:
                violations.append(
                    Violation(
                        file_path=file_path,
                        line_number=sub_node.lineno,
                        message=f"{RULE_ID}: `__init__` of {node.name} does not call `self.post_init`",
                    )
                )
            break

    return violations
