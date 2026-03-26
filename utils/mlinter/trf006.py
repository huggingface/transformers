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

"""TRF006: forward with cache arguments should reference cache control/state variables."""

import ast
from pathlib import Path

from ._helpers import Violation, _class_methods, _function_argument_names, _function_uses_name, _has_rule_suppression


RULE_ID = ""  # Set by discovery


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        forward_method = _class_methods(node).get("forward")
        if forward_method is None:
            continue

        arg_names = _function_argument_names(forward_method)
        cache_state_args = {"past_key_values", "past_key_value"}
        has_cache_state_arg = bool(arg_names.intersection(cache_state_args))
        if not has_cache_state_arg:
            continue

        if "use_cache" in arg_names and _function_uses_name(forward_method, "use_cache"):
            continue
        if any(_function_uses_name(forward_method, arg_name) for arg_name in cache_state_args):
            continue

        violations.append(
            Violation(
                file_path=file_path,
                line_number=forward_method.lineno,
                message=(
                    f"{RULE_ID}: {node.name}.forward exposes past_key_values/use_cache but does not reference them."
                ),
            )
        )

    return violations
