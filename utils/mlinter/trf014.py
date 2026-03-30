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

"""TRF014: `trust_remote_code` should never be used in native model integrations."""

import ast
from pathlib import Path

from ._helpers import Violation


RULE_ID = ""  # Set by discovery


class TrustRemoteCodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.violations: list[Violation] = []

    def _add(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            Violation(
                file_path=self.file_path,
                line_number=node.lineno,
                message=f"{RULE_ID}: {message}",
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        """
        Three cases covered by this
            1. `foo(..., trust_remote_code=...)`
            2. `foo(**{..., "trust_remote_code": ...})`
            3. `foo(**dict(trust_remote_code=...))`

        Not covered:
               `kwargs = {"trust_remote_code": True}; foo(**kwargs)`
        """
        for keyword in node.keywords:
            if keyword.arg == "trust_remote_code":
                self._add(node, "`trust_remote_code` must not be passed as a keyword argument.")

            elif keyword.arg is None:
                value = keyword.value

                if isinstance(value, ast.Dict):
                    for key in value.keys:
                        if isinstance(key, ast.Constant) and key.value == "trust_remote_code":
                            self._add(node, "`trust_remote_code` must not be passed through `**kwargs`.")

                elif isinstance(value, ast.Call):
                    if isinstance(value.func, ast.Name) and value.func.id == "dict":
                        for kw in value.keywords:
                            if kw.arg == "trust_remote_code":
                                self._add(
                                    node,
                                    "`trust_remote_code` must not be passed through `**kwargs` (dict constructor).",
                                )

        self.generic_visit(node)


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    visitor = TrustRemoteCodeVisitor(file_path)
    visitor.visit(tree)
    return visitor.violations
