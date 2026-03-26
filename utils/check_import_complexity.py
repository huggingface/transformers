#!/usr/bin/env python3
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
"""Check that `import transformers` does not pull in too many modules.

Traces the full import tree triggered by `import transformers` using a custom
``importlib.abc.MetaPathFinder`` and counts every module that gets loaded.
If the count exceeds ``MAX_IMPORT_COUNT`` the check fails, signalling a
potential regression in import speed.

Usage:
    python utils/check_import_complexity.py            # CI check mode
    python utils/check_import_complexity.py --display   # show the full import tree
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


MAX_IMPORT_COUNT = 1500


# ---------------------------------------------------------------------------
# Import-tree data structures
# ---------------------------------------------------------------------------


@dataclass
class ImportNode:
    name: str
    children: list[ImportNode] = field(default_factory=list)


class LoaderProxy(importlib.abc.Loader):
    """Wrap a real loader to track the import stack during exec_module."""

    def __init__(self, wrapped: Any, tracer: ImportTreeTracer, fullname: str):
        self._wrapped = wrapped
        self._tracer = tracer
        self._fullname = fullname

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        self._tracer.push(self._fullname)
        try:
            if hasattr(self._wrapped, "exec_module"):
                self._wrapped.exec_module(module)
            elif hasattr(self._wrapped, "load_module"):
                self._wrapped.load_module(self._fullname)
            else:
                raise ImportError(f"Loader for {self._fullname!r} has neither exec_module nor load_module")
        finally:
            self._tracer.pop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class ImportTreeFinder(importlib.abc.MetaPathFinder):
    """Intercept imports to build a parent/child tree of loaded modules."""

    def __init__(self, tracer: ImportTreeTracer, original_meta_path: list[Any]):
        self._tracer = tracer
        self._original = list(original_meta_path)

    def find_spec(self, fullname: str, path=None, target=None):
        if self._tracer.is_seen(fullname):
            return None

        for finder in self._original:
            try:
                spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
            except Exception:
                continue

            if spec is None:
                continue

            self._tracer.record(fullname)

            if spec.loader is not None:
                spec.loader = LoaderProxy(spec.loader, self._tracer, fullname)

            return spec

        return None


class ImportTreeTracer:
    def __init__(self) -> None:
        self._local = threading.local()
        self._nodes: dict[str, ImportNode] = {}
        self._roots: list[ImportNode] = []
        self._seen: set[str] = set()

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def is_seen(self, fullname: str) -> bool:
        return fullname in self._seen

    def _get_or_create(self, fullname: str) -> ImportNode:
        if fullname not in self._nodes:
            self._nodes[fullname] = ImportNode(name=fullname)
        return self._nodes[fullname]

    def record(self, fullname: str) -> None:
        if fullname in self._seen:
            return
        self._seen.add(fullname)
        node = self._get_or_create(fullname)
        stack = self._stack()
        if stack:
            parent = self._get_or_create(stack[-1])
            if all(c.name != fullname for c in parent.children):
                parent.children.append(node)
        else:
            if all(r.name != fullname for r in self._roots):
                self._roots.append(node)

    def push(self, fullname: str) -> None:
        self._stack().append(fullname)

    def pop(self) -> None:
        stack = self._stack()
        if stack:
            stack.pop()

    @property
    def count(self) -> int:
        return len(self._seen)

    @property
    def roots(self) -> list[ImportNode]:
        return self._roots


# ---------------------------------------------------------------------------
# Tracing entry-point
# ---------------------------------------------------------------------------


def trace_import(target: str) -> ImportTreeTracer:
    tracer = ImportTreeTracer()
    original_meta_path = list(sys.meta_path)
    finder = ImportTreeFinder(tracer, original_meta_path)
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module(target)
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
    return tracer


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_tree(nodes: list[ImportNode]) -> str:
    lines: list[str] = []

    def _walk(node: ImportNode, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.name}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            _walk(child, child_prefix, i == len(node.children) - 1)

    for i, root in enumerate(nodes):
        _walk(root, "", i == len(nodes) - 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Check import complexity for `import transformers`.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the full import tree (for debugging regressions).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=MAX_IMPORT_COUNT,
        help=f"Maximum allowed number of imported modules (default: {MAX_IMPORT_COUNT}).",
    )
    args = parser.parse_args()

    try:
        tracer = trace_import("transformers")
    except Exception as exc:
        print(f"ERROR: `import transformers` failed: {exc}", file=sys.stderr)
        return 1

    if args.display:
        print(format_tree(tracer.roots))
        print()
        print(f"Total modules imported: {tracer.count}")
        return 0

    if tracer.count > args.max_count:
        print(
            f"Import complexity regression: `import transformers` triggered {tracer.count} module imports "
            f"(maximum allowed: {args.max_count}).\n"
            f"\n"
            f"Run the following command to display the full import tree and identify the cause:\n"
            f"\n"
            f"    python utils/check_import_complexity.py --display\n"
        )
        return 1

    print(f"Import complexity OK: {tracer.count} modules (max {args.max_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
