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
"""Helper utilities for modular_model_detector."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from datetime import datetime
from functools import cache
from pathlib import Path

import transformers

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_HEADER = "\033[1;36m"
ANSI_SECTION = "\033[1;35m"
ANSI_ROW = "\033[0;37m"
ANSI_HIGHLIGHT_TOP = "\033[1;32m"
ANSI_HIGHLIGHT_OLD = "\033[1;33m"
ANSI_HIGHLIGHT_CANDIDATE = "\033[1;34m"

_RELEASE_RE = re.compile(
    r"(?:^|[\*_`\s>])(?:this|the)\s+model\s+was\s+released\s+on\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE
)


def _parse_identifier(identifier: str) -> tuple[str, str] | None:
    return tuple(identifier.split(":", 1)) if ":" in identifier else None


def _split_match_name(match_name: str) -> tuple[str | None, str | None]:
    return tuple(match_name.split(".", 1)) if "." in match_name else (None, None)


def _compute_idf(tokens_map: dict[str, list[str]]) -> tuple[dict[str, float], float]:
    doc_count = len(tokens_map)
    if doc_count == 0:
        return {}, 1.0
    df: dict[str, int] = {}
    for tokens in tokens_map.values():
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    idf = {token: math.log((doc_count + 1) / (count + 1)) + 1.0 for token, count in df.items()}
    default_idf = math.log((doc_count + 1) / 1) + 1.0
    return idf, default_idf


def _weighted_jaccard(
    query_tokens: set[str], candidate_tokens: set[str], idf_map: dict[str, float], default_idf: float
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    intersection = query_tokens & candidate_tokens
    if not intersection:
        return 0.0
    union = query_tokens | candidate_tokens
    union_weight = sum(idf_map.get(token, default_idf) for token in union)
    if union_weight <= 0:
        return 0.0
    intersection_weight = sum(idf_map.get(token, default_idf) for token in intersection)
    return intersection_weight / union_weight


def _colorize_heading(text: str) -> str:
    return f"{ANSI_HEADER}{ANSI_BOLD}{text}{ANSI_RESET}"


def _format_table(headers: list[str], rows: list[tuple[str, ...] | None], row_styles: list[str] | None = None) -> str:
    if not rows:
        return f"{ANSI_ROW}(no matches){ANSI_RESET}"

    widths = [len(header) for header in headers]
    for row in rows:
        if row is None:
            continue
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    total_width = sum(widths) + 3 * (len(headers) - 1)

    styled_rows = []
    style_idx = 0
    for row in rows:
        if row is None:
            styled_rows.append(f"{ANSI_SECTION}{'-' * total_width}{ANSI_RESET}")
            continue

        line = " | ".join(cell.ljust(widths[col_idx]) for col_idx, cell in enumerate(row))
        style = ANSI_ROW
        if row_styles and style_idx < len(row_styles) and row_styles[style_idx]:
            style = row_styles[style_idx]
        styled_rows.append(f"{style}{line}{ANSI_RESET}")
        style_idx += 1

    return "\n".join([f"{ANSI_SECTION}{header_line}{ANSI_RESET}", divider] + styled_rows)


def _parse_release_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


@cache
def _load_definition_line_map(models_root: Path, relative_path: str) -> dict[str, int]:
    file_path = models_root / relative_path
    try:
        source = file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    line_map: dict[str, int] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            line_map[node.name] = getattr(node, "lineno", None) or 1
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = f"{node.name}.{child.name}"
                        line_map[key] = getattr(child, "lineno", None) or 1
    return line_map


def _resolve_definition_location(models_root: Path, relative_path: str, definition: str) -> tuple[str, str]:
    full_path = models_root / relative_path
    line = _load_definition_line_map(models_root, relative_path).get(definition)
    line_str = str(line) if line is not None else "?"
    return str(full_path), line_str


def _build_display_path(models_root: Path, relative_path: str, match_name: str) -> tuple[str, str]:
    full_path, line = _resolve_definition_location(models_root, relative_path, match_name)
    return f"{full_path}:{line} ({match_name})", line


def build_date_data() -> dict[str, str]:
    root_dir = transformers.__file__.split("src/transformers")[0]
    root = Path(root_dir).joinpath("docs/source/en/model_doc")
    result: dict[str, str] = {}

    for md_path in root.glob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logging.info(f"Failed to read md for {md_path}")
            continue

        m = _RELEASE_RE.search(text)
        if m:
            model_id = md_path.stem
            result[model_id] = m.group(1)

    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
