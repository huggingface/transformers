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
"""Template loading and validation for response_template dicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re

from ...utils import logging
from .content_parsers import CONTENT_PARSERS, validate_transform_strings


logger = logging.get_logger(__name__)


@dataclass
class ResponseTemplateField:
    name: str
    open_re: Any
    open_literals: list[str] | None
    open_literal_can_extend: bool
    close_re: Any
    close_literals: list[str] | None
    close_literal_can_extend: bool
    content: str
    content_args: dict
    repeats: bool
    optional: bool
    transform: Any
    transform_each: bool


@dataclass
class ResponseTemplate:
    defaults: dict
    fields: dict[str, ResponseTemplateField]
    start_anchor_re: Any
    implicit: str | None = None
    start_anchor_literals: list[str] | None = None

    def truncate_past_last_anchor(self, text: str) -> str:
        """When parsing responses, we can't just parse the tokens generated
        by the model. This is because chat templates often include early parts of the message such as <start_of_turn>
        or <think> tokens in the prefill. Therefore, we take the whole chat as input, but truncate to the start
        of the most recent assistant message, which can include tokens from the template as well as the output."""
        last_end: int | None = None
        for m in self.start_anchor_re.finditer(text):
            last_end = m.end()
        if last_end is None:
            kind = "start_anchor" if self.start_anchor_literals is not None else "start_anchor_pattern"
            logger.info(
                f"response_template defines {kind} but the anchor was not found "
                "in the prefix; the parser will process the entire prefix instead."
            )
            return text
        return text[last_end:]


def _compile_anchor(scope: str, field: dict, literal_key: str, pattern_key: str) -> tuple[Any, list[str] | None, bool]:
    """Compile a region's start/end anchor. An anchor can be:

    - a regex (`pattern_key`),
    - a single literal string (`literal_key`), or
    - a list of literal strings (`literal_key`), in which case any of these match

    All forms compile to a regex for matching, but we keep the literal strings around too: when present,
    the parser can emit regions faster, since it knows exactly how many trailing bytes might be a
    partial-match prefix (and that a finished literal match can never be extended by future bytes, except
    when a literal is a strict prefix of another in the same list).

    Returns `(compiled_re, literals, can_extend)`:

    - `compiled_re`: the compiled pattern.
    - `literals`: the literal strings, or `None` for a regex anchor.
    - `can_extend`: `True` iff one literal is a strict prefix of another, in which case a match at the
      buffer edge could still be lengthened by future input.
    """
    if literal_key in field and pattern_key in field:
        raise ValueError(f"{scope}: cannot specify both '{literal_key}' and '{pattern_key}'")
    if literal_key in field:
        raw = field[literal_key]
        if isinstance(raw, str):
            literals = [raw]
        elif isinstance(raw, list):
            if not raw:
                raise ValueError(f"{scope}: '{literal_key}' list must contain at least one literal")
            if not all(isinstance(s, str) for s in raw):
                raise ValueError(f"{scope}: '{literal_key}' list must contain only strings")
            literals = list(dict.fromkeys(raw))  # dedupe, preserve first-seen order
        else:
            raise ValueError(f"{scope}: '{literal_key}' must be a string or list of strings, got {type(raw).__name__}")
        if any(s == "" for s in literals):
            raise ValueError(f"{scope}: '{literal_key}' literals cannot be empty strings")
        # Sort longest-first so alternation prefers the longer alternative when both could match.
        ordered = sorted(literals, key=len, reverse=True)
        can_extend = any(a != b and a.startswith(b) for a in literals for b in literals)
        pattern = "|".join(re.escape(s) for s in ordered)
        return re.compile(pattern, re.DOTALL), literals, can_extend
    if pattern_key in field:
        try:
            return re.compile(field[pattern_key], re.DOTALL), None, False
        except re.error as e:
            raise ValueError(f"{scope}: invalid {pattern_key} regex: {e}") from e
    return None, None, False


def _validate_template_shape(spec: dict) -> None:
    """Validate the top-level response_template structure: it must be a dict of the expected version with
    only known keys, a dict of `defaults`, and a non-empty dict of `fields`. Per-field specs are checked
    separately in `_build_field`."""
    if not isinstance(spec, dict):
        raise ValueError(f"response_template must be a dict, got {type(spec).__name__}")
    version = spec.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported response_template version: {version}")
    if unknown_template_keys := set(spec) - {"version", "defaults", "fields", "start_anchor", "start_anchor_pattern"}:
        raise ValueError(f"Unknown keys in response_template: {sorted(unknown_template_keys)}")
    if not isinstance(spec.get("defaults", {}), dict):
        raise ValueError("response_template.defaults must be a dict")
    fields_raw = spec.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_template.fields must be a non-empty dict")


def _build_field(name: str, field: dict) -> ResponseTemplateField:
    """Validate a single field spec and compile it into a `ResponseTemplateField`."""

    _ALLOWED_FIELD_KEYS = {
        "open",
        "open_pattern",
        "close",
        "close_pattern",
        "content",
        "content_args",
        "repeats",
        "optional",
        "transform",
        "transform_each",
    }
    scope = f"Field '{name}'"
    if not isinstance(field, dict):
        raise ValueError(f"{scope} must be a dict")
    if unknown_field_keys := set(field) - _ALLOWED_FIELD_KEYS:
        raise ValueError(f"{scope}: unknown keys {sorted(unknown_field_keys)}")
    content = field.get("content", "text")
    if content not in CONTENT_PARSERS:
        raise ValueError(f"{scope}: unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}")
    open_re, open_literals, open_literal_can_extend = _compile_anchor(scope, field, "open", "open_pattern")
    close_re, close_literals, close_literal_can_extend = _compile_anchor(scope, field, "close", "close_pattern")
    transform = field.get("transform")
    transform_each = field.get("transform_each", False)
    if not isinstance(transform_each, bool):
        raise ValueError(f"{scope}: transform_each must be a bool, got {type(transform_each).__name__}")
    if transform_each and transform is None:
        raise ValueError(f"{scope}: transform_each is set but no transform was provided")
    if transform is not None:
        validate_transform_strings(scope, transform)
    else:
        # Named captures only reach the output through a transform, so flag any that would be silently dropped.
        captured_names = set()
        if open_re is not None:
            captured_names |= set(open_re.groupindex)
        if close_re is not None:
            captured_names |= set(close_re.groupindex)
        if captured_names:
            raise ValueError(
                f"{scope}: open_pattern/close_pattern declares named group(s) "
                f"{sorted(captured_names)}, but the field has no 'transform'. Named captures "
                f"are only surfaced through a 'transform' template (where they appear "
                f"alongside 'content'). Either add a 'transform' that uses the captures, or "
                f"remove the named groups from the pattern."
            )
    return ResponseTemplateField(
        name=name,
        open_re=open_re,
        open_literals=open_literals,
        open_literal_can_extend=open_literal_can_extend,
        close_re=close_re,
        close_literals=close_literals,
        close_literal_can_extend=close_literal_can_extend,
        content=content,
        content_args=field.get("content_args", {}),
        repeats=field.get("repeats", False),
        optional=field.get("optional", True),
        transform=transform,
        transform_each=transform_each,
    )


def load_response_template(spec: dict | ResponseTemplate) -> ResponseTemplate:
    if isinstance(spec, ResponseTemplate):
        return spec
    _validate_template_shape(spec)

    fields = {name: _build_field(name, raw) for name, raw in spec["fields"].items()}
    # A field without an open anchor is the implicit-open / leftover sink; at most one is allowed.
    implicit_fields = [name for name, field in fields.items() if field.open_re is None]
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )

    start_anchor_re, start_anchor_literals, _ = _compile_anchor(
        "response_template", spec, "start_anchor", "start_anchor_pattern"
    )
    if start_anchor_re is None:
        raise ValueError("response_template must define 'start_anchor' or 'start_anchor_pattern'.")

    return ResponseTemplate(
        defaults=dict(spec.get("defaults", {})),
        fields=fields,
        implicit=implicit_fields[0] if implicit_fields else None,
        start_anchor_re=start_anchor_re,
        start_anchor_literals=start_anchor_literals,
    )


__all__ = ["ResponseTemplate", "ResponseTemplateField", "load_response_template"]
