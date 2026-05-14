# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Template loading and validation for response_template dicts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...utils import logging
from .content_parsers import CONTENT_PARSERS


logger = logging.get_logger(__name__)


@dataclass
class ResponseTemplateField:
    name: str
    open_re: re.Pattern | None
    open_lits: list[str] | None
    open_lit_can_extend: bool
    close_re: re.Pattern | None
    close_lits: list[str] | None
    close_lit_can_extend: bool
    content: str
    content_args: dict
    repeats: bool
    optional: bool
    transform: str | None


@dataclass
class ResponseTemplate:
    defaults: dict
    fields: dict[str, ResponseTemplateField]
    implicit: str | None = None
    start_anchor_re: re.Pattern | None = None
    start_anchor_lits: list[str] | None = None

    def truncate_past_last_anchor(self, text: str, *, log_if_missing: bool = True) -> str:
        """When parsing responses, we can't just parse the tokens generated
        by the model. This is because chat templates often include early parts of the message such as <start_of_turn>
        or <think> tokens in the prefill. Therefore, we take the whole chat as input, but truncate to the start
        of the most recent assistant message, which can include tokens from the template as well as the output."""
        if self.start_anchor_re is None:
            return text
        last_end: int | None = None
        for m in self.start_anchor_re.finditer(text):
            last_end = m.end()
        if last_end is None:
            if log_if_missing:
                kind = "start_anchor" if self.start_anchor_lits is not None else "start_anchor_pattern"
                logger.warning_once(
                    f"response_template defines {kind} but the anchor was not found "
                    "in the prefix; the parser will process the entire prefix instead."
                )
            return text
        return text[last_end:]


def _compile_anchor(
    scope: str, field: dict, lit_key: str, pat_key: str
) -> tuple[re.Pattern | None, list[str] | None, bool]:
    """The start and end anchors for a region can be either regexes, a single literal string, or a list of
    literal strings (an "any of these" alternation). Although all forms get compiled to a regex for matching,
    we keep the literal strings around too - when present, the parser can emit regions more quickly, since it
    knows exactly how many trailing bytes might be a partial-match prefix (and that a finished literal match
    can never be extended by future bytes, with the one exception of literals that are prefixes of others in
    the same list).

    Returns `(compiled_re, literals, can_extend)` where `literals` is `None` for regex anchors and a list of
    literal strings otherwise; `can_extend` is `True` iff one literal in the list is a strict prefix of
    another, in which case a successful match at the buffer edge could still be lengthened by future input.
    """
    if lit_key in field and pat_key in field:
        raise ValueError(f"{scope}: cannot specify both '{lit_key}' and '{pat_key}'")
    if lit_key in field:
        raw = field[lit_key]
        if isinstance(raw, str):
            lits = [raw]
        elif isinstance(raw, list):
            if not raw:
                raise ValueError(f"{scope}: '{lit_key}' list must contain at least one literal")
            if not all(isinstance(s, str) for s in raw):
                raise ValueError(f"{scope}: '{lit_key}' list must contain only strings")
            lits = list(dict.fromkeys(raw))  # dedupe, preserve first-seen order
        else:
            raise ValueError(f"{scope}: '{lit_key}' must be a string or list of strings, got {type(raw).__name__}")
        if any(s == "" for s in lits):
            raise ValueError(f"{scope}: '{lit_key}' literals cannot be empty strings")
        # `"eos"` is a magic literal mapping to end-of-stream; only meaningful on its own.
        if "eos" in lits:
            if len(lits) > 1:
                raise ValueError(f"{scope}: the 'eos' literal cannot be combined with other literals")
            return re.compile(r"\Z"), lits, False
        # Sort longest-first so alternation prefers the longer alternative when both could match.
        ordered = sorted(lits, key=len, reverse=True)
        can_extend = any(a != b and a.startswith(b) for a in lits for b in lits)
        pattern = "|".join(re.escape(s) for s in ordered)
        return re.compile(pattern, re.DOTALL), lits, can_extend
    if pat_key in field:
        try:
            return re.compile(field[pat_key], re.DOTALL), None, False
        except re.error as e:
            raise ValueError(f"{scope}: invalid {pat_key} regex: {e}") from e
    return None, None, False


def load_response_template(spec: dict | ResponseTemplate) -> ResponseTemplate:
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
    }
    # Error checking / validation block
    if isinstance(spec, ResponseTemplate):
        return spec
    if not isinstance(spec, dict):
        raise ValueError(f"response_template must be a dict, got {type(spec).__name__}")
    if unknown_template_keys := set(spec) - {"version", "defaults", "fields", "start_anchor", "start_anchor_pattern"}:
        raise ValueError(f"Unknown keys in response_template: {sorted(unknown_template_keys)}")
    version = spec.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported response_template version: {version}")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("response_template.defaults must be a dict")
    fields_raw = spec.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_template.fields must be a non-empty dict")

    fields: dict[str, ResponseTemplateField] = {}
    implicit_fields: list[str] = []
    for name, field in fields_raw.items():
        if not isinstance(field, dict):
            raise ValueError(f"Field '{name}' must be a dict")
        if unknown_field_keys := set(field) - _ALLOWED_FIELD_KEYS:
            raise ValueError(f"Field '{name}': unknown keys {sorted(unknown_field_keys)}")
        content = field.get("content", "text")
        if content not in CONTENT_PARSERS:
            raise ValueError(
                f"Field '{name}': unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}"
            )
        open_re, open_lits, open_lit_can_extend = _compile_anchor(f"Field '{name}'", field, "open", "open_pattern")
        close_re, close_lits, close_lit_can_extend = _compile_anchor(
            f"Field '{name}'", field, "close", "close_pattern"
        )
        if open_re is None:
            implicit_fields.append(name)
        transform = field.get("transform")
        if transform is None:
            captured_names = set()
            if open_re is not None:
                captured_names |= set(open_re.groupindex)
            if close_re is not None:
                captured_names |= set(close_re.groupindex)
            if captured_names:
                raise ValueError(
                    f"Field '{name}': open_pattern/close_pattern declares named group(s) "
                    f"{sorted(captured_names)}, but the field has no 'transform'. Named captures "
                    f"are only surfaced through a 'transform' jmespath expression (where they appear "
                    f"alongside 'content'). Either add a 'transform' that uses the captures, or "
                    f"remove the named groups from the pattern."
                )
        fields[name] = ResponseTemplateField(
            name=name,
            open_re=open_re,
            open_lits=open_lits,
            open_lit_can_extend=open_lit_can_extend,
            close_re=close_re,
            close_lits=close_lits,
            close_lit_can_extend=close_lit_can_extend,
            content=content,
            content_args=field.get("content_args", {}),
            repeats=field.get("repeats", False),
            optional=field.get("optional", True),
            transform=transform,
        )
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )
    start_anchor_re, start_anchor_lits, _ = _compile_anchor(
        "response_template", spec, "start_anchor", "start_anchor_pattern"
    )
    return ResponseTemplate(
        defaults=dict(defaults),
        fields=fields,
        implicit=implicit_fields[0] if implicit_fields else None,
        start_anchor_re=start_anchor_re,
        start_anchor_lits=start_anchor_lits,
    )


__all__ = ["ResponseTemplate", "ResponseTemplateField", "load_response_template"]
