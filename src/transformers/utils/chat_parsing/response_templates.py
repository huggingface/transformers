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
from typing import Any

from ...utils import logging
from .content_parsers import CONTENT_PARSERS


logger = logging.get_logger(__name__)


@dataclass
class ResponseTemplateField:
    name: str
    open_re: re.Pattern | None
    open_lit: str | None
    close_re: re.Pattern | None
    close_lit: str | None
    content: str
    content_args: dict
    repeats: bool
    optional: bool
    assemble: Any


@dataclass
class ResponseTemplate:
    defaults: dict
    fields: dict[str, ResponseTemplateField]
    implicit: str | None = None
    start_anchor_re: re.Pattern | None = None
    start_anchor_lit: str | None = None

    def truncate_past_last_anchor(self, text: str, *, log_if_missing: bool = True) -> str:
        """When parsing responses, we can't just parse the tokens generated
        by the model. This is because chat templates often include early parts of the message such as <start_of_turn>
        or <think> tokens in the prefill. Therefore, we take the whole chat as input, but truncate to the start
        of the most recent assistant message, which can include tokens from the template as well as the output."""
        if self.start_anchor_re is None:
            return text
        if self.start_anchor_lit is not None:
            idx = text.rfind(self.start_anchor_lit)
            if idx < 0:
                if log_if_missing:
                    logger.warning_once(
                        "response_template defines start_anchor but the anchor was not found "
                        "in the prefix; the parser will process the entire prefix instead."
                    )
                return text
            return text[idx + len(self.start_anchor_lit) :]
        last_end: int | None = None
        for m in self.start_anchor_re.finditer(text):
            last_end = m.end()
        if last_end is None:
            if log_if_missing:
                logger.warning_once(
                    "response_template defines start_anchor_pattern but the anchor was not "
                    "found in the prefix; the parser will process the entire prefix instead."
                )
            return text
        return text[last_end:]


def _compile_anchor(scope: str, field: dict, lit_key: str, pat_key: str) -> tuple[re.Pattern | None, str | None]:
    """The start and end anchors for a region can be either regexes or literal strings. Although both get compiled
    to a regex for actual matching, we keep the literal string around too - when it's present, we can emit
    regions much more quickly, rather than needing to keep a safety buffer as we do with regexes."""
    if lit_key in field and pat_key in field:
        raise ValueError(f"{scope}: cannot specify both '{lit_key}' and '{pat_key}'")
    if lit_key in field:
        lit = field[lit_key]
        if lit == "eos":
            return re.compile(r"\Z"), None
        return re.compile(re.escape(lit), re.DOTALL), lit
    if pat_key in field:
        try:
            return re.compile(field[pat_key], re.DOTALL), None
        except re.error as e:
            raise ValueError(f"{scope}: invalid {pat_key} regex: {e}") from e
    return None, None


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
        "assemble",
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
        open_re, open_lit = _compile_anchor(f"Field '{name}'", field, "open", "open_pattern")
        close_re, close_lit = _compile_anchor(f"Field '{name}'", field, "close", "close_pattern")
        if open_re is None:
            implicit_fields.append(name)
        fields[name] = ResponseTemplateField(
            name=name,
            open_re=open_re,
            open_lit=open_lit,
            close_re=close_re,
            close_lit=close_lit,
            content=content,
            content_args=field.get("content_args", {}),
            repeats=field.get("repeats", False),
            optional=field.get("optional", True),
            assemble=field.get("assemble"),
        )
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )
    start_anchor_re, start_anchor_lit = _compile_anchor(
        "response_template", spec, "start_anchor", "start_anchor_pattern"
    )
    return ResponseTemplate(
        defaults=dict(defaults),
        fields=fields,
        implicit=implicit_fields[0] if implicit_fields else None,
        start_anchor_re=start_anchor_re,
        start_anchor_lit=start_anchor_lit,
    )


__all__ = ["ResponseTemplate", "ResponseTemplateField", "load_response_template"]
