# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Spec loading and validation for response_format dicts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .content_parsers import CONTENT_PARSERS


SUPPORTED_VERSION = 1
_ALLOWED_FIELD_KEYS = {
    "open", "open_pattern", "close", "close_pattern",
    "content", "content_args", "repeats", "optional", "assemble", "coerce",
}  # fmt: skip
_ALLOWED_TOP_KEYS = {"version", "defaults", "fields"}
_COERCE_OPTS = {"int", "float", "bool"}


@dataclass
class FieldSpec:
    name: str
    open_literal: str | None = None
    open_pattern: re.Pattern | None = None
    close_literal: str | None = None
    close_pattern: re.Pattern | None = None
    content: str = "text"
    content_args: dict = field(default_factory=dict)
    repeats: bool = False
    optional: bool = True
    assemble: Any = None
    coerce: str | None = None

    @property
    def is_implicit(self) -> bool:
        return self.open_literal is None and self.open_pattern is None


@dataclass
class ResponseFormatSpec:
    version: int
    defaults: dict[str, Any]
    fields: dict[str, FieldSpec]

    @property
    def implicit_field_name(self) -> str | None:
        return next((n for n, f in self.fields.items() if f.is_implicit), None)


def _compile(name: str, key: str, pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.DOTALL)
    except re.error as e:
        raise ValueError(f"Field '{name}': invalid {key} regex: {e}") from e


def load_spec(spec_dict: dict) -> ResponseFormatSpec:
    """Validate and load a response_format dict into a ResponseFormatSpec."""
    if not isinstance(spec_dict, dict):
        raise ValueError(f"response_format must be a dict, got {type(spec_dict).__name__}")
    unknown = set(spec_dict) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in response_format: {sorted(unknown)}")
    version = spec_dict.get("version", SUPPORTED_VERSION)
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported response_format version: {version}")
    defaults = spec_dict.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("response_format.defaults must be a dict")
    fields_raw = spec_dict.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_format.fields must be a non-empty dict")

    fields: dict[str, FieldSpec] = {}
    implicit_fields: list[str] = []
    for name, fd in fields_raw.items():
        if not isinstance(fd, dict):
            raise ValueError(f"Field '{name}' must be a dict")
        bad = set(fd) - _ALLOWED_FIELD_KEYS
        if bad:
            raise ValueError(f"Field '{name}': unknown keys {sorted(bad)}")
        if "open" in fd and "open_pattern" in fd:
            raise ValueError(f"Field '{name}': cannot specify both 'open' and 'open_pattern'")
        if "close" in fd and "close_pattern" in fd:
            raise ValueError(f"Field '{name}': cannot specify both 'close' and 'close_pattern'")
        content = fd.get("content", "text")
        if content not in CONTENT_PARSERS:
            raise ValueError(
                f"Field '{name}': unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}"
            )
        coerce = fd.get("coerce")
        if coerce is not None and coerce not in _COERCE_OPTS:
            raise ValueError(f"Field '{name}': coerce must be one of {sorted(_COERCE_OPTS)}")
        if "open" not in fd and "open_pattern" not in fd:
            implicit_fields.append(name)
        fields[name] = FieldSpec(
            name=name,
            open_literal=fd.get("open"),
            open_pattern=_compile(name, "open_pattern", fd["open_pattern"]) if "open_pattern" in fd else None,
            close_literal=fd.get("close"),
            close_pattern=_compile(name, "close_pattern", fd["close_pattern"]) if "close_pattern" in fd else None,
            content=content,
            content_args=fd.get("content_args", {}),
            repeats=fd.get("repeats", False),
            optional=fd.get("optional", True),
            assemble=fd.get("assemble"),
            coerce=coerce,
        )
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )
    return ResponseFormatSpec(version=version, defaults=dict(defaults), fields=fields)


def validate_spec(spec_dict: dict) -> None:
    """Validate a response_format dict. Raises ValueError on malformed specs."""
    load_spec(spec_dict)
