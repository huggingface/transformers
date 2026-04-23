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
        for name, f in self.fields.items():
            if f.is_implicit:
                return name
        return None


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
    "coerce",
}
_ALLOWED_TOP_KEYS = {"version", "defaults", "fields"}
_COERCE_OPTS = {"int", "float", "bool"}


def validate_spec(spec_dict: dict) -> None:
    """Validate a response_format dict. Raises ValueError on malformed specs."""
    if not isinstance(spec_dict, dict):
        raise ValueError(f"response_format must be a dict, got {type(spec_dict).__name__}")
    unknown_top = set(spec_dict) - _ALLOWED_TOP_KEYS
    if unknown_top:
        raise ValueError(f"Unknown keys in response_format: {sorted(unknown_top)}")
    version = spec_dict.get("version", SUPPORTED_VERSION)
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported response_format version: {version}")
    defaults = spec_dict.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("response_format.defaults must be a dict")
    fields_raw = spec_dict.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_format.fields must be a non-empty dict")
    implicit = 0
    for name, fd in fields_raw.items():
        if not isinstance(fd, dict):
            raise ValueError(f"Field '{name}' must be a dict")
        unknown = set(fd) - _ALLOWED_FIELD_KEYS
        if unknown:
            raise ValueError(f"Field '{name}': unknown keys {sorted(unknown)}")
        if "open" in fd and "open_pattern" in fd:
            raise ValueError(f"Field '{name}': cannot specify both 'open' and 'open_pattern'")
        if "close" in fd and "close_pattern" in fd:
            raise ValueError(f"Field '{name}': cannot specify both 'close' and 'close_pattern'")
        content = fd.get("content", "text")
        if content not in CONTENT_PARSERS:
            raise ValueError(
                f"Field '{name}': unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}"
            )
        if fd.get("coerce") is not None and fd["coerce"] not in _COERCE_OPTS:
            raise ValueError(f"Field '{name}': coerce must be one of {sorted(_COERCE_OPTS)}")
        if "open" not in fd and "open_pattern" not in fd:
            implicit += 1
        if "open_pattern" in fd:
            try:
                re.compile(fd["open_pattern"])
            except re.error as e:
                raise ValueError(f"Field '{name}': invalid open_pattern regex: {e}") from e
        if "close_pattern" in fd:
            try:
                re.compile(fd["close_pattern"])
            except re.error as e:
                raise ValueError(f"Field '{name}': invalid close_pattern regex: {e}") from e
    if implicit > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            "implicit-open / leftover sink). Found: "
            + ", ".join(n for n, fd in fields_raw.items() if "open" not in fd and "open_pattern" not in fd)
        )


def load_spec(spec_dict: dict) -> ResponseFormatSpec:
    """Validate and load a response_format dict into a ResponseFormatSpec."""
    validate_spec(spec_dict)
    fields: dict[str, FieldSpec] = {}
    for name, fd in spec_dict["fields"].items():
        fields[name] = FieldSpec(
            name=name,
            open_literal=fd.get("open"),
            open_pattern=re.compile(fd["open_pattern"], re.DOTALL) if "open_pattern" in fd else None,
            close_literal=fd.get("close"),
            close_pattern=re.compile(fd["close_pattern"], re.DOTALL) if "close_pattern" in fd else None,
            content=fd.get("content", "text"),
            content_args=fd.get("content_args", {}),
            repeats=fd.get("repeats", False),
            optional=fd.get("optional", True),
            assemble=fd.get("assemble"),
            coerce=fd.get("coerce"),
        )
    return ResponseFormatSpec(
        version=spec_dict.get("version", SUPPORTED_VERSION),
        defaults=dict(spec_dict.get("defaults", {})),
        fields=fields,
    )
