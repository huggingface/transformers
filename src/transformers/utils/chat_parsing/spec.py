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

from .content_parsers import CONTENT_PARSERS


_ALLOWED_FIELD_KEYS = {
    "open", "open_pattern", "close", "close_pattern",
    "content", "content_args", "repeats", "optional", "assemble", "coerce",
}  # fmt: skip
_ALLOWED_TOP_KEYS = {"version", "defaults", "fields"}
_COERCE_OPTS = {"int", "float", "bool"}
_EOS_RE = re.compile(r"\Z")


def _compile_anchor(name: str, fd: dict, lit_key: str, pat_key: str) -> re.Pattern | None:
    if lit_key in fd and pat_key in fd:
        raise ValueError(f"Field '{name}': cannot specify both '{lit_key}' and '{pat_key}'")
    if lit_key in fd:
        lit = fd[lit_key]
        return _EOS_RE if lit == "eos" else re.compile(re.escape(lit), re.DOTALL)
    if pat_key in fd:
        try:
            return re.compile(fd[pat_key], re.DOTALL)
        except re.error as e:
            raise ValueError(f"Field '{name}': invalid {pat_key} regex: {e}") from e
    return None


def load_spec(spec_dict: dict) -> dict:
    """Validate and normalize a response_format dict. Idempotent."""
    if spec_dict.get("_normalized"):
        return spec_dict
    if not isinstance(spec_dict, dict):
        raise ValueError(f"response_format must be a dict, got {type(spec_dict).__name__}")
    unknown = set(spec_dict) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in response_format: {sorted(unknown)}")
    version = spec_dict.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported response_format version: {version}")
    defaults = spec_dict.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("response_format.defaults must be a dict")
    fields_raw = spec_dict.get("fields", {})
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise ValueError("response_format.fields must be a non-empty dict")

    fields: dict[str, dict] = {}
    implicit_fields: list[str] = []
    for name, fd in fields_raw.items():
        if not isinstance(fd, dict):
            raise ValueError(f"Field '{name}' must be a dict")
        bad = set(fd) - _ALLOWED_FIELD_KEYS
        if bad:
            raise ValueError(f"Field '{name}': unknown keys {sorted(bad)}")
        content = fd.get("content", "text")
        if content not in CONTENT_PARSERS:
            raise ValueError(
                f"Field '{name}': unknown content parser '{content}'. Available: {sorted(CONTENT_PARSERS)}"
            )
        coerce = fd.get("coerce")
        if coerce is not None and coerce not in _COERCE_OPTS:
            raise ValueError(f"Field '{name}': coerce must be one of {sorted(_COERCE_OPTS)}")
        open_re = _compile_anchor(name, fd, "open", "open_pattern")
        close_re = _compile_anchor(name, fd, "close", "close_pattern")
        if open_re is None:
            implicit_fields.append(name)
        fields[name] = {
            "name": name,
            "open_re": open_re,
            "close_re": close_re,
            "content": content,
            "content_args": fd.get("content_args", {}),
            "repeats": fd.get("repeats", False),
            "optional": fd.get("optional", True),
            "assemble": fd.get("assemble"),
            "coerce": coerce,
        }
    if len(implicit_fields) > 1:
        raise ValueError(
            "At most one field may omit 'open'/'open_pattern' (that field becomes the "
            f"implicit-open / leftover sink). Found: {', '.join(implicit_fields)}"
        )
    return {
        "_normalized": True,
        "defaults": dict(defaults),
        "fields": fields,
        "implicit": implicit_fields[0] if implicit_fields else None,
    }
