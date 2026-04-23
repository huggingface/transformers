# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Text parsers: identity-like, for free-form content regions."""

from __future__ import annotations


class TextParser:
    name = "text"

    def parse(self, text: str, args: dict) -> str:
        return text.strip() if args.get("strip", True) else text


class RawParser:
    name = "raw"

    def parse(self, text: str, args: dict) -> str:
        return text
