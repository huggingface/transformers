#!/usr/bin/env python
# coding=utf-8

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

import json


def collect_encoder_boundary_tokens(config) -> list[str]:
    """Collect text tokens used as encoder boundary markers."""

    token_keys = {"start_tokens", "end_tokens", "sep_tokens"}
    collected: list[str] = []
    seen: set[str] = set()

    def _maybe_add(token):
        if not isinstance(token, str) or token == "None" or token in seen:
            return
        seen.add(token)
        collected.append(token)

    def _visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_keys:
                    _maybe_add(value)
                _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    # Encoder implementations default `end_tokens` to "\n" when the config omits it.
    _maybe_add("\n")

    for attr in ("image_encoder", "video_encoder", "sound_encoder"):
        encoder_config = getattr(config, attr, None)
        if isinstance(encoder_config, str):
            try:
                encoder_config = json.loads(encoder_config)
            except Exception:
                continue
        _visit(encoder_config)

    return collected


__all__ = ["collect_encoder_boundary_tokens"]
