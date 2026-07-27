# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Jinja chat template for the Onyx SFT/RL models.

The template renders the Onyx chat format for ``apply_chat_template``: system /
user / assistant / tool turns, private ``to=self`` reasoning, and the ATEM
tool-calling format. It handles both plain-string content and multimodal
list-of-parts content (image / video sentinels), so a single template serves
both; the ``multimodal`` arg is retained for API compatibility.

The canonical template text lives in the sibling ``chat_template.jinja`` file
rather than an inline Python string constant (matching the HF convention, e.g.
the Gemma release). The converter writes it to ``chat_template.jinja`` in the
output repo, which is the source of truth consumed at runtime.
"""

from __future__ import annotations

from pathlib import Path


_CHAT_TEMPLATE_FILE = Path(__file__).parent / "chat_template.jinja"

ONYX_CHAT_TEMPLATE = _CHAT_TEMPLATE_FILE.read_text(encoding="utf-8")


def build_chat_template(multimodal: bool = True) -> str:
    """Return the Onyx jinja chat template."""
    return ONYX_CHAT_TEMPLATE


__all__ = ["build_chat_template"]
