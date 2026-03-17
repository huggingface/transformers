# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Base handler for endpoint handlers.
"""

from __future__ import annotations

from pydantic import BaseModel

from transformers import GenerationConfig

from ..model_manager import ModelManager


class BaseHandler:
    """Base class for endpoint handlers. Stores the model manager."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    @staticmethod
    def _apply_default_generation_config(generation_config: GenerationConfig) -> None:
        """Apply sensible serving defaults. Many models ship with too few max_new_tokens."""
        if generation_config.max_new_tokens is None or generation_config.max_new_tokens < 1024:
            generation_config.max_new_tokens = 1024

    @staticmethod
    def chunk_to_sse(chunk: BaseModel | str) -> str:
        """Format a pydantic model or string as a Server-Sent Event.

        Serializes with `exclude_none=True` — some clients (e.g. Cursor) assume
        that when a field exists in the JSON, it has data.

        Args:
            chunk: A pydantic BaseModel (ChatCompletionChunk, Response event, etc.)
                or a pre-formatted string (error paths).

        Returns:
            An SSE-formatted string: `data: {json}\\n\\n`
        """
        if isinstance(chunk, str):
            return chunk if chunk.startswith("data: ") else f"data: {chunk}\n\n"
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
