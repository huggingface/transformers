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
Agent-aware output formatting for the Transformers CLI.

Builds on the ``huggingface_hub`` output framework (``Output`` singleton,
``is_agent()``, ``OutputFormatWithAuto``). Mode is resolved once at CLI
startup via the top-level ``--format`` option (see ``cli/transformers.py``).
"""

import json
import sys
from typing import Any

from huggingface_hub.cli._output import Output, OutputFormatWithAuto


class TransformersOutput(Output):
    """Transformers-specific output sink.

    Adds two methods on top of ``huggingface_hub.Output``:
    - ``emit(result, task)`` — structured result, wrapped in a ``{"task", "result"}``
      envelope for agent/json consumers, or flattened to a readable form for humans.
    - ``progress(message)`` — status line to stderr in human mode, suppressed otherwise.
    """

    def emit(self, result: Any, task: str) -> None:
        """Print a task result, shaped for the current output mode."""
        match self.mode:
            case OutputFormatWithAuto.json:
                print(json.dumps({"task": task, "result": result}, default=str))
            case OutputFormatWithAuto.agent:
                print(json.dumps({"task": task, "result": result}, indent=2, default=str))
            case OutputFormatWithAuto.quiet:
                print(_quiet(result))
            case _:
                print(_human(result))

    def progress(self, message: str) -> None:
        """Print a progress line to stderr. Suppressed in agent/json modes."""
        if self.mode in (OutputFormatWithAuto.agent, OutputFormatWithAuto.json, OutputFormatWithAuto.quiet):
            return
        print(f"... {message}", file=sys.stderr)


def _human(result: Any) -> str:
    """Readable multi-line rendering for terminal users."""
    if isinstance(result, dict):
        # Single-key dicts (e.g. {"answer": "..."}) render as the bare value —
        # commands like vqa/caption/ocr/transcribe are read, not parsed.
        if len(result) == 1:
            return str(next(iter(result.values())))
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    if isinstance(result, list):
        lines = []
        for item in result:
            if isinstance(item, dict):
                lines.append("  ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(result)


def _quiet(result: Any) -> str:
    """Single-value rendering for shell pipelines."""
    if isinstance(result, dict):
        return str(next(iter(result.values()), ""))
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(next(iter(first.values()), ""))
        return str(first)
    return str(result)


out = TransformersOutput()
