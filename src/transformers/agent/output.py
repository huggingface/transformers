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

Leverages the ``huggingface_hub`` output framework (``Output`` singleton,
``is_agent()``, ``OutputFormatWithAuto``) to automatically adapt output
between human and agent contexts.  The ``out`` singleton is the main
entry point -- it is a ``TransformersOutput`` instance that extends the
``huggingface_hub`` ``Output`` class with Transformers-specific
conveniences: task envelopes, ``progress()`` for stderr-only messages,
and ``emit()`` for structured result output.
"""

import json
import sys
from typing import Any

from huggingface_hub.cli._output import Output, OutputFormatWithAuto


class TransformersOutput(Output):
    """Transformers-specific output sink.

    Extends the ``huggingface_hub`` ``Output`` class with:
    - ``emit()`` for structured result output with optional task envelopes
    - ``progress()`` for status messages that always go to stderr
      (never polluting stdout in agent/JSON mode)
    """

    def emit(self, result: Any, task: str | None = None, *, output_json: bool | None = None) -> None:
        """Format and print *result* for the current context (agent vs human).

        When running in JSON or agent mode, the output is printed as structured
        JSON.  If *task* is provided, the JSON payload is wrapped in an envelope
        ``{"task": ..., "result": ...}`` so the caller can route the response.

        In human mode, the output is printed as a human-readable multi-line
        string (the legacy ``format_output`` style).
        """
        mode = self.mode
        if output_json is True:
            mode = OutputFormatWithAuto.json
        elif output_json is False:
            mode = OutputFormatWithAuto.human

        match mode:
            case OutputFormatWithAuto.json | OutputFormatWithAuto.agent:
                payload = {"task": task, "result": result} if task is not None else result
                indent = 2 if mode == OutputFormatWithAuto.agent else None
                print(json.dumps(payload, indent=indent, default=str))
            case OutputFormatWithAuto.human | OutputFormatWithAuto.auto:
                print(self._format_human(result))
            case OutputFormatWithAuto.quiet:
                print(self._quiet_result(result))

    def progress(self, message: str) -> None:
        """Print a progress/status message to stderr (never stdout).

        In agent/JSON mode, progress messages are suppressed entirely to keep
        stdout clean for structured output.  In human mode, the message is
        printed to stderr.
        """
        match self.mode:
            case OutputFormatWithAuto.agent | OutputFormatWithAuto.json:
                pass
            case _:
                print(f"... {message}", file=sys.stderr)

    @staticmethod
    def _format_human(result: Any) -> str:
        """Format *result* as a human-readable multi-line string."""
        if isinstance(result, list):
            lines = []
            for item in result:
                if isinstance(item, dict):
                    lines.append("  ".join(f"{k}: {v}" for k, v in item.items()))
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            lines.append("  ".join(f"{k}: {v}" for k, v in sub.items()))
                        else:
                            lines.append(str(sub))
                else:
                    lines.append(str(item))
            return "\n".join(lines)

        if isinstance(result, dict):
            return "\n".join(f"{k}: {v}" for k, v in result.items())

        return str(result)

    @staticmethod
    def _quiet_result(result: Any) -> str:
        """Extract a minimal string for quiet mode."""
        if isinstance(result, dict):
            first_value = next(iter(result.values()), "")
            return str(first_value)
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                first_value = next(iter(first.values()), "")
                return str(first_value)
            return str(first)
        return str(result)


out = TransformersOutput()


def emit(result: Any, task: str | None = None, *, output_json: bool | None = None) -> str:
    """Format and return *result* for the current context (agent vs human).

    Convenience wrapper that returns the formatted string instead of printing
    it.  This preserves backward compatibility with call sites that do
    ``print(emit(...))``.

    Prefer calling ``out.emit()`` directly in new code -- it prints to stdout
    directly and respects the full ``Output`` mode hierarchy (including quiet).
    """
    mode = out.mode
    if output_json is True:
        mode = OutputFormatWithAuto.json
    elif output_json is False:
        mode = OutputFormatWithAuto.human

    match mode:
        case OutputFormatWithAuto.json | OutputFormatWithAuto.agent:
            payload = {"task": task, "result": result} if task is not None else result
            indent = 2 if mode == OutputFormatWithAuto.agent else None
            return json.dumps(payload, indent=indent, default=str)
        case OutputFormatWithAuto.human | OutputFormatWithAuto.auto:
            return out._format_human(result)
        case OutputFormatWithAuto.quiet:
            return out._quiet_result(result)
