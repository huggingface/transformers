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

import json
import os
import sys
from typing import Any


def _is_agent_context() -> bool:
    """Detect whether we are running inside an agent (non-interactive / piped) context.

    An agent context is detected when:
    - the HF_AGENT environment variable is set, or
    - stdout is not a TTY (piped / redirected), or
    - the --json flag equivalent is detected via HF_OUTPUT_FORMAT=json.

    This is a heuristic; callers can override by passing output_json=True explicitly.
    """
    if os.environ.get("HF_AGENT"):
        return True
    if os.environ.get("HF_OUTPUT_FORMAT") == "json":
        return True
    if not sys.stdout.isatty():
        return True
    return False


def emit(result: Any, task: str | None = None, *, output_json: bool | None = None) -> str:
    """Format and return *result* for the current context (agent vs human).

    Parameters
    ----------
    result : Any
        The data to format. Typically a dict, list of dicts, or string produced
        by a CLI command.
    task : str | None
        Optional task identifier (e.g. ``"classify"``, ``"ner"``).  When
        emitting JSON in agent mode, the output is wrapped in an envelope
        ``{"task": ..., "result": ...}`` so the caller can route the response.
    output_json : bool | None
        Explicit override.  ``True`` forces JSON, ``False`` forces
        human-readable, ``None`` (default) auto-detects from the environment.

    Returns
    -------
    str
        The formatted string (the caller is responsible for printing it).
    """
    use_json = output_json if output_json is not None else _is_agent_context()

    if use_json:
        if task is not None:
            payload = {"task": task, "result": result}
        else:
            payload = result
        return json.dumps(payload, indent=2, default=str)

    return _format_human(result)


def _format_human(result: Any) -> str:
    """Format *result* as a human-readable multi-line string.

    This is the same logic previously in ``_common.format_output`` without
    the JSON branch, consolidated here as the single source of truth.
    """
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
