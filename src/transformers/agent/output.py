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
Transformers CLI output helpers.

Re-exports the shared ``huggingface_hub`` output singleton and adds two
small helpers used by the agentic CLI:

- ``answer(value, key="answer")`` — single-value result: bare text for humans
  (so ``transformers vqa ...`` prints the answer, not ``answer: ...``), structured
  dict in every other mode so agents keep the key.
- ``progress(message)`` — stderr status line in human mode, suppressed otherwise.

Commands use ``out.table`` / ``out.result`` / ``out.text`` / ``out.dict``
directly for everything else — see ``huggingface_hub.cli._output``.
"""

import sys

from huggingface_hub.cli._output import OutputFormatWithAuto, out


def answer(value: str, key: str = "answer") -> None:
    """Emit a single textual result.

    In human and quiet modes, prints ``value`` as a bare string. In agent/json
    modes, prints ``{key: value}`` so programmatic consumers keep the field name.
    """
    if out.mode in (OutputFormatWithAuto.human, OutputFormatWithAuto.quiet):
        out.text(value)
    else:
        out.dict({key: value})


def progress(message: str) -> None:
    """Print a progress line to stderr. Suppressed in agent/json/quiet modes."""
    if out.mode == OutputFormatWithAuto.human:
        print(f"... {message}", file=sys.stderr)
