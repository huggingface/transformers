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
"""
Shared fixtures and helpers for the agentic CLI test suite.

The parent ``tests/cli/conftest.py`` already provides the ``cli`` fixture (a
``typer.testing.CliRunner`` wrapping ``transformers.cli.transformers.app``).
This conftest adds two things on top:

  * an autouse fixture that silences progress bars and library logging so that
    stdout stays JSON-clean across tests that parse it,
  * an ``extract_json`` helper that strips any residual progress noise before
    handing stdout to ``json.loads``.
"""

from __future__ import annotations

import json
import re

import pytest


# tqdm-style "[00:00<00:00, 5397.44it/s]" segments slip into stdout during
# weight loading. Without stripping them, ``json.loads`` happily latches on to
# the leading ``[`` of a progress bar.
_TQDM_BAR_RE = re.compile(r"\[\d+:\d+<[^\]]*\]")


@pytest.fixture(autouse=True)
def _quiet_hf_progress(monkeypatch):
    """Silence Hub progress bars and library logging so stdout stays JSON-clean."""
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "error")
    monkeypatch.setenv("TQDM_DISABLE", "1")


@pytest.fixture
def extract_json():
    """Return a callable that pulls the first JSON document out of CLI stdout."""

    def _extract(stdout: str):
        cleaned = _TQDM_BAR_RE.sub("", stdout)
        # Drop any partial progress segments that survived (they end with ``\r``).
        cleaned = cleaned.split("\r")[-1]
        start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1), default=-1)
        assert start != -1, f"No JSON found in CLI output:\n{stdout}"
        return json.loads(cleaned[start:])

    return _extract
