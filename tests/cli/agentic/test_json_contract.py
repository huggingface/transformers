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
JSON-output contract for the agentic CLI.

The agentic CLI is targeted at AI agents, so any command that exposes
``--json`` (or ``--output-json``) must produce output that round-trips through
``json.loads``. These tests are slow because they actually load tiny models —
gate them behind ``RUN_SLOW=1``.

This file is intentionally a *skeleton*: the full per-command matrix should be
filled in incrementally as tiny-random checkpoints are confirmed for each
task. Add a new entry to ``CASES`` to extend coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from transformers.testing_utils import slow

from .test_app import EXPECTED_COMMANDS, _agentic_callbacks


# Static check (fast, always runs): every command we expect to emit JSON has
# either a ``--json`` or ``--output-json`` flag in its signature. This catches
# accidental removal of the flag during a refactor without needing to load any
# model.


JSON_CAPABLE_COMMANDS = EXPECTED_COMMANDS - {
    # Commands that legitimately don't expose --json (heavy, side-effecting,
    # streaming, or output-by-design human-readable). Update if this changes.
    "train",
    "quantize",
    "export",
    "speak",
    "audio-generate",
    "multimodal-chat",
    "generate",
    "detect-watermark",
    "tokenize",  # has its own --json, see explicit test below
    "embed",  # writes raw embeddings via --output (.npy/.json)
    "depth",  # writes a depth map PNG via --output
    "inspect-forward",
}


def test_every_json_capable_command_has_a_json_flag():
    import inspect as _inspect

    callbacks_by_resolved_name = {fn.__name__.replace("_", "-"): fn for fn in _agentic_callbacks()}

    missing = []
    for name in sorted(JSON_CAPABLE_COMMANDS):
        fn = callbacks_by_resolved_name.get(name)
        if fn is None:
            continue
        params = _inspect.signature(fn).parameters
        if not ({"output_json", "json_output"} & set(params)) and "json" not in params:
            missing.append(name)
    assert not missing, f"Commands missing a JSON output flag: {missing}"


# Slow end-to-end matrix. Each case actually invokes the CLI with a tiny-random
# model, parses stdout as JSON, and asserts the top-level shape.
#
# Add new cases here as you confirm tiny-random checkpoints per task. Pin every
# case to ``hf-internal-testing/tiny-random-*`` so the suite is independent of
# README default checkpoints.


@dataclass
class JsonCase:
    name: str  # display name (also the pytest id)
    args: list[str]  # CLI args, including the command name and ``--json``
    expects_list: bool = False
    expects_dict: bool = False
    required_keys: tuple[str, ...] = field(default_factory=tuple)


CASES: list[JsonCase] = [
    # NOTE: these checkpoints should be confirmed to exist + be loadable before
    # un-skipping. Treat the list below as a working seed — fill in the rest as
    # you go.
    JsonCase(
        name="classify-supervised",
        args=[
            "classify",
            "--model",
            "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
            "--text",
            "hello world",
            "--json",
        ],
        expects_list=True,
        required_keys=("label", "score"),
    ),
    JsonCase(
        name="qa",
        args=[
            "qa",
            "--model",
            "hf-internal-testing/tiny-random-DistilBertForQuestionAnswering",
            "--question",
            "who?",
            "--context",
            "Alice met Bob.",
            "--json",
        ],
        expects_dict=True,
        required_keys=("answer", "score", "start", "end"),
    ),
    JsonCase(
        name="tokenize",
        args=[
            "tokenize",
            "--model",
            "hf-internal-testing/tiny-random-gpt2",
            "--text",
            "Hello, world!",
            "--json",
        ],
        expects_dict=True,
        required_keys=("tokens", "token_ids", "num_tokens"),
    ),
    JsonCase(
        name="inspect",
        args=[
            "inspect",
            "hf-internal-testing/tiny-random-gpt2",
            "--json",
        ],
        expects_dict=True,
        required_keys=("model_type",),
    ),
]


@slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_json_output_is_parseable(cli, extract_json, case):
    result = cli(*case.args)
    assert result.exit_code == 0, f"CLI failed:\nargs={case.args}\noutput=\n{result.output}"
    payload = extract_json(result.output)

    if case.expects_list:
        assert isinstance(payload, list) and payload, f"Expected non-empty list, got {payload!r}"
        first = payload[0]
        assert isinstance(first, dict)
        for key in case.required_keys:
            assert key in first, f"Missing key {key!r} in {first!r}"
    elif case.expects_dict:
        assert isinstance(payload, dict)
        for key in case.required_keys:
            assert key in payload, f"Missing key {key!r} in {payload!r}"
