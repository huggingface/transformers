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
Wiring tests for the agentic CLI.

These tests do not run inference, do not download anything, and do not need
torch. They verify that:

  * every agentic command we expect is registered on the main Typer app,
  * Typer can build a ``--help`` page for each one (i.e. argument annotations
    and lazy imports inside command bodies don't break parser construction).

Running real inference is covered separately, in the slow-marked tests.
"""

from __future__ import annotations

import pytest

import transformers.cli.transformers as cli_module
from transformers.cli.agentic import app as agentic_app


# Names registered by ``register_agentic_commands``. Update this set when adding
# or renaming a command — the test below will tell you exactly what drifted.
EXPECTED_COMMANDS: set[str] = {
    # text
    "classify",
    "ner",
    "token-classify",
    "qa",
    "table-qa",
    "summarize",
    "translate",
    "fill-mask",
    # vision / video
    "image-classify",
    "detect",
    "segment",
    "depth",
    "keypoints",
    "video-classify",
    # audio
    "transcribe",
    "audio-classify",
    "speak",
    "audio-generate",
    # multimodal
    "vqa",
    "document-qa",
    "caption",
    "ocr",
    "multimodal-chat",
    # generation
    "generate",
    "detect-watermark",
    # utilities
    "embed",
    "tokenize",
    "inspect",
    "inspect-forward",
    "benchmark-quantization",
    # heavy
    "train",
    "quantize",
    "export",
}


def _resolved_name(cmd) -> str:
    """Mirror Typer's default: ``name`` if set, else function name with ``_`` → ``-``."""
    if cmd.name:
        return cmd.name
    return cmd.callback.__name__.replace("_", "-")


def _registered_agentic_names() -> set[str]:
    """Resolve the set of command names actually registered on the main app."""
    return {_resolved_name(c) for c in cli_module.app.registered_commands if c.callback is not None}


def test_register_agentic_commands_is_called_from_main_cli():
    # Defensive: the main CLI is the only integration point. If this import
    # path moves, this test should fail loudly.
    assert hasattr(agentic_app, "register_agentic_commands")
    # Every expected command should have been wired onto the main app.
    missing = EXPECTED_COMMANDS - _registered_agentic_names()
    assert not missing, f"Agentic commands not registered on main CLI: {sorted(missing)}"


def test_no_unexpected_agentic_commands():
    # Catch silent additions: if you add a command, also add it to EXPECTED_COMMANDS.
    # We allow non-agentic commands to coexist (e.g. ``chat``, ``serve``, ``env``,
    # ``version``, ``download``) so we only check the agentic surface.
    agentic_callbacks = {c.__name__ for c in _agentic_callbacks()}
    on_app = {
        _resolved_name(c)
        for c in cli_module.app.registered_commands
        if c.callback is not None and c.callback.__name__ in agentic_callbacks
    }
    extra = on_app - EXPECTED_COMMANDS
    assert not extra, f"New agentic commands found, please update EXPECTED_COMMANDS: {sorted(extra)}"


def _agentic_callbacks():
    """All callables exposed by ``register_agentic_commands`` for set-membership checks."""
    from transformers.cli.agentic.audio import audio_classify, audio_generate, speak, transcribe
    from transformers.cli.agentic.export import export
    from transformers.cli.agentic.generate import detect_watermark, generate
    from transformers.cli.agentic.multimodal import caption, document_qa, multimodal_chat, ocr, vqa
    from transformers.cli.agentic.quantize import quantize
    from transformers.cli.agentic.text import (
        classify,
        fill_mask,
        ner,
        qa,
        summarize,
        table_qa,
        token_classify,
        translate,
    )
    from transformers.cli.agentic.train import train
    from transformers.cli.agentic.utilities import benchmark_quantization, embed, inspect, inspect_forward, tokenize
    from transformers.cli.agentic.vision import depth, detect, image_classify, keypoints, segment, video_classify

    return [
        classify,
        ner,
        token_classify,
        qa,
        table_qa,
        summarize,
        translate,
        fill_mask,
        image_classify,
        detect,
        segment,
        depth,
        keypoints,
        video_classify,
        transcribe,
        audio_classify,
        speak,
        audio_generate,
        vqa,
        document_qa,
        caption,
        ocr,
        multimodal_chat,
        generate,
        detect_watermark,
        embed,
        tokenize,
        inspect,
        inspect_forward,
        benchmark_quantization,
        train,
        quantize,
        export,
    ]


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_command_help_builds(cli, command):
    """``transformers <command> --help`` should always succeed.

    This catches:
      * broken ``Annotated[...]`` types,
      * import-time failures inside lazy imports that Typer evaluates while
        building help text,
      * accidental removal of a registered command.

    No model is loaded — Typer builds the help purely from signatures.
    """
    result = cli(command, "--help")
    assert result.exit_code == 0, f"`{command} --help` failed:\n{result.output}"
    # Typer help always contains "Usage:" — cheap sanity check.
    assert "Usage:" in result.output


def test_every_agentic_command_has_a_docstring():
    """Agent UX: every command must have a docstring (it becomes the help text)."""
    missing = [fn.__name__ for fn in _agentic_callbacks() if not (fn.__doc__ and fn.__doc__.strip())]
    assert not missing, f"Agentic commands missing docstrings: {missing}"
