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
Register all agentic CLI commands on a Typer app.

This is the single integration point between the agentic CLI and the
main ``transformers`` CLI. It exposes one function:

    ``register_agentic_commands(app)``

which adds ~30 commands to the given Typer app. The main CLI calls this
from ``transformers.cli.transformers``. Removing that one call disables
the entire agentic module with no other changes required.
"""

from .audio import audio_classify, audio_generate, speak, transcribe
from .export import export
from .generate import detect_watermark, generate
from .multimodal import caption, document_qa, multimodal_chat, ocr, vqa
from .quantize import quantize
from .text import classify, fill_mask, ner, qa, summarize, table_qa, token_classify, translate
from .train import train
from .utilities import benchmark_quantization, embed, inspect, inspect_forward, tokenize
from .vision import depth, detect, image_classify, keypoints, segment, video_classify


def register_agentic_commands(app):
    """Register all agentic CLI commands on the given Typer app instance."""
    app.command()(classify)
    app.command()(ner)
    app.command(name="token-classify")(token_classify)
    app.command()(qa)
    app.command(name="table-qa")(table_qa)
    app.command()(summarize)
    app.command()(translate)
    app.command(name="fill-mask")(fill_mask)
    app.command(name="image-classify")(image_classify)
    app.command()(detect)
    app.command()(segment)
    app.command()(depth)
    app.command()(keypoints)
    app.command(name="video-classify")(video_classify)
    app.command()(transcribe)
    app.command(name="audio-classify")(audio_classify)
    app.command()(speak)
    app.command(name="audio-generate")(audio_generate)
    app.command()(vqa)
    app.command(name="document-qa")(document_qa)
    app.command()(caption)
    app.command()(ocr)
    app.command(name="multimodal-chat")(multimodal_chat)
    app.command()(generate)
    app.command(name="detect-watermark")(detect_watermark)
    app.command()(embed)
    app.command()(tokenize)
    app.command(name="inspect")(inspect)
    app.command(name="inspect-forward")(inspect_forward)
    app.command(name="benchmark-quantization")(benchmark_quantization)
    app.command()(train)
    app.command()(quantize)
    app.command()(export)
