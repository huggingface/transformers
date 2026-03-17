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
API contract: request types, constants, and format conversion between OpenAI and HF.
"""

from __future__ import annotations

import base64
import enum
import re
import tempfile
from io import BytesIO

from transformers.utils.import_utils import is_openai_available, is_vision_available


if is_vision_available():
    from PIL import Image

if is_openai_available():
    from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

    class TransformersCompletionCreateParamsStreaming(CompletionCreateParamsStreaming, total=False):
        generation_config: str


X_REQUEST_ID = "x-request-id"

# Fields accepted by the OpenAI schema but not yet supported.
# Receiving these raises an error to avoid silent misbehaviour.
# NOTE: "stop" is NOT in this set — we map it to stop_strings.
UNUSED_CHAT_COMPLETION_FIELDS = {
    "audio",
    "function_call",
    "functions",
    "logprobs",
    "max_completion_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "service_tier",
    "store",
    "stream_options",
    "tool_choice",
    "top_logprobs",
    "user",
    "web_search_options",
}


class Modality(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"
    STT = "STT"
    TTS = "TTS"


# ---------------------------------------------------------------------------
# Message preprocessing: OpenAI messages → processor-compatible format
# ---------------------------------------------------------------------------


def get_processor_inputs_from_messages(messages: list[dict], modality: Modality) -> list[dict]:
    """Convert OpenAI-format messages to the format expected by HF processors."""
    processor_inputs = []

    for message in messages:
        parsed = {"role": message["role"], "content": []}

        if modality == Modality.LLM:
            if isinstance(message["content"], str):
                parsed["content"] = message["content"]
            elif isinstance(message["content"], list):
                texts = [c["text"] for c in message["content"] if c["type"] == "text"]
                parsed["content"] = " ".join(texts)

        elif modality == Modality.VLM:
            if isinstance(message["content"], str):
                parsed["content"].append({"type": "text", "text": message["content"]})
            else:
                for content in message["content"]:
                    if content["type"] == "text":
                        parsed["content"].append(content)
                    elif content["type"] == "image_url":
                        url = content["image_url"]["url"]
                        if "base64" in url:
                            image_data = re.sub("^data:image/.+;base64,", "", url)
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                            file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            image.save(file.name)
                            url = file.name
                        parsed["content"].append({"type": "image", "url": url})

        processor_inputs.append(parsed)
    return processor_inputs
