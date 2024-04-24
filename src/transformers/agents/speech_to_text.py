#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

from .tools import Tool
from huggingface_hub import InferenceClient


class SpeechToTextTool(Tool):
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"

    inputs = {"audio": {"type": Any, "description": "The audio to transcribe"}}
    output_type = str

    def __init__(self, checkpoint=None):
        super().__init__(checkpoint=checkpoint)
        self.client = InferenceClient()

    def __call__(self, audio):
        return self.client.automatic_speech_recognition(audio).text
