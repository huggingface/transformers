#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ..models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from .tools import PipelineTool


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "distil-whisper/distil-large-v3"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    pre_processor_class = WhisperProcessor
    model_class = WhisperForConditionalGeneration

    inputs = {"audio": {"type": "audio", "description": "The audio to transcribe"}}
    output_type = "string"

    def encode(self, audio):
        return self.pre_processor(audio, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]
