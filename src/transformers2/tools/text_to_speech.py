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
import torch

from ..models.speecht5 import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from ..utils import is_datasets_available
from .base import PipelineTool


if is_datasets_available():
    from datasets import load_dataset


class TextToSpeechTool(PipelineTool):
    default_checkpoint = "microsoft/speecht5_tts"
    description = (
        "This is a tool that reads an English text out loud. It takes an input named `text` which should contain the "
        "text to read (in English) and returns a waveform object containing the sound."
    )
    name = "text_reader"
    pre_processor_class = SpeechT5Processor
    model_class = SpeechT5ForTextToSpeech
    post_processor_class = SpeechT5HifiGan

    inputs = ["text"]
    outputs = ["audio"]

    def setup(self):
        if self.post_processor is None:
            self.post_processor = "microsoft/speecht5_hifigan"
        super().setup()

    def encode(self, text, speaker_embeddings=None):
        inputs = self.pre_processor(text=text, return_tensors="pt", truncation=True)

        if speaker_embeddings is None:
            if not is_datasets_available():
                raise ImportError("Datasets needs to be installed if not passing speaker embeddings.")

            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    def forward(self, inputs):
        with torch.no_grad():
            return self.model.generate_speech(**inputs)

    def decode(self, outputs):
        with torch.no_grad():
            return self.post_processor(outputs).cpu().detach()
