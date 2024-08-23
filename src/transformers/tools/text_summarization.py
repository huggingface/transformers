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
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool


class TextSummarizationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextSummarizationTool

    summarizer = TextSummarizationTool()
    summarizer(long_text)
    ```
    """

    default_checkpoint = "philschmid/bart-large-cnn-samsum"
    description = (
        "This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, "
        "and returns a summary of the text."
    )
    name = "summarizer"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    inputs = ["text"]
    outputs = ["text"]

    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt", truncation=True)

    def forward(self, inputs):
        return self.model.generate(**inputs)[0]

    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
