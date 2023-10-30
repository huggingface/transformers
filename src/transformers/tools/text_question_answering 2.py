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


QA_PROMPT = """Here is a text containing a lot of information: '''{text}'''.

Can you answer this question about the text: '{question}'"""


class TextQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "google/flan-t5-base"
    description = (
        "This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the "
        "text where to find the answer, and `question`, which is the question, and returns the answer to the question."
    )
    name = "text_qa"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    inputs = ["text", "text"]
    outputs = ["text"]

    def encode(self, text: str, question: str):
        prompt = QA_PROMPT.format(text=text, question=question)
        return self.pre_processor(prompt, return_tensors="pt")

    def forward(self, inputs):
        output_ids = self.model.generate(**inputs)

        in_b, _ = inputs["input_ids"].shape
        out_b = output_ids.shape[0]

        return output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])[0][0]

    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
