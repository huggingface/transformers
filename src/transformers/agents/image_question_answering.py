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
from PIL import Image

from ..models.auto import AutoModelForVisualQuestionAnswering, AutoProcessor
from ..utils import requires_backends
from .tools import PipelineTool


class ImageQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "dandelin/vilt-b32-finetuned-vqa"
    description = (
        "This is a tool that answers a question about an image. It returns a text that is the answer to the question."
    )
    name = "image_qa"
    pre_processor_class = AutoProcessor
    model_class = AutoModelForVisualQuestionAnswering

    inputs = {
        "image": {
            "type": "image",
            "description": "The image containing the information. Can be a PIL Image or a string path to the image.",
        },
        "question": {"type": "string", "description": "The question in English"},
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", question: str):
        return self.pre_processor(image, question, return_tensors="pt")

    def forward(self, inputs):
        with torch.no_grad():
            return self.model(**inputs).logits

    def decode(self, outputs):
        idx = outputs.argmax(-1).item()
        return self.model.config.id2label[idx]
