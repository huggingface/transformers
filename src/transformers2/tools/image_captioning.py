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
from typing import TYPE_CHECKING

from ..models.auto import AutoModelForVision2Seq
from ..utils import requires_backends
from .base import PipelineTool


if TYPE_CHECKING:
    from PIL import Image


class ImageCaptioningTool(PipelineTool):
    default_checkpoint = "Salesforce/blip-image-captioning-base"
    description = (
        "This is a tool that generates a description of an image. It takes an input named `image` which should be the "
        "image to caption, and returns a text that contains the description in English."
    )
    name = "image_captioner"
    model_class = AutoModelForVision2Seq

    inputs = ["image"]
    outputs = ["text"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    def encode(self, image: "Image"):
        return self.pre_processor(images=image, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(**inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
