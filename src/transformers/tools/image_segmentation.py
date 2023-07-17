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
import numpy as np
import torch

from ..models.clipseg import CLIPSegForImageSegmentation
from ..utils import is_vision_available, requires_backends
from .base import PipelineTool


if is_vision_available():
    from PIL import Image


class ImageSegmentationTool(PipelineTool):
    description = (
        "This is a tool that creates a segmentation mask of an image according to a label. It cannot create an image."
        "It takes two arguments named `image` which should be the original image, and `label` which should be a text "
        "describing the elements what should be identified in the segmentation mask. The tool returns the mask."
    )
    default_checkpoint = "CIDAS/clipseg-rd64-refined"
    name = "image_segmenter"
    model_class = CLIPSegForImageSegmentation

    inputs = ["image", "text"]
    outputs = ["image"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", label: str):
        return self.pre_processor(text=[label], images=[image], padding=True, return_tensors="pt")

    def forward(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def decode(self, outputs):
        array = outputs.cpu().detach().numpy()
        array[array <= 0] = 0
        array[array > 0] = 1
        return Image.fromarray((array * 255).astype(np.uint8))
