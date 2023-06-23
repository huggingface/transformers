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

from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer
from .base import PipelineTool


class TextClassificationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """

    default_checkpoint = "facebook/bart-large-mnli"
    description = (
        "This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which "
        "should be the text to classify, and `labels`, which should be the list of labels to use for classification. "
        "It returns the most likely label in the list of provided `labels` for the input text."
    )
    name = "text_classifier"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    inputs = ["text", ["text"]]
    outputs = ["text"]

    def setup(self):
        super().setup()
        config = self.model.config
        self.entailment_id = -1
        for idx, label in config.id2label.items():
            if label.lower().startswith("entail"):
                self.entailment_id = int(idx)
        if self.entailment_id == -1:
            raise ValueError("Could not determine the entailment ID from the model config, please pass it at init.")

    def encode(self, text, labels):
        self._labels = labels
        return self.pre_processor(
            [text] * len(labels),
            [f"This example is {label}" for label in labels],
            return_tensors="pt",
            padding="max_length",
        )

    def decode(self, outputs):
        logits = outputs.logits
        label_id = torch.argmax(logits[:, 2]).item()
        return self._labels[label_id]
