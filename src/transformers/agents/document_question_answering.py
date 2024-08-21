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
import re

import numpy as np
import torch

from ..models.auto import AutoProcessor
from ..models.vision_encoder_decoder import VisionEncoderDecoderModel
from ..utils import is_vision_available
from .tools import PipelineTool


if is_vision_available():
    from PIL import Image


class DocumentQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "naver-clova-ix/donut-base-finetuned-docvqa"
    description = "This is a tool that answers a question about an document (pdf). It returns a text that contains the answer to the question."
    name = "document_qa"
    pre_processor_class = AutoProcessor
    model_class = VisionEncoderDecoderModel

    inputs = {
        "document": {
            "type": "image",
            "description": "The image containing the information. Can be a PIL Image or a string path to the image.",
        },
        "question": {"type": "text", "description": "The question in English"},
    }
    output_type = "text"

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the DocumentQuestionAnsweringTool.")

        super().__init__(*args, **kwargs)

    def encode(self, document: "Image", question: str):
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = self.pre_processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if isinstance(document, str):
            img = Image.open(document).convert("RGB")
            img_array = np.array(img).transpose(2, 0, 1)
            document = torch.tensor(img_array)
        pixel_values = self.pre_processor(document, return_tensors="pt").pixel_values

        return {"decoder_input_ids": decoder_input_ids, "pixel_values": pixel_values}

    def forward(self, inputs):
        return self.model.generate(
            inputs["pixel_values"].to(self.device),
            decoder_input_ids=inputs["decoder_input_ids"].to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.pre_processor.tokenizer.pad_token_id,
            eos_token_id=self.pre_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.pre_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        ).sequences

    def decode(self, outputs):
        sequence = self.pre_processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.pre_processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.pre_processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        sequence = self.pre_processor.token2json(sequence)

        return sequence["answer"]
