# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, AyaVisionProcessor
from transformers.testing_utils import require_read_token, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from transformers import GotOcr2ImageProcessor


@require_read_token
@require_vision
class AyaVisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AyaVisionProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = GotOcr2ImageProcessor(
            do_resize=True,
            size={"height": 20, "width": 20},
            max_patches=2,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-vision-8b", padding_side="left")
        processor_kwargs = self.prepare_processor_dict()
        processor = AyaVisionProcessor.from_pretrained(
            "CohereForAI/aya-vision-8b",
            image_processor=image_processor,
            tokenizer=tokenizer,
            **processor_kwargs,
        )
        processor.save_pretrained(self.tmpdirname)

    def prepare_processor_dict(self):
        return {"patch_size": 10, "img_size": 20}

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # todo: yoni, fix this test
    @unittest.skip("Chat template has long system prompt")
    def test_chat_template_accepts_processing_kwargs(self, **kwargs):
        pass

    # Override as AyaVisionProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <image>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <image>"]
        return ["lower newer <image>", "<image> upper older longer string"] + ["<image> lower newer"] * (
            batch_size - 2
        )

    @require_torch
    def test_process_interleaved_images_videos(self):
        processor = self.get_processor()

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                        },
                        {
                            "type": "image",
                            "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                        },
                        {"type": "text", "text": "What are the differences between these two images?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://llava-vl.github.io/static/images/view.jpg",
                        },
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                }
            ],
        ]

        inputs_batched = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Process non batched inputs to check if the pixel_values and input_ids are reconstructed in the correct order when batched together
        images_patches_index = 0
        for i, message in enumerate(messages):
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            # We slice with [-inputs["input_ids"].shape[1] :] as the input_ids are left padded
            torch.testing.assert_close(
                inputs["input_ids"][0], inputs_batched["input_ids"][i][-inputs["input_ids"].shape[1] :]
            )
            torch.testing.assert_close(
                inputs["pixel_values"],
                inputs_batched["pixel_values"][
                    images_patches_index : images_patches_index + inputs["pixel_values"].shape[0]
                ],
            )
            images_patches_index += inputs["pixel_values"].shape[0]
