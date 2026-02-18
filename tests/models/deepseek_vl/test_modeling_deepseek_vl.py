# Copyright 2025 HuggingFace Inc.
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
"""Testing suite for the PyTorch DeepseekVL model."""

import unittest

from transformers import (
    AutoProcessor,
    DeepseekVLConfig,
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    LlamaConfig,
    SiglipVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...vlm_tester import VLMModelTest, VLMModelTester


class DeepseekVLVisionText2TextModelTester(VLMModelTester):
    base_model_class = DeepseekVLModel
    config_class = DeepseekVLConfig
    text_config_class = LlamaConfig
    vision_config_class = SiglipVisionConfig
    conditional_generation_class = DeepseekVLForConditionalGeneration

    @property
    def num_image_tokens(self):
        return (self.image_size // self.patch_size) ** 2

    def get_vision_config(self):
        config = super().get_vision_config()
        config.vision_use_head = False
        return config


@require_torch
class DeepseekVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekVLVisionText2TextModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekVLModel,
            "image-text-to-text": DeepseekVLForConditionalGeneration,
            "any-to-any": DeepseekVLForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )


@require_torch
@require_torch_accelerator
@slow
class DeepseekVLIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "deepseek-community/deepseek-vl-1.3b-chat"

    def test_model_text_generation(self):
        model = DeepseekVLForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        model.to(torch_device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        EXPECTED_TEXT = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: Describe this image.\n\nAssistant:In the image, a majestic snow leopard is captured in a moment of tranquility. The snow leopard'  # fmt: skip

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = processor.decode(output[0], skip_special_tokens=True)

        self.assertEqual(
            text,
            EXPECTED_TEXT,
        )

    def test_model_text_generation_batched(self):
        model = DeepseekVLForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        model.to(torch_device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id)

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                        },
                        {"type": "text", "text": "What animal do you see in the image?"},
                    ],
                }
            ],
        ]
        EXPECTED_TEXT = [
            "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: Describe this image.\n\nAssistant:The image depicts a snowy landscape with a focus on a bear. The bear is standing on all",  # fmt: skip
            "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: What animal do you see in the image?\n\nAssistant:I see a bear in the image.What is the significance of the color red in the",  # fmt: skip
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, padding=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = processor.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT, text)

    def test_model_text_generation_with_multi_image(self):
        model = DeepseekVLForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        model.to(torch_device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the difference between"},
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": " and "},
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                    },
                ],
            }
        ]
        EXPECTED_TEXT = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: What's the difference between and \n\nAssistant:The image is a photograph featuring two cats lying on a pink blanket. The cat on the left is"  # fmt: skip

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = processor.decode(output[0], skip_special_tokens=True)

        self.assertEqual(
            text,
            EXPECTED_TEXT,
        )
