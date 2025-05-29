# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llama4 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch_large_gpu,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Llama4ForConditionalGeneration,
        Llama4Processor,
    )


@slow
@require_torch_large_gpu
@require_read_token
class Llama4IntegrationTest(unittest.TestCase):
    model_id = "ll-re/Llama-4-17B-Omni-Instruct"
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]
        cls.model = Llama4ForConditionalGeneration.from_pretrained(
            "ll-re/Llama-4-17B-Omni-Instruct", device_map="auto", torch_dtype=torch.float32
        )

    def setUp(self):
        self.processor = Llama4Processor.from_pretrained("ll-re/Llama-4-17B-Omni-Instruct", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def test_model_17b_16e_fp16(self):
        EXPECTED_TEXT = [
            "The capital of France is Paris, which is located in the north-central part of the country. Paris is known for its iconic landmarks such as the",
            "Roses are red, violets are blue, and this poem is about you. Roses are red, violets are blue, and I love",
        ]

        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=100)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        print(output_text)
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_17b_16e_batch(self):
        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like',
            "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a cow"
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)
