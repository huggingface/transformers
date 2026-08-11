# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_processing_common import url_to_local_path


if is_torch_available():
    from transformers import (
        AutoModelForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Processor,
    )


@slow
@require_torch
@unittest.skip(reason="Update after release")  # TODO @vasqu
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"
        self.assistant_name = "google/gemma-4-E2B-it-assistant"
        self.processor = Gemma4Processor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_with_image(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)
        assistant = AutoModelForCausalLM.from_pretrained(self.assistant_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, assistant_model=assistant, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        assistant = AutoModelForCausalLM.from_pretrained(self.assistant_name, device_map=torch_device)

        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, assistant_model=assistant, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)
