# Copyright 2025 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE 4.5 model."""

import unittest

from transformers import (
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Exaone4_5_ForConditionalGeneration,
        Exaone4_5_Processor,
    )


@require_torch
class Exaone4_5_IntegrationTest(unittest.TestCase):
    # model_id = "LGAI-EXAONE/EXAONE-4.5-33B"
    model_id = "/ex_disk/jwhwang/served_model/EXAONE-4.5-33B"
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = Exaone4_5_ForConditionalGeneration.from_pretrained(cls.model_id, device_map="auto")
        cls.processor = Exaone4_5_Processor.from_pretrained(cls.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [70045, 1109, 115406, 16943, 11697, 115365, 19816, 12137, 375]
        input_ids = torch.tensor([input_ids]).to(self.model.model.language_model.embed_tokens.weight.device)

        with torch.no_grad():
            out = self.model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor(
            [[46.0681, 45.8148, 71.2274, 36.8956, 44.1011, 21.7848, 28.1107, 62.5165, 45.9560]]
        )
        EXPECTED_SLICE = torch.tensor(
            [43.5000, 44.0000, 43.7500, 46.0000, 50.5000, 47.2500, 47.5000, 47.5000, 46.7500, 47.2500]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

    @slow
    def test_model_generation_text_only(self):
        EXPECTED_TEXT = (
            '\nTell me about the Miracle on the Han river.\n\n<think>\n\n</think>\n\nThe **"Miracle on the Han River"**'
            " is a term used to describe the rapid economic development and industrialization that South Korea experienced"
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Tell me about the Miracle on the Han river."}]}
        ]
        input_ids = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.model.language_model.embed_tokens.weight.device)

        generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=20, do_sample=False)
        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        print(text)
        self.assertEqual(EXPECTED_TEXT, text)

    @slow
    def test_model_generation_image_text(self):
        IMAGE_URL = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )
        EXPECTED_TEXT = "\n\nDescribe the image.\n\n<think>\n\n</think>\n\nThe image captures a young, fluffy wild cat\u2014likely a lynx kitten or bobcat cub"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image = load_image(IMAGE_URL).convert("RGB")

        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(torch_device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        print(text)
        self.assertEqual(EXPECTED_TEXT, text)
