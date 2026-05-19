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
"""Testing suite for the PyTorch Gemma3 model."""

import tempfile
import unittest
from io import BytesIO

import requests
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    Gemma3TextConfig,
    ShieldGemma2Config,
    SiglipVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    from transformers import ShieldGemma2ForImageClassification, ShieldGemma2Processor


@require_torch
class ShieldGemma2ModelTest(unittest.TestCase):
    def get_config(self):
        text_config = Gemma3TextConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
        )
        vision_config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=8,
            patch_size=4,
        )
        return ShieldGemma2Config(
            text_config=text_config,
            vision_config=vision_config,
            mm_tokens_per_image=4,
            image_token_index=98,
        )

    def test_sdpa_can_dispatch_composite_models(self):
        config = self.get_config()
        model = ShieldGemma2ForImageClassification(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            model_sdpa = ShieldGemma2ForImageClassification.from_pretrained(
                tmpdirname,
                attn_implementation="sdpa",
            )
            model_eager = ShieldGemma2ForImageClassification.from_pretrained(
                tmpdirname,
                attn_implementation="eager",
            )

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
        self.assertTrue(model_sdpa.model.config._attn_implementation == "sdpa")
        self.assertTrue(model_sdpa.model.model.language_model.config._attn_implementation == "sdpa")
        self.assertTrue(model_sdpa.model.model.vision_tower.config._attn_implementation == "sdpa")

        self.assertTrue(model_eager.config._attn_implementation == "eager")
        self.assertTrue(model_eager.model.config._attn_implementation == "eager")
        self.assertTrue(model_eager.model.model.language_model.config._attn_implementation == "eager")
        self.assertTrue(model_eager.model.model.vision_tower.config._attn_implementation == "eager")


@slow
@require_torch_accelerator
class ShieldGemma2IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        model = ShieldGemma2ForImageClassification.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )

        inputs = processor(images=[image], return_tensors="pt").to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)

    def test_model_sdpa(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        model = ShieldGemma2ForImageClassification.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            attn_implementation="sdpa",
        )

        inputs = processor(images=[image], return_tensors="pt").to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)
