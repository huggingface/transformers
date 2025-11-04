# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ModernVBERT model."""

import unittest

import requests
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    ModernVBertConfig,
    ModernVBertForMaskedLM,
    ModernVBertModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class ModernVBertModelTester:
    def __init__(
        self,
        parent,
        seq_length=7,
        text_config={
            "dtype": "float32",
            "hidden_size": 768,
            "intermediate_size": 1152,
            "mlp_bias": False,
            "model_type": "modernvbert_text",
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "text_model_name": "jhu-clsp/ettin-encoder-150m",
            "vocab_size": 50368
        },
        is_training=True,
        vision_config={
            "dtype": "float32",
            "embed_dim": 768,
            "image_size": 512,
            "intermediate_size": 3072,
            "model_type": "modernvbert_vision",
            "num_hidden_layers": 12,
            "patch_size": 16,
            "vision_model_name": "google/siglip2-base-patch16-512"
        },
    ):
        self.parent = parent
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 30

    def get_config(self):
        return ModernVBertConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                1,
                3,
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class ModernVBertModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ModernVBertForMaskedLM`.
    """

    all_model_classes = (
        (
            ModernVBertModel,
            ModernVBertForMaskedLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": ModernVBertModel, "fill-mask": ModernVBertForMaskedLM} if is_torch_available() else {}
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = ModernVBertModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=ModernVBertConfig, has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()


@require_torch
class ModernVBertForMaskedLMIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "ModernVBERT/modernvbert"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
        self.text = "This [MASK] is on the wall."

    @slow
    def test_masked_lm_inference(self):
        model = ModernVBertForMaskedLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        ).to(torch_device)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.text},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[self.image], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        masked_index = inputs["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
        predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
        
        self.assertEqual(predicted_token_id.item(), 13497)

        top_5 = outputs.logits[0, masked_index].topk(5)
        EXPECTED_TOP_5_INDICES = torch.tensor([13497, 5406, 2460, 7512, 3665], device=torch_device)
        
        self.assertTrue(torch.allclose(top_5.indices, EXPECTED_TOP_5_INDICES))
        
        # Loosen the tolerance for the values
        EXPECTED_TOP_5_VALUES = torch.tensor([21.2242, 21.1116, 19.1809, 18.4607, 17.9964], device=torch_device)
        self.assertTrue(torch.allclose(top_5.values, EXPECTED_TOP_5_VALUES, atol=1e-4, rtol=1e-4))
