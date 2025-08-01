# coding=utf-8
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

import re
import tempfile
import unittest

from transformers import (
    AutoProcessor,
    DeepseekVLConfig,
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch


class DeepseekVLModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=25,
        num_channels=3,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        text_config={
            "num_hidden_layers": 2,
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "pad_token_id": 1,
        },
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "intermediate_size": 37,
            "image_size": 32,
            "patch_size": 8,
            "hidden_act": "gelu",
            "vision_use_head": False,
            "num_attention_heads": 4,
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache

        self.text_config = text_config
        self.vision_config = vision_config
        self.vision_config["num_channels"] = self.num_channels

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.image_size = vision_config["image_size"]
        self.num_image_tokens = 16
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_id = 0

    def get_config(self):
        return DeepseekVLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        # create text and vision inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        # fill image_tokens
        input_ids[input_ids == self.num_image_tokens] = config.text_config.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id

        return config, input_ids, attention_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DeepseekVLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (DeepseekVLModel, DeepseekVLForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekVLModel,
            "image-text-to-text": DeepseekVLForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DeepseekVLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekVLConfig, has_text_modality=False)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for VLMs.
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    # Copied from tests.models.siglip.test_modeling_siglip.SiglipVisionModelTest.test_initialization
    def test_initialization(self):
        pass

    @require_torch_sdpa
    # Copied from tests.models.janus.test_modeling_janus.JanusVisionText2TextModelTest.test_sdpa_can_dispatch_composite_models
    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            # SigLip has one shared cls attr for all models, so we assign both submodels heer
            vision_attn = language_attn = "sdpa" if model._supports_sdpa else "eager"

            if hasattr(model_sdpa, "vision_model") and hasattr(model_sdpa, "language_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.language_model.config._attn_implementation == language_attn)
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if any(re.finditer(r"Attention(?!Pool)", class_name)):
                    self.assertTrue(submodule.config._attn_implementation == "eager")

            for name, submodule in model_sdpa.named_modules():
                class_name = submodule.__class__.__name__
                if any(re.finditer(r"Attention(?!Pool)", class_name)):
                    self.assertTrue(submodule.config._attn_implementation == "sdpa")


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
            "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: Describe this image.\n\nAssistant:In the image, a majestic snow leopard is captured in a moment of tranquility. The snow leopard",  # fmt: skip
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
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
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
