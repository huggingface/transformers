# coding=utf-8
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
"""Testing suite for the PyTorch InternVL model."""

import unittest

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    InternS1Config,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import InternS1ForConditionalGeneration, InternS1Model


class InternS1VisionText2TextMoEModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        image_seq_length=64,
        vision_feature_layer=-1,
        ignore_index=-100,
        image_token_id=1,
        num_channels=3,
        image_size=64,
        model_type="intern_s1",
        is_training=True,
        text_config={
            "model_type": "qwen3_moe",
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "output_channels": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "mlp_ratio": 2,
            "tie_word_embeddings": False,
            "bos_token_id": 3,
            "eos_token_id": 4,
            "pad_token_id": 5,
            "num_experts": 8,
            "output_router_logits": True,
        },
        vision_config={
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 24,
            "image_size": 64,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_act": "quick_gelu",
            "use_absolute_position_embeddings": True,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_id = image_token_id
        self.model_type = model_type
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.vision_feature_layer = vision_feature_layer
        self.is_training = is_training
        self.image_seq_length = image_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length + image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return InternS1Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            image_token_id=self.image_token_id,
            image_seq_length=self.image_seq_length,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = InternS1ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_model_fp16_autocast_forward(self, config, input_ids, pixel_values, attention_mask):
        config.torch_dtype = torch.float16
        model = InternS1ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type=torch_device, dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


class InternS1VisionText2TextDenseModelTester(InternS1VisionText2TextMoEModelTester):
    def __init__(
        self,
        *args,
        text_config={
            "model_type": "qwen3",
            "vocab_size": 99,
            "hidden_size": 16,
            "head_dim": 4,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "output_channels": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "mlp_ratio": 2,
            "tie_word_embeddings": False,
            "bos_token_id": 3,
            "eos_token_id": 4,
            "pad_token_id": 5,
        },
        **kwargs,
    ):
        super().__init__(*args, text_config=text_config, **kwargs)


@require_torch
class InternS1MoEModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (InternS1ForConditionalGeneration, InternS1Model) if is_torch_available() else ()
    all_generative_model_classes = (InternS1ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": InternS1ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = InternS1VisionText2TextMoEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InternS1Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_torch
class InternS1DenseModelTest(InternS1MoEModelTest):
    def setUp(self):
        self.model_tester = InternS1VisionText2TextDenseModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InternS1Config, has_text_modality=False)


class InternS1IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.small_model_checkpoint = "internlm/Intern-S1-mini-hf"
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        processor = AutoProcessor.from_pretrained(self.small_model_checkpoint)
        model = AutoModelForImageTextToText.from_pretrained(
            self.small_model_checkpoint, device_map=torch_device, torch_dtype=torch.float16
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": "Please describe the image shortly."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, dtype=torch.float16)
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        expected_output = (
            "\nOkay, let's see. The user wants a short description of the image. The image shows two "
            "cats lying on a pink couch. One cat is lying on its back with its belly exposed, "
            "and the other is lying on its side"
        )

        self.assertEqual(decoded_output, expected_output)
