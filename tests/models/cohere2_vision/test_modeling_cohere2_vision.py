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
"""Testing suite for the PyTorch GotOcr2 model."""

import unittest

from transformers import (
    AutoProcessor,
    Cohere2VisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    get_device_properties,
    require_deterministic_for_xpu,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Cohere2VisionForConditionalGeneration,
        Cohere2VisionModel,
    )


class Cohere2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        downsample_factor=2,
        alignment_intermediate_size=32,
        ignore_index=-100,
        image_token_id=2,
        num_channels=3,
        image_size=64,
        is_training=True,
        text_config={
            "model_type": "cohere2",
            "vocab_size": 99,
            "hidden_size": 128,
            "intermediate_size": 37,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "output_channels": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "tie_word_embeddings": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 64,
            "patch_size": 8,
            "vision_use_head": False,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_id = image_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.alignment_intermediate_size = alignment_intermediate_size
        self.is_training = is_training
        self.num_channels = num_channels
        self.image_size = image_size
        self.image_seq_length = 16
        self.seq_length = seq_length + self.image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return Cohere2VisionConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            downsample_factor=self.downsample_factor,
            alignment_intermediate_size=self.alignment_intermediate_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        image_num_patches = torch.tensor([1] * self.batch_size).to(torch_device)

        return config, pixel_values, image_num_patches

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, image_num_patches = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_num_patches": image_num_patches,
        }
        return config, inputs_dict


@require_torch
class Cohere2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Cohere2VisionModel,
            Cohere2VisionForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Cohere2VisionForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": Cohere2VisionForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Cohere2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Cohere2VisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Siglip backbone uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass


@require_read_token
@require_torch
class Cohere2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_checkpoint = "CohereLabs/command-a-vision-07-2025"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def get_model(self, dummy=True):
        device_type, major, _ = get_device_properties()
        dtype = torch.float16

        # too large to fit into A10
        config = Cohere2VisionConfig.from_pretrained(self.model_checkpoint)
        if dummy:
            config.text_config.num_hidden_layers = 4
            config.text_config.layer_types = config.text_config.layer_types[:4]

        model = Cohere2VisionForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            config=config,
            dtype=dtype,
            device_map="auto",
        )
        return model

    @slow
    @require_torch_accelerator
    def test_model_integration_forward(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        model = self.get_model(dummy=False)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": "Please describe the image explicitly."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, dtype=torch.float16)
        # Forward
        with torch.inference_mode():
            output = model(**inputs)

        actual_logits = output.logits[0, -1, :5].cpu()

        EXPECTED_LOGITS = Expectations(
            {
                ("xpu", 3): [0.4109, 0.1532, 0.8018, 2.1328, 0.5483],
                # 4-bit
                ("cuda", 7): [0.1097, 0.3481, 3.8340, 9.7969, 2.0488],
                ("cuda", 8): [2.4277, 1.6875, 1.8789, 2.1875, 1.9375],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), dtype=torch.float16)

        self.assertTrue(
            torch.allclose(actual_logits, expected_logits, atol=0.1),
            f"Actual logits: {actual_logits}"
            f"\nExpected logits: {expected_logits}"
            f"\nDifference: {torch.abs(actual_logits - expected_logits)}",
        )

    @slow
    @require_torch_accelerator
    @require_deterministic_for_xpu
    def test_model_integration_generate_text_only(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        model = self.get_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write a haiku"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, dtype=torch.float16)
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        expected_outputs = Expectations(
            {
                ("cuda", 8): "<|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|>",
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(decoded_output, expected_output)

    @slow
    @require_torch_accelerator
    @require_deterministic_for_xpu
    def test_model_integration_generate_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        model = self.get_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": "Please describe the image explicitly."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, dtype=torch.float16)
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        expected_outputs = Expectations(
            {
                ("cuda", 8): '<|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|>',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(decoded_output, expected_output)

    @slow
    @require_torch_accelerator
    def test_model_integration_batched_generate(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        model = self.get_model(dummy=False)
        # Prepare inputs
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                        {"type": "text", "text": "Describe this image"},
                    ],
                },
            ],
        ]
        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=5)

        # Check first output
        decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("cuda", 8): 'Dock stretches to calm',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

        # Check second output
        decoded_output = processor.decode(output[1, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_outputs = Expectations(
            {
                ("cuda", 8): 'The image depicts a',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

    @slow
    @require_torch_accelerator
    @require_deterministic_for_xpu
    def test_model_integration_batched_generate_multi_image(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        model = self.get_model()
        # Prepare inputs
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                },
            ],
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
                        {
                            "type": "text",
                            "text": "These images depict two different landmarks. Can you identify them?",
                        },
                    ],
                },
            ],
        ]
        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.float16)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=10)

        # Check first output
        decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        # Batching seems to alter the output slightly, but it is also the case in the original implementation. This seems to be expected: https://github.com/huggingface/transformers/issues/23017#issuecomment-1649630232
        expected_outputs = Expectations(
            {
                ("cuda", 8): '<|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|>',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

        # Check second output
        decoded_output = processor.decode(output[1, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("cuda", 8): '<|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|><|CHATBOT_TOKEN|>',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )
