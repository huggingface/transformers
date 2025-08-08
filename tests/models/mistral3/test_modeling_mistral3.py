# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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

import accelerate

from transformers import (
    AutoProcessor,
    Mistral3Config,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Mistral3ForConditionalGeneration,
        Mistral3Model,
    )


class Mistral3VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        image_seq_length=4,
        vision_feature_layer=-1,
        ignore_index=-100,
        image_token_index=1,
        num_channels=3,
        image_size=30,
        model_type="mistral3",
        is_training=True,
        text_config={
            "model_type": "mistral",
            "vocab_size": 99,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000000.0,
            "sliding_window": None,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 4,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "image_size": 30,
            "patch_size": 6,
            "num_channels": 3,
            "hidden_act": "gelu",
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_index = image_token_index
        self.model_type = model_type
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.vision_feature_layer = vision_feature_layer
        self.is_training = is_training
        self.image_seq_length = image_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length + self.image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return Mistral3Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            image_token_index=self.image_token_index,
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
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=torch.long, device=torch_device
        )

        # input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_torch
class Mistral3ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Mistral3Model,
            Mistral3ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Mistral3ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": Mistral3ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = Mistral3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Mistral3Config, has_text_modality=False)

    def test_config(self):
        # overwritten from `tests/test_configuration_common.py::ConfigTester` after #36077
        # TODO: avoid overwritten once there is a better fix for #36077
        def check_config_can_be_init_without_params():
            config = self.config_tester.config_class()
            self.config_tester.parent.assertIsNotNone(config)

        self.config_tester.check_config_can_be_init_without_params = check_config_can_be_init_without_params
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

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flex_attention_with_grads(self):
        pass


@slow
@require_torch_accelerator
class Mistral3IntegrationTest(unittest.TestCase):
    @require_read_token
    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        self.model = Mistral3ForConditionalGeneration.from_pretrained(self.model_checkpoint, dtype=torch.bfloat16)
        accelerate.cpu_offload(self.model, execution_device=torch_device)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_read_token
    def test_mistral3_integration_generate_text_only(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.chat_template = processor.chat_template.replace('strftime_now("%Y-%m-%d")', '"2025-06-20"')

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
        ).to(torch_device, dtype=torch.bfloat16)

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        expected_outputs = Expectations(
            {
                ("xpu", 3): "Sure, here is a haiku for you:\n\nWhispers of the breeze,\nCherry blossoms softly fall,\nSpring's gentle embrace.",
                ("cuda", 8): "Sure, here is a haiku for you:\n\nWhispers of the breeze,\nCherry blossoms softly fall,\nSpring's gentle embrace.",
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(decoded_output, expected_output)

    @require_read_token
    def test_mistral3_integration_generate(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.chat_template = processor.chat_template.replace('strftime_now("%Y-%m-%d")', '"2025-06-20"')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device, dtype=torch.bfloat16)
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        expected_outputs = Expectations(
            {
                ("xpu", 3): "The image features two cats resting on a pink blanket. The cat on the left is a kitten",
                ("cuda", 8): 'The image features two cats lying on a pink surface, which appears to be a couch or a bed',
                ("rocm", (9, 4)): "The image features two cats lying on a pink surface, which appears to be a couch or a bed",
                ("rocm", (9, 5)): "The image features two tabby cats lying on a pink surface, which appears to be a cushion or"
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(decoded_output, expected_output)

    @require_read_token
    @require_deterministic_for_xpu
    def test_mistral3_integration_batched_generate(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.chat_template = processor.chat_template.replace('strftime_now("%Y-%m-%d")', '"2025-06-20"')
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://huggingface.co/ydshieh/kosmos-2.5/resolve/main/view.jpg"},
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
        ).to(torch_device, dtype=torch.bfloat16)

        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=25)

        gen_tokens = output[:, inputs["input_ids"].shape[1] :]

        # Check first output
        decoded_output = processor.decode(gen_tokens[0], skip_special_tokens=True)

        expected_outputs = Expectations(
            {
                ("xpu", 3): "Calm lake's mirror gleams,\nWhispering pines stand in silence,\nPath to peace begins.",
                ("cuda", 8): "Wooden path to calm,\nReflections whisper secrets,\nNature's peace unfolds.",
                ("rocm", (9, 5)): "Calm waters reflect\nWooden path to distant shore\nSilence in the scene"
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

        # Check second output
        decoded_output = processor.decode(gen_tokens[1], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("xpu", 3): "The image depicts a vibrant urban scene in what appears to be Chinatown. The focal point is a traditional Chinese archway",
                ("cuda", 8): 'The image depicts a street scene in what appears to be a Chinatown district. The focal point is a traditional Chinese arch',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

    @require_read_token
    @require_deterministic_for_xpu
    def test_mistral3_integration_batched_generate_multi_image(self):
        processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        processor.chat_template = processor.chat_template.replace('strftime_now("%Y-%m-%d")', '"2025-06-20"')

        # Prepare inputs
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://huggingface.co/ydshieh/kosmos-2.5/resolve/main/view.jpg"},
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
                            "url": "https://huggingface.co/ydshieh/kosmos-2.5/resolve/main/Statue-of-Liberty-Island-New-York-Bay.jpg",
                        },
                        {
                            "type": "image",
                            "url": "https://huggingface.co/ydshieh/kosmos-2.5/resolve/main/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
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
        ).to(torch_device, dtype=torch.bfloat16)

        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=25)
        gen_tokens = output[:, inputs["input_ids"].shape[1] :]

        # Check first output
        decoded_output = processor.decode(gen_tokens[0], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("cuda", 8): 'Calm waters reflect\nWooden path to distant shore\nSilence in the scene',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

        # Check second output
        decoded_output = processor.decode(gen_tokens[1], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("xpu", 3): "Certainly! The images depict two iconic landmarks:\n\n1. The first image shows the Statue of Liberty in New York City.",
                ("cuda", 8): 'Certainly! The images depict two famous landmarks in the United States:\n\n1. The first image shows the Statue of Liberty,',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )
