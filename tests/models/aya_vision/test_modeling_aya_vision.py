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

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AyaVisionConfig,
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
        AyaVisionForConditionalGeneration,
        AyaVisionModel,
    )


class AyaVisionVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        vision_feature_layer=-1,
        downsample_factor=2,
        ignore_index=-100,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        image_token_index=2,
        num_channels=3,
        image_size=64,
        model_type="aya_vision",
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
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.model_type = model_type
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.vision_feature_layer = vision_feature_layer
        self.downsample_factor = downsample_factor
        self.is_training = is_training
        self.num_channels = num_channels
        self.image_size = image_size
        self.image_seq_length = (image_size // (vision_config["patch_size"] * downsample_factor)) ** 2
        self.seq_length = seq_length + self.image_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return AyaVisionConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            image_token_index=self.image_token_index,
            vision_feature_layer=self.vision_feature_layer,
            downsample_factor=self.downsample_factor,
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
        # input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class AyaVisionModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            AyaVisionModel,
            AyaVisionForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (AyaVisionForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": AyaVisionForConditionalGeneration,
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
        self.model_tester = AyaVisionVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AyaVisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Cohere2's forcefully disables sdpa due to softcapping")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("Cohere2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("Cohere2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Cohere2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Cohere2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    # todo: yoni - fix or improve the test
    @unittest.skip("Difference is slightly higher than the threshold")
    def test_batching_equivalence(self):
        pass


@require_read_token
@require_torch
class AyaVisionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_checkpoint = "CohereForAI/aya-vision-8b"
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls):
        # Use 4-bit on T4
        device_type, major, _ = get_device_properties()
        load_in_4bit = (device_type == "cuda") and (major < 8)
        dtype = None if load_in_4bit else torch.float16

        if cls.model is None:
            cls.model = AyaVisionForConditionalGeneration.from_pretrained(
                cls.model_checkpoint,
                device_map=torch_device,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
        return cls.model

    @slow
    @require_torch_accelerator
    def test_small_model_integration_forward(self):
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
        # Forward
        with torch.inference_mode():
            output = model(**inputs)

        actual_logits = output.logits[0, -1, :5].cpu()

        EXPECTED_LOGITS = Expectations(
            {
                ("xpu", 3): [0.4109, 0.1532, 0.8018, 2.1328, 0.5483],
                # 4-bit
                ("cuda", 7): [0.1097, 0.3481, 3.8340, 9.7969, 2.0488],
                ("cuda", 8): [1.6396, 0.6094, 3.1992, 8.5234, 2.1875],
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
    def test_small_model_integration_generate_text_only(self):
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
            generate_ids = model.generate(**inputs, max_new_tokens=25, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        expected_outputs = Expectations(
            {
                ("xpu", 3): "Whispers on the breeze,\nLeaves dance under moonlit skies,\nNature's quiet song.",
                # 4-bit
                ("cuda", 7): "Sure, here's a haiku for you:\n\nMorning dew sparkles,\nPetals unfold in sunlight,\n",
                ("cuda", 8): "Whispers on the breeze,\nLeaves dance under moonlit skies,\nNature's quiet song.",
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(decoded_output, expected_output)

    @slow
    @require_torch_accelerator
    @require_deterministic_for_xpu
    def test_small_model_integration_generate_chat_template(self):
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
            generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            decoded_output = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        expected_outputs = Expectations(
            {
                ("xpu", 3): 'The image depicts a cozy scene of two cats resting on a bright pink blanket. The cats,',
                # 4-bit
                ("cuda", 7): 'The image depicts two cats comfortably resting on a pink blanket spread across a sofa. The cats,',
                ("cuda", 8): 'The image depicts a cozy scene of two cats resting on a bright pink blanket. The cats,',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(decoded_output, expected_output)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_batched_generate(self):
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
                        {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                        {"type": "text", "text": "Describe this image"},
                    ],
                },
            ],
        ]
        inputs = processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        # Check first output
        decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        expected_outputs = Expectations(
            {
                ("xpu", 3): "Wooden path to water,\nMountains echo in stillness,\nPeaceful forest lake.",
                # 4-bit
                ("cuda", 7): "Wooden bridge stretches\nMirrored lake below, mountains rise\nPeaceful, serene",
                ("cuda", 8): 'Wooden path to water,\nMountains echo in stillness,\nPeaceful forest scene.',
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
                ("xpu", 3): 'This image captures a vibrant street scene in a bustling urban area, likely in an Asian city. The focal point is a',
                # 4-bit
                ("cuda", 7): 'This vibrant image captures a bustling street scene in a multicultural urban area, featuring a traditional Chinese gate adorned with intricate red and',
                ("cuda", 8): 'This image captures a vibrant street scene in a bustling urban area, likely in an Asian city. The focal point is a',
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
    def test_small_model_integration_batched_generate_multi_image(self):
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
        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        # Check first output
        decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        # Batching seems to alter the output slightly, but it is also the case in the original implementation. This seems to be expected: https://github.com/huggingface/transformers/issues/23017#issuecomment-1649630232
        expected_outputs = Expectations(
            {
                ("xpu", 3): "Wooden path to water,\nMountains echo in stillness,\nPeaceful forest lake.",
                ("cuda", 7): 'Wooden bridge stretches\nMirrored lake below, mountains rise\nPeaceful, serene',
                ("cuda", 8): 'Wooden path to water,\nMountains echo in stillness,\nPeaceful forest scene.',
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
                ("xpu", 3): "The first image showcases the Statue of Liberty, a colossal neoclassical sculpture on Liberty Island in New York Harbor. Standing at ",
                ("cuda", 7): 'The first image showcases the Statue of Liberty, a monumental sculpture located on Liberty Island in New York Harbor. Standing atop a',
                ("cuda", 8): 'The first image showcases the Statue of Liberty, a colossal neoclassical sculpture on Liberty Island in New York Harbor. Standing at ',
            }
        )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )
