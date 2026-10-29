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
"""Testing suite for the PyTorch Idefics3 model."""

import unittest
from io import BytesIO

import pytest
import requests

from transformers import (
    AutoProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        BitsAndBytesConfig,
        Idefics3Config,
        Idefics3ForConditionalGeneration,
        Idefics3Model,
    )

if is_vision_available():
    from PIL import Image


class Idefics3VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        is_training=True,
        batch_size=2,
        scale_factor=2,
        num_images=2,
        seq_length=7,
        vision_config={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 56,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 2,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "image_token_id": 57,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "sliding_window": 32,
            "attention_dropout": 0.0,
        },
        use_cache=False,
        tie_word_embeddings=False,
        # should be smaller than vocab size - 16 to survive clamping
        image_token_id=57,
    ):
        self.parent = parent
        self.pad_token_id = text_config["pad_token_id"]
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_images = num_images
        self.scale_factor = scale_factor
        self.image_seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (self.scale_factor**2))
            * self.num_images
        )
        self.seq_length = seq_length + self.image_seq_length
        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        # Hack - add properties here so use common tests
        self.vocab_size = text_config["vocab_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]

        self.vision_config = vision_config
        self.text_config = text_config

    def get_config(self):
        return Idefics3Config(
            use_cache=self.use_cache,
            image_token_id=self.image_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            vision_config=self.vision_config,
            text_config=self.text_config,
            vocab_size=self.vocab_size,
            scale_factor=self.scale_factor,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_images,
                3,  # Idefics3ImageProcessorPil always generates RGB pixel values
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Idefics3ModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics3`.
    """

    all_model_classes = (Idefics3Model,) if is_torch_available() else ()
    # Idefics3 merges batch_size and num_frames in the first output dimension
    skip_test_image_features_output_shape = True

    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = Idefics3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Idefics3Config, has_text_modality=False, common_properties=["image_token_id"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="inputs_embeds cannot be passed in without input_ids")
    def test_inputs_embeds():
        pass

    @unittest.skip(reason="inputs_embeds cannot be passed in without input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    @unittest.skip(reason="Compile not yet supported in idefics3 models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch
class Idefics3ForConditionalGenerationModelTest(GenerationTesterMixin, ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics3ForConditionalGeneration`.
    """

    all_model_classes = (Idefics3ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": Idefics3ForConditionalGeneration} if is_torch_available() else ()
    skip_test_image_features_output_shape = (
        True  # Idefics3 merges batch_size and num_frames in the first output dimension
    )

    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = Idefics3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics3Config, has_text_modality=False)

    @unittest.skip(reason="inputs_embeds cannot be passed in without input_ids")
    def test_inputs_embeds():
        pass

    @unittest.skip(reason="Model does not support padding right")
    def test_flash_attn_2_inference_padding_right(self):
        pass

    @pytest.mark.generate
    @slow
    @unittest.skip(
        reason="Idefics3 doesn't support SDPA for all backbones, vision backbones has only eager/FA2 attention"
    )
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip(reason="Compile not yet supported in Idefics3 models end-to-end")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch
class Idefics3ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
        self.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/statue_of_liberty.jpg"
                ).content
            )
        )
        self.image2 = Image.open(
            BytesIO(
                requests.get(
                    "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/skyline_chicago.jpg"
                ).content
            )
        )
        self.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/dreamstime_golden_gate_flowers.jpg"
                ).content
            )
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_integration_test(self):
        model = Idefics3ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/Idefics3-8B-Llama3",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image1
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        expected_generated_text = Expectations(
            {
                ("cuda", None): " the Statue of Liberty, which is a colossal ne",
                ("rocm", (9, 4)): " the Statue of Liberty, which is a symbol of",
            }
        ).get_expectation()
        self.assertEqual(generated_texts[0], expected_generated_text)

    @slow
    @require_bitsandbytes
    def test_integration_test_4bit(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Idefics3ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/Idefics3-8B-Llama3",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create pixel inputs
        text = ["<image>In this image, we see", "bla, bla <image><image>"]
        images = [[self.image1], [self.image2, self.image3]]
        inputs = self.processor(text=text, images=images, padding=True, return_tensors="pt")
        inputs.to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        expected_generated_texts = [
            " the Statue of Liberty, trees, buildings, water",
            " The golden gate bridge is in the image.",
        ]
        self.assertEqual(generated_texts, expected_generated_texts)
