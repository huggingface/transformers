# coding=utf-8
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
"""Testing suite for the PyTorch Idefics2 model."""

import gc
import unittest
from io import BytesIO

import requests

from transformers import (
    AutoProcessor,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    Idefics2Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class Idefics2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        is_training=True,
        batch_size=2,
        num_images=2,
        seq_length=10,
        additional_vocab_size=0,
        vocab_size=100,
        hidden_size=64,
        intermediate_size=56,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=256,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,  # None in the original configuration_mistral, we set it to the unk_token_id
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=32_001,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=32,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        attention_dropout=0.0,
        vision_config={
            "image_size": 12,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        perceiver_config={
            # "hidden_size": 4096,
            "hidden_act": "silu",
            "resampler_n_latents": 2,
            "resampler_depth": 2,
            "resampler_n_heads": 2,
            "num_key_value_heads": 1,
            "resampler_head_dim": 12,
            "qk_layer_norms_perceiver": False,
            "attention_dropout": 0.0,
        },
    ):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_images = num_images
        self.num_channels = 3
        self.seq_length = seq_length
        self.additional_vocab_size = additional_vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.qk_layer_norms = qk_layer_norms
        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_lm_head = freeze_lm_head
        self.freeze_vision_layers = freeze_vision_layers
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.attention_dropout = attention_dropout

        self.vision_config = vision_config
        self.perceiver_config = perceiver_config

    def get_config(self):
        return Idefics2Config(
            vision_config=self.vision_config,
            perceiver_config=self.perceiver_config,
            additional_vocab_size=self.additional_vocab_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            image_token_id=self.image_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            sliding_window=self.sliding_window,
            qk_layer_norms=self.qk_layer_norms,
            freeze_text_layers=self.freeze_text_layers,
            freeze_text_module_exceptions=self.freeze_text_module_exceptions,
            freeze_lm_head=self.freeze_lm_head,
            freeze_vision_layers=self.freeze_vision_layers,
            freeze_vision_module_exceptions=self.freeze_vision_module_exceptions,
            attention_dropout=self.attention_dropout,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_images,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Idefics2ModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics2`.
    """

    all_model_classes = (Idefics2Model,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = Idefics2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics2Config, has_text_modality=False)


@require_torch
class Idefics2ForConditionalGenerationModelTest(GenerationTesterMixin, ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics2ForConditionalGeneration`.
    """

    all_model_classes = (Idefics2ForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Idefics2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics2Config, has_text_modality=False)


@require_torch
class Idefics2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("amyeroberts/idefics2")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_integration_test(self):
        torch_device = "cuda"
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "amyeroberts/idefics2",
            torch_dtype=torch.bfloat16,
        )
        model.to(torch_device)

        # Create pixel inputs
        image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )
        # Create inputs
        prompts = [
            [image1, "In this image, we see"],
            ["bla, bla", image2, image3],
        ]
        inputs = self.processor(prompts, return_tensors="pt", padding=True)
        inputs.to(torch_device)

        generated_ids = model.generate(**inputs, bad_words_ids=self.processor.bad_words_ids, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Batch affects generated text. Single batch output: ['In this image, we see the Statue of Liberty in New York City.']
        expected_generated_text = "In this image, we see the beautiful city skyline. The city is a"
        self.assertEqual(generated_texts[0], expected_generated_text)

    @slow
    @require_bitsandbytes
    def test_integration_test_4bit(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2", load_in_4bit=True)

        # Create pixel inputs
        image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        prompts = [image1, "In this image, we see"]
        inputs = self.processor(prompts, return_tensors="pt")

        generated_ids = model.generate(**inputs, bad_words_ids=self.processor.bad_words_ids, max_new_tokens=10)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = "In this image, we see the beautiful cityscape of New York City. The"
        self.assertEqual(generated_texts[0], expected_generated_text)
