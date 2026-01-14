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
"""Testing suite for the LFM2-VL model."""

import math
import unittest
from io import BytesIO

import pytest
import requests

from transformers import AutoProcessor, is_torch_available
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_vision_available

from ...causal_lm_tester import CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration, Lfm2VlModel
    from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache


class Lfm2VlModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Lfm2VlConfig
        base_model_class = Lfm2VlModel
        causal_lm_class = Lfm2VlForConditionalGeneration

    def __init__(
        self,
        parent,
        is_training=True,
        batch_size=2,
        scale_factor=2,
        num_images=2,
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "num_patches": 16,
            "patch_size": 4,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 100,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": True,
            "rope_theta": 1000000.0,
            "conv_bias": False,
            "conv_L_cache": 3,
            "block_multiple_of": 2,
            "full_attn_idxs": [0],
        },
        image_token_id=4,
        downsample_factor=4,
        projector_hidden_size=32,
    ):
        super().__init__(parent)
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.is_training = is_training
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.num_images = num_images
        self.downsample_factor = downsample_factor
        self.projector_hidden_size = projector_hidden_size
        self.image_seq_length = 4

    def get_config(self):
        return Lfm2VlConfig(
            vision_config=self.vision_config,
            text_config=self.text_config,
            image_token_id=self.image_token_id,
            downsample_factor=self.downsample_factor,
            projector_hidden_size=self.projector_hidden_size,
        )

    def prepare_config_and_inputs(self):
        # Create dummy pixel values: [num_images, num_patches, channels * patch_size^2]
        patch_size = self.vision_config["patch_size"]
        pixel_values = floats_tensor([self.num_images, 64, 3 * patch_size * patch_size])

        # Spatial shapes: one (height_patches, width_patches) per image
        patches = int(math.sqrt(64))
        spatial_shapes = torch.tensor([[patches, patches]] * self.num_images, dtype=torch.long, device=torch_device)

        # Pixel attention mask: mark all patches as valid (no padding)
        pixel_attention_mask = torch.ones((self.num_images, 64), dtype=torch.long, device=torch_device)
        config = self.get_config()
        return config, pixel_values, spatial_shapes, pixel_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, spatial_shapes, pixel_attention_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        # For simplicity just set the last n tokens to the image token
        input_ids[input_ids == self.image_token_id] = self.text_config["pad_token_id"]
        input_ids[:, -self.image_seq_length :] = self.image_token_id

        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spatial_shapes": spatial_shapes,
            "pixel_attention_mask": pixel_attention_mask,
        }
        return config, inputs_dict


@require_torch
class Lfm2VlModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Lfm2VlModel, Lfm2VlForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Lfm2VlModel,
            "text-generation": Lfm2VlForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    model_tester_class = Lfm2VlModelTester
    _is_composite = True

    def setUp(self):
        self.model_tester = Lfm2VlModelTester(self)
        common_properties = ["image_token_id", "projector_hidden_size"]
        self.config_tester = ConfigTester(
            self, config_class=Lfm2VlConfig, has_text_modality=False, common_properties=common_properties
        )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, Lfm2HybridConvCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)
        conv_shape = (batch_size, config.hidden_size, config.conv_L_cache)

        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == "full_attention":
                self.assertEqual(past_key_values.key_cache[i].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[i].shape, attention_shape)
            else:
                self.assertEqual(past_key_values.conv_cache[i].shape, conv_shape)

    def _check_caches_are_equal(self, cache1: Lfm2HybridConvCache, cache2: Lfm2HybridConvCache):
        """Text model uses lfm2, which has non-standard cache"""
        if not isinstance(cache1, Lfm2HybridConvCache) or not isinstance(cache2, Lfm2HybridConvCache):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_cache[idx], cache2.conv_cache[idx])

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        "Lfm2 backbone alternates between attention and conv layers, so attention are only returned for attention layers"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        "Lfm2 backbone has a special cache format which is not compatible with compile as it has static address for conv cache"
    )
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Backbone Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Backbone Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Backbone Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass


@require_torch_accelerator
@slow
class Lfm2VlForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-1.6B")
        self.processor.tokenizer.padding_side = "left"
        self.image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )
        self.image2 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_integration_test(self):
        model = Lfm2VlForConditionalGeneration.from_pretrained(
            "LiquidAI/LFM2-VL-1.6B",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs.to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = "In this image, we see a cat and a dog lying on a pink blanket. They are both sleeping peacefully. They are"
        self.assertEqual(generated_texts[0], expected_generated_text)

    def test_integration_test_high_resolution(self):
        model = Lfm2VlForConditionalGeneration.from_pretrained(
            "LiquidAI/LFM2-VL-1.6B",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image2
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs.to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = (
            "In this image, we see the Statue of Liberty, standing tall on its pedestal. The statue is made of metal,"
        )
        self.assertEqual(generated_texts[0], expected_generated_text)

    def test_integration_test_batched(self):
        model = Lfm2VlForConditionalGeneration.from_pretrained(
            "LiquidAI/LFM2-VL-450M",
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = ["<image>In this image, we see", "<image>In this image, we see a cat"]
        images = [[self.image2], [self.image]]
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = [
            "In this image, we see a panoramic view of the New York City skyline. The iconic Statics and the New York",
            "In this image, we see a cat that is lying on its side on a cat bed.",
        ]
        self.assertListEqual(generated_texts, expected_generated_text)
