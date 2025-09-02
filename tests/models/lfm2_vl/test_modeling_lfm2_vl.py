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
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_vision_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration, Lfm2VlModel


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
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_channels": 3,
            "num_patches": 256,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
        },
        text_config={
            "vocab_size": 65536,
            "hidden_size": 2560,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 128_000,
            "initializer_range": 0.02,
            "norm_eps": 0.00001,
            "use_cache": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": True,
            "rope_theta": 1000000.0,
            "conv_bias": False,
            "conv_L_cache": 3,
            "block_multiple_of": 256,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": True,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
        },
        image_token_id=396,
        downsample_factor=2,
        max_image_tokens=256,
        encoder_patch_size=16,
        use_image_special_tokens=True,
        do_image_splitting=True,
        min_tiles=2,
        max_tiles=10,
        tile_size=512,
        max_pixels_tolerance=2.0,
        use_thumbnail=True,
        seq_length=1024,
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
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.use_image_special_tokens = use_image_special_tokens
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance
        self.use_thumbnail = use_thumbnail
        self.seq_length = seq_length

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0
        self.max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

    def get_config(self):
        return Lfm2VlConfig(
            vision_config=self.vision_config,
            text_config=self.text_config,
            image_token_id=self.image_token_id,
            downsample_factor=self.downsample_factor,
            max_image_tokens=self.max_image_tokens,
            encoder_patch_size=self.encoder_patch_size,
            use_image_special_tokens=self.use_image_special_tokens,
            do_image_splitting=self.do_image_splitting,
            min_tiles=self.min_tiles,
            max_tiles=self.max_tiles,
            tile_size=self.tile_size,
            max_pixels_tolerance=self.max_pixels_tolerance,
            use_thumbnail=self.use_thumbnail,
        )

    def prepare_config_and_inputs(self):
        # Create dummy pixel values: [num_images, num_patches, channels * patch_size^2]
        pixel_values = floats_tensor(
            [self.num_images, self.max_num_patches, 3 * self.encoder_patch_size * self.encoder_patch_size]
        )
        # Compute square grid size in patches
        patches = int(math.sqrt(self.max_num_patches))
        # Spatial shapes: one (height_patches, width_patches) per image
        spatial_shapes = torch.tensor([[patches, patches]] * self.num_images, dtype=torch.long)
        # Pixel attention mask: mark all patches as valid (no padding)
        pixel_attention_mask = torch.ones((self.num_images, self.max_num_patches), dtype=torch.long)
        config = self.get_config()
        return config, pixel_values, spatial_shapes, pixel_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, spatial_shapes, pixel_attention_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.max_image_tokens
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        pixel_values = pixel_values.to(torch_device)
        spatial_shapes = spatial_shapes.to(torch_device)
        pixel_attention_mask = pixel_attention_mask.to(torch_device)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spatial_shapes": spatial_shapes,
            "pixel_attention_mask": pixel_attention_mask,
        }
        return config, inputs_dict


@require_torch
class Lfm2VlModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Lfm2VlModel, Lfm2VlForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Lfm2VlModel,
            "text-generation": Lfm2VlForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = Lfm2VlModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2VlForConditionalGeneration if is_torch_available() else None

    @unittest.skip(
        "Lfm2 alternates between attention and conv layers, so attention are only returned for attention layers"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip("Lfm2 has a special cache format as it alternates between attention and conv layers")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("Lfm2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Lfm2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Lfm2 has a special cache format which is not compatible with contrastive search")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "Lfm2 has a special cache format which is not compatible with compile as it has static address for conv cache"
    )
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch_accelerator
@require_read_token
@slow
class Lfm2VlForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-1.6B")
        self.image = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_integration_test(self):
        model = Lfm2VlForConditionalGeneration.from_pretrained(
            "LiquidAI/LFM2-VL-1.6B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Create inputs
        text = "<image>In this image, we see"
        images = self.image
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs.to(device=torch_device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=9)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        expected_generated_text = "In this image, we see the Statue of Liberty, standing tall on"
        self.assertEqual(generated_texts[0], expected_generated_text)
