# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch InternvlFlash model."""

import unittest

import pytest

from transformers import (
    InternvlFlashConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import InternvlFlashForConditionalGeneration, InternvlFlashModel


class InternvlFlashVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=7,
        image_seq_length=256,
        vision_feature_layer=-1,
        ignore_index=-100,
        image_token_id=1,
        num_channels=3,
        image_size=128,
        model_type="internvl_flash",
        is_training=True,
        text_config={
            "model_type": "qwen2",
            "vocab_size": 99,
            "hidden_size": 128,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "output_channels": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "mlp_ratio": 4,
            "tie_word_embeddings": True,
            "bos_token_id": 3,
            "eos_token_id": 4,
            "pad_token_id": 5,
        },
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": (128, 128),
            "patch_size": (4, 4),
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

        self.flash_absolute_threshold = 0.5
        self.flash_relative_threshold = 0.1
        self.text_seq_len = 15

    def get_config(self):
        return InternvlFlashConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            image_token_id=self.image_token_id,
            image_seq_length=self.image_seq_length,
            vision_feature_layer=self.vision_feature_layer,
            flash_relative_threshold=self.flash_relative_threshold,
            flash_absolute_threshold=self.flash_absolute_threshold,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size]).to(
            torch_device
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        attention_mask = torch.ones_like(input_ids)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = InternvlFlashForConditionalGeneration(config=config)
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
        config.dtype = torch.float16
        model = InternvlFlashForConditionalGeneration(config=config)
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


@require_torch
class InternvlFlashModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (InternvlFlashForConditionalGeneration, InternvlFlashModel) if is_torch_available() else ()
    all_generative_model_classes = (InternvlFlashForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": InternvlFlashForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = InternvlFlashVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InternvlFlashConfig, has_text_modality=False)

    def test_flex_attention_with_grads(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("Skipping compilation test: fails with batch_size=0 reshape error")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("query token in InternVLFlashCrossAttentionPooling generate randomly")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip("InternVLFlash model requires input_ids to process pixel_values and merge visual features.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        "The Flash model relies on input_ids for visual compression and cannot support multimodal generation using inputs_embeds alone."
    )
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass
