# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PPChart2Table model."""

import unittest

from transformers import (
    AutoProcessor,
    PPChart2TableConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        PPChart2TableForConditionalGeneration,
    )


if is_vision_available():
    from transformers.image_utils import load_image


class PPChart2TableVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=64,
        image_token_index=1,
        model_type="pp_chart2table",
        is_training=True,
        text_config={
            "model_type": "qwen2",
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "output_channels": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 4,
        },
        vision_config={
            "num_hidden_layers": 2,
            "output_channels": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "mlp_dim": 64,
            "num_attention_heads": 2,
            "patch_size": 2,
            "image_size": 64,
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
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return PPChart2TableConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            image_token_index=self.image_token_index,
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

        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class PPChart2TableModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPChart2TableForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-text-to-text": PPChart2TableForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = PPChart2TableVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PPChart2TableConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()


@slow
@require_torch
class PPChart2TableIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
        self.model = PPChart2TableForConditionalGeneration.from_pretrained(model_path).to(torch_device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png"
        self.image = load_image(url)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test_pp_chart2table(self):
        inputs = self.processor(self.image, return_tensors="pt").to(torch_device)
        generate_ids = self.model.generate(
            **inputs,
            use_cache=True,
            do_sample=False,
            max_new_tokens=32,
        )
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)\n"
        self.assertEqual(decoded_output, expected_output)

    def test_small_model_integration_test_pp_chart2table_batched(self):
        inputs = self.processor([self.image, self.image], return_tensors="pt").to(torch_device)
        generate_ids = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=6)
        decoded_output = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = ["年份 | 单家", "年份 | 单家"]
        self.assertEqual(decoded_output, expected_output)
