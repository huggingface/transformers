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
"""Testing suite for the PyTorch Molmo2 model."""

import copy
import unittest

from transformers import (
    Molmo2Config,
    Molmo2ForConditionalGeneration,
    Molmo2Model,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


class Molmo2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=378,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 128,
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "model_type": "molmo2_text",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "use_qk_norm": False,
            "layer_norm_eps": 1e-6,
        },
        vit_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "image_default_input_size": (378, 378),
            "image_patch_size": 14,
            "image_num_pos": 729,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        adapter_config={
            "vit_layers": (-3, -9),
            "pooling_attention_mask": False,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "intermediate_size": 37,
            "text_hidden_size": 32,
            "hidden_act": "silu",
        },
        image_start_token_id=3,
        image_end_token_id=4,
        image_patch_id=5,
        image_col_id=6,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.is_training = is_training

        self.vit_config = vit_config
        self.adapter_config = adapter_config
        self.text_config = text_config

        self.vocab_size = text_config["vocab_size"]
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.head_dim = text_config["head_dim"]
        self.hidden_size = text_config["hidden_size"]
        self.intermediate_size = text_config["intermediate_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_key_value_heads = text_config["num_key_value_heads"]
        self.rope_theta = text_config["rope_theta"]
        self.hidden_act = text_config["hidden_act"]
        self.max_position_embeddings = text_config["max_position_embeddings"]
        self.model_type = text_config["model_type"]

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_patch_id = image_patch_id
        self.image_col_id = image_col_id
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        from transformers.models.molmo2.configuration_molmo2 import (
            Molmo2AdapterConfig,
            Molmo2TextConfig,
            Molmo2VitConfig,
        )

        return Molmo2Config(
            text_config=Molmo2TextConfig(**self.text_config),
            vit_config=Molmo2VitConfig(**self.vit_config),
            adapter_config=Molmo2AdapterConfig(**self.adapter_config),
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            image_patch_id=self.image_patch_id,
            image_col_id=self.image_col_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vit_config.image_patch_size
        num_patches = (self.image_size // patch_size) ** 2
        pixel_values = floats_tensor(
            [
                self.batch_size,
                1,  # num_crops
                num_patches,
                patch_size * patch_size * self.num_channels,
            ]
        )
        image_token_pooling = torch.randint(
            -1, num_patches, (self.batch_size, self.num_image_tokens, 4), device=torch_device
        )
        image_grids = torch.tensor([[27, 27, 27, 27]] * self.batch_size, device=torch_device)

        return config, pixel_values, image_token_pooling, image_grids

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, image_token_pooling, image_grids = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_patch_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_patch_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_token_pooling": image_token_pooling,
            "image_grids": image_grids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Molmo2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Molmo2ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Molmo2Model,
            Molmo2ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = Molmo2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Molmo2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config, inputs_dict = config_and_inputs
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            with torch.no_grad():
                _ = model(**inputs_dict)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:1, ...]
            curr_input_dict["image_token_pooling"] = curr_input_dict["image_token_pooling"][:1, ...]
            curr_input_dict["image_grids"] = curr_input_dict["image_grids"][:1, ...]
            # This should work as we're just reducing batch size
            _ = model(**curr_input_dict)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config, inputs_dict = config_and_inputs
        for model_class in self.all_model_classes:
            if model_class.__name__ == "Molmo2ForConditionalGeneration":
                model = model_class(config).to(torch_device)
                model.eval()
                outputs = model(**inputs_dict)
                past_key_values = outputs.past_key_values

                # Generate with past_key_values
                generated_ids = model.generate(
                    input_ids=inputs_dict["input_ids"][:, :5],
                    pixel_values=inputs_dict.get("pixel_values"),
                    image_token_pooling=inputs_dict.get("image_token_pooling"),
                    image_grids=inputs_dict.get("image_grids"),
                    past_key_values=past_key_values,
                    max_new_tokens=5,
                )
                self.assertIsNotNone(generated_ids)
                self.assertEqual(generated_ids.shape[0], inputs_dict["input_ids"].shape[0])

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(**inputs_dict)

            output = outputs[0]

            # Encoder-/Decoder-only models
            hidden_states = outputs.hidden_states[0]
            attentions = outputs.attentions[0]

            hidden_states.retain_grad()
            attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)
            self.assertIsNotNone(attentions.grad)
