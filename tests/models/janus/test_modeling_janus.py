# coding=utf-8
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
"""Testing suite for the PyTorch Janus model."""

import unittest

from transformers import (
    JanusConfig,
    JanusForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    pass


class JanusVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        # image_token_index=0,
        seq_length=25,
        initializer_range=0.02,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
            "vision_feature_select_strategy": "default",
            "vision_feature_layer": -1,
        },
        use_cache=False,
        vq_num_embeds=12,
        vq_embed_dim=12,
        vq_channel_multiplier=[1, 2],
        vq_img_token_start_id=10,  # has to be less than vocab size when added with vq_num_embeds
    ):
        self.parent = parent
        self.initializer_range = initializer_range
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        # self.image_token_index = image_token_index
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.num_image_tokens = vision_config["num_image_tokens"]
        self.use_cache = use_cache

        # vq model params
        self.vq_num_embeds = vq_num_embeds
        self.vq_embed_dim = vq_embed_dim
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id

    def get_vq_config(self):
        return {
            "embed_dim": self.vq_embed_dim,
            "num_embeddings": self.vq_num_embeds,
            "latent_channels": self.vq_embed_dim,
            "in_channels": 3,
            "base_channels": 32,  # we have a GroupNorm of 32 groups, so can't do less
            "channel_multiplier": self.vq_channel_multiplier,
            "initializer_range": self.initializer_range,
        }

    def get_config(self):
        return JanusConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vq_config=self.get_vq_config(),
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # print(config)
        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # set the 16 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        # input_ids[input_ids == config.image_token_index] = self.pad_token_id
        # input_ids[:, :self.num_image_tokens] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
class JanusForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `PaliGemmaForConditionalGeneration`.
    """

    all_model_classes = (JanusForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": JanusForConditionalGeneration}
    fx_compatible = False
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = JanusVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JanusConfig, has_text_modality=False)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)
