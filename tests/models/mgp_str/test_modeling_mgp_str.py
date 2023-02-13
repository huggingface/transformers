# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MGPSTR model. """


import unittest

from transformers import MGPSTRConfig
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    from transformers import MGPSTRForSceneTextRecognition


class MGPSTRModelTester:
    def __init__(
        self,
        parent,
        is_training=False,
        batch_size=13,
        image_size=(32, 128),
        patch_size=4,
        num_channels=3,
        max_token_length=27,
        num_character_labels=38,
        num_bpe_labels=50257,
        num_wordpiece_labels=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
    ):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_token_length = max_token_length
        self.num_character_labels = num_character_labels
        self.num_bpe_labels = num_bpe_labels
        self.num_wordpiece_labels = num_wordpiece_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return MGPSTRConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            max_token_length=self.max_token_length,
            num_character_labels=self.num_character_labels,
            num_bpe_labels=self.num_bpe_labels,
            num_wordpiece_labels=self.num_wordpiece_labels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
        )

    def create_and_check_model(self, config, pixel_values):
        model = MGPSTRForSceneTextRecognition(config)
        model.to(torch_device)
        model.eval()
        generated_ids, attens = model(pixel_values)
        self.parent.assertEqual(
            generated_ids[0].shape, (self.batch_size, self.max_token_length, self.num_character_labels)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MGPSTRModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MGPSTRForSceneTextRecognition,) if is_torch_available() else ()
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = MGPSTRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MGPSTRConfig, has_text_modality=False)

    # not implemented currently
    def test_config(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # not implemented currently
    def test_inputs_embeds(self):
        pass

    # not implemented currently
    def test_model_common_attributes(self):
        pass

    # not implemented currently
    def test_forward_signature(self):
        pass

    # not implemented currently
    def test_determinism(self):
        pass

    # not implemented currently
    def test_feed_forward_chunking(self):
        pass

    # not implemented currently
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    # not implemented currently
    def test_hidden_states_output(self):
        pass

    # not implemented currently
    def test_initialization(self):
        pass

    # not implemented currently
    def test_model_main_input_name(self):
        pass

    # not implemented currently
    def test_model_outputs_equivalence(self):
        pass

    # not implemented currently
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # not implemented currently
    def test_save_load(self):
        pass

    # not implemented currently
    def test_save_load_fast_init_from_base(self):
        pass

    # not implemented currently
    def test_save_load_fast_init_to_base(self):
        pass

    # not implemented currently
    def test_tied_model_weights_key_ignore(self):
        pass
