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
"""Testing suite for the PyTorch LFM2Bidirectional model."""

import unittest

from transformers import Lfm2BidirectionalConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import Lfm2BidirectionalModel


class Lfm2BidirectionalModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        conv_L_cache=3,
        layer_types=["conv", "full_attention"],
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.conv_L_cache = conv_L_cache
        self.layer_types = layer_types
        self.num_hidden_layers = len(layer_types)
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self):
        return Lfm2BidirectionalConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            conv_L_cache=self.conv_L_cache,
            layer_types=self.layer_types,
            block_auto_adjust_ff_dim=False,
            initializer_range=self.initializer_range,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Lfm2BidirectionalModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Lfm2BidirectionalModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Lfm2BidirectionalModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Lfm2BidirectionalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Lfm2BidirectionalConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        """LFM2Bidirectional alternates between attention and short-conv layers, so only attention layers
        produce entries in `outputs.attentions`."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])


@require_torch
@slow
class Lfm2BidirectionalIntegrationTest(unittest.TestCase):
    pass
