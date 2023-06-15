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
""" Testing suite for the PyTorch VITS model. """

import copy
import inspect
import tempfile
import unittest

from transformers import VitsConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from transformers import VitsModel


@require_torch
class VitsModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
        vocab_size=38,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(2)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return VitsConfig(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            inter_channels=self.intermediate_size,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = VitsModel(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        result = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.audio.shape,
            (self.batch_size, self.seq_length * 256),
        )


@require_torch
class VitsModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VitsModel,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    input_name = "input_ids"

    def setUp(self):
        self.model_tester = VitsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VitsConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass

    @slow
    def test_torchscript_output_attentions(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_simple(self):
        # disabled because this model doesn't have decoder_input_ids
        pass
