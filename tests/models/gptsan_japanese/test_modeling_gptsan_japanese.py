# coding=utf-8
# Copyright 2023 Google GPTSANJapanese Authors and HuggingFace Inc. team.
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


import unittest

from transformers import GPTSANJapaneseConfig, GPTSANJapaneseModel, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


class GPTSANJapaneseModelTester:
    def __init__(
        self,
        parent,
        vocab_size=36000,
        batch_size=13,
        num_contexts=7,
        # For common tests
        is_training=True,
        hidden_size=32,
        ext_size=42,
        num_hidden_layers=5,
        num_ext_layers=2,
        num_attention_heads=4,
        num_experts=2,
        d_ff=32,
        d_ext=80,
        d_spout=33,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        expert_capacity=100,
        router_jitter_noise=0.0,
    ):

        self.vocab_size = vocab_size
        self.parent = parent
        self.batch_size = batch_size
        self.num_contexts = num_contexts
        # For common tests
        self.seq_length = self.num_contexts
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_ext_layers = num_ext_layers
        self.ext_size = ext_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.d_ff = d_ff
        self.d_ext = d_ext
        self.d_spout = d_spout
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise

    def get_large_model_config(self):
        return GPTSANJapaneseConfig.from_pretrained("Tanrei/GPTSAN-japanese")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, input_ids)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (config, {"input_ids": input_ids})

    def get_config(self):
        return GPTSANJapaneseConfig(
            vocab_size=36000,
            num_contexts=self.seq_length,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_ext=self.d_ext,
            d_spout=self.d_spout,
            num_switch_layers=self.num_hidden_layers - self.num_ext_layers,
            num_ext_layers=self.num_ext_layers,
            num_heads=self.num_attention_heads,
            num_experts=self.num_experts,
            expert_capacity=self.expert_capacity,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            router_jitter_noise=self.router_jitter_noise,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
    ):
        model = GPTSANJapaneseModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
        )
        assert result


@require_torch
class GPTSANJapaneseModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (GPTSANJapaneseModel,) if is_torch_available() else ()
    fx_compatible = False
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_cpu_offload = False
    test_disk_offload = False
    test_model_parallelism = False
    test_retain_grad_hidden_states_attentions = False
    # The small GPTSAN_JAPANESE model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = GPTSANJapaneseModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTSANJapaneseConfig, d_model=37)

    def test_config(self):
        GPTSANJapaneseConfig()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)
