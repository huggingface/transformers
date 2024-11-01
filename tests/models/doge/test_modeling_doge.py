# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Doge model."""

import unittest


from transformers import DogeConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        DogeForCausalLM,
        DogeForSequenceClassification,
        DogeModel
    )


class DogeModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        cross_domain_intermediate_size=128,
        private_expert_intermediate_size=32,
        num_cdmmoe_experts=2,
        hidden_act="gelu",
        hidden_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        num_labels=3,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.cross_domain_intermediate_size = cross_domain_intermediate_size
        self.private_expert_intermediate_size = private_expert_intermediate_size
        self.num_cdmmoe_experts = num_cdmmoe_experts
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id


    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return DogeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            cross_domain_intermediate_size=self.cross_domain_intermediate_size,
            private_expert_intermediate_size=self.private_expert_intermediate_size,
            num_cdmmoe_experts=self.num_cdmmoe_experts,
            hidden_act=self.hidden_act,
            hidden_dropout=self.hidden_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, input_mask
    ):
        model = DogeModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        model = DogeForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class LlamaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DogeModel,
            DogeForCausalLM,
            DogeForSequenceClassification
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (DogeForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DogeModel,
            "text-classification": DogeForSequenceClassification,
            "text-generation": DogeForCausalLM,
            "zero-shot": DogeForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DogeForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = DogeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DogeConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


    @unittest.skip(reason="doge buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

