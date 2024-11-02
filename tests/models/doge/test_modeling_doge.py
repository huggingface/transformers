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
        DogeModel,
    )


class DogeModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        type_sequence_label_size=2,
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        hidden_dropout=0.1,
        hidden_act="silu",
        initializer_range=0.02,
        max_position_embeddings=512,
        num_attention_heads=2,
        num_inner_values=1,
        cross_domain_intermediate_size=128,
        private_expert_intermediate_size=32,
        num_cdmmoe_experts=128,
        num_cdmmoe_heads=1,
        num_cdmmoe_experts_per_head=2,
        num_labels=2,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_inner_values = num_inner_values
        self.cross_domain_intermediate_size = cross_domain_intermediate_size
        self.private_expert_intermediate_size = private_expert_intermediate_size
        self.num_cdmmoe_experts = num_cdmmoe_experts
        self.num_cdmmoe_heads = num_cdmmoe_heads
        self.num_cdmmoe_experts_per_head = num_cdmmoe_experts_per_head
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        sequence_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor(
                [self.batch_size], self.type_sequence_label_size
            )

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels

    def get_config(self):
        return DogeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dropout=self.hidden_dropout,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_inner_values=self.num_inner_values,
            cross_domain_intermediate_size=self.cross_domain_intermediate_size,
            private_expert_intermediate_size=self.private_expert_intermediate_size,
            num_cdmmoe_experts=self.num_cdmmoe_experts,
            num_cdmmoe_heads=self.num_cdmmoe_heads,
            num_cdmmoe_experts_per_head=self.num_cdmmoe_experts_per_head,
            num_labels=self.num_labels,
            pad_token_id=self.pad_token_id,
            is_decoder=False,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels):
        model = DogeModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = DogeModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

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
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size)
        )

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = DogeModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[
            :, -3:, random_slice_idx
        ].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(
            torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class DogeModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    all_model_classes = (
        (DogeModel, DogeForCausalLM, DogeForSequenceClassification)
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
    fx_compatible = False
    has_attentions = False

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

    def test_doge_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 2
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size], self.model_tester.type_sequence_label_size
        )
        model = DogeForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    def test_doge_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size], self.model_tester.type_sequence_label_size
        )
        model = DogeForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    def test_doge_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 2
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels],
            self.model_tester.type_sequence_label_size,
        ).to(torch.float)
        model = DogeForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    @unittest.skip(
        reason="doge buffers include complex numbers, which breaks this test"
    )
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("Broken by the loss update will fix soon @ArthurZucker")
    def test_torch_fx_output_loss(self, *args, **kwargs):
        pass
