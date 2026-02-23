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
"""Testing suite for the PyTorch Param2MoE model."""

import unittest

from transformers import is_torch_available
from transformers.models.param2moe.configuration_param2moe import Param2MoEConfig
from transformers.models.param2moe.modeling_param2moe import Param2MoEForCausalLM, Param2MoEModel
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


# if is_torch_available():
#     import torch

#     from transformers import (
#         Param2MoEForCausalLM,
#         Param2MoEModel,
#     )


class Param2MoEModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        num_experts=8,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=16,
        first_k_dense_replace=1,
        output_router_logits=False,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Param2MoEConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.n_group,
            topk_group=self.topk_group,
            moe_intermediate_size=self.moe_intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            output_router_logits=self.output_router_logits,
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            mtp_loss_scaling_factor=self.mtp_loss_scaling_factor,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = Param2MoEModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = Param2MoEModel(config)
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
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Param2MoEForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Param2MoEForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
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
            output_hidden_states=True,
        )
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.hidden_states[-1].shape[-1]).item()

        output_from_no_past_slice = output_from_no_past.hidden_states[-1][0, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past.hidden_states[-1][0, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[0] == next_tokens.shape[1])
        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_router_logits(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.output_router_logits = True
        model = Param2MoEForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertTrue(result.router_logits is not None)
        # Check that router logits are returned for each layer (except first_k_dense_replace layers)
        expected_num_routers = config.num_hidden_layers - config.first_k_dense_replace
        self.parent.assertEqual(len(result.router_logits), expected_num_routers)
        # Check router logits shape
        for router_logits in result.router_logits:
            self.parent.assertEqual(
                router_logits.shape,
                (self.batch_size, self.seq_length, config.num_experts),
            )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Param2MoEModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Param2MoEModel, Param2MoEForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Param2MoEForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Param2MoEModel, "text-generation": Param2MoEForCausalLM}
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = Param2MoEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Param2MoEConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_router_logits(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_router_logits(*config_and_inputs)

    @unittest.skip(reason="Param2MoE does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Param2MoE has some layers using `eager` attention and don't support SDPA")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Param2MoE has some layers using `eager` attention and don't support SDPA")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Param2MoE buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Param2MoE uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass


@require_torch
class Param2MoEIntegrationTest(unittest.TestCase):
    @slow
    def test_model_param2moe_logits(self):
        # Test with a small model to verify output shapes and basic functionality
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        config = Param2MoEConfig(
            vocab_size=32000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
        )
        model = Param2MoEForCausalLM(config)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        # Check output shape
        expected_shape = (1, 8, 32000)
        self.assertEqual(outputs.logits.shape, expected_shape)

    @slow
    def test_model_param2moe_with_router_logits(self):
        # Test router logits output
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        config = Param2MoEConfig(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            num_experts=8,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            output_router_logits=True,
        )
        model = Param2MoEForCausalLM(config)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, output_router_logits=True)
        # Check that router logits are returned
        self.assertIsNotNone(outputs.router_logits)
        # Should have router logits for all layers except first_k_dense_replace
        expected_num_routers = config.num_hidden_layers - config.first_k_dense_replace
        self.assertEqual(len(outputs.router_logits), expected_num_routers)

    @slow
    def test_model_param2moe_generation(self):
        # Test generation capability
        input_ids = torch.tensor([[1, 2, 3]])
        config = Param2MoEConfig(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = Param2MoEForCausalLM(config)
        model.eval()
        # Test generation
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        # Check that generation extended the sequence
        self.assertEqual(generated.shape[1], input_ids.shape[1] + 5)

    @slow
    def test_model_param2moe_with_mtp(self):
        # Test Multi-Token Prediction functionality
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 6]])
        config = Param2MoEConfig(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
            num_nextn_predict_layers=1,
            mtp_loss_scaling_factor=0.1,
        )
        model = Param2MoEForCausalLM(config)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        # Check that MTP loss is computed
        self.assertIsNotNone(outputs.mtp_loss)
        # Check that MTP logits are returned
        self.assertIsNotNone(outputs.mtp_logits)
        self.assertEqual(len(outputs.mtp_logits), config.num_nextn_predict_layers)
