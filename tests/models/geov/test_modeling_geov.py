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
""" Testing suite for the PyTorch GeoV model. """


import unittest

from transformers import AutoTokenizer, GeoVConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import GeoVForCausalLM, GeoVModel


class GeoVModelTester:
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
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        max_position_embeddings=512,
        num_labels=3,
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
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return GeoVConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
        )

    def prepare_config_and_inputs_for_decoder(self):
        config, input_ids, input_mask, token_labels = self.prepare_config_and_inputs()

        config.is_decoder = True

        return config, input_ids, input_mask, token_labels

    def create_and_check_model(self, config, input_ids, input_mask):
        model = GeoVModel(config=config)
        model.to(torch_device)
        model.eval()
        _ = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, input_mask):
        config.add_cross_attention = True
        model = GeoVModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, token_labels):
        model = GeoVForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, input_mask):
        config.is_decoder = True
        model = GeoVForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True)
        output_from_no_past = output_from_no_past["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, token_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GeoVModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (GeoVModel, GeoVForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (GeoVForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = GeoVModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GeoVConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, input_mask)

    def test_model_as_decoder(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_decoder_model_past_large_inputs(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(config, input_ids, input_mask)

    def test_model_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass


@require_torch
class GeoVLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/vpj/data/hf_9b")
        for checkpointing in [True, False]:
            model = GeoVForCausalLM.from_pretrained("/home/ubuntu/vpj/data/hf_9b")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            expected_output = (
                "My favorite food is the chicken and rice.\n\nI love to cook and bake. I love to cook and bake"
            )

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)
