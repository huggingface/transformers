# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023 G42 Systems.
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

"""Testing suite for the PyTorch JAIS model."""

import unittest
from typing import List

from parameterized import parameterized

from transformers import JAISConfig, StaticCache, is_torch_available, set_seed
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        JAISLMHeadModel,
        JAISForSequenceClassification,
        JAISForTokenClassification,
        JAISModel,
    )


class JAISModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
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
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return JAISConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = JAISModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder
    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = JAISModel(config)
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
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm
    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = JAISLMHeadModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = JAISLMHeadModel(config=config)
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
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class JAISModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (JAISModel, JAISLMHeadModel, JAISForSequenceClassification, JAISForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (JAISLMHeadModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": JAISModel,
            "text-classification": JAISForSequenceClassification,
            "text-generation": JAISLMHeadModel,
            "token-classification": JAISForTokenClassification,
            "zero-shot": JAISForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.setUp
    def setUp(self):
        self.model_tester = JAISModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JAISConfig, hidden_size=37)

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model
    def test_jais_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = JAISForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model_for_single_label
    def test_jais_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = JAISForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_sequence_classification_model_for_multi_label
    def test_jais_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = JAISForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # This funstion is mostly unnecessary for JAIS since it is based on ALiBi and not RoPE
    @parameterized.expand([("longrope",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = JAISModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        n_factors = config.hidden_size // config.num_attention_heads // 2
        config.rope_scaling = {
            "type": scaling_type,
            "short_factor": [5.0 for _ in range(n_factors)],
            "long_factor": [5.0 for _ in range(n_factors)],
        }
        scaled_model = JAISModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Scaling changes the RoPE embeddings, both for the short and long outputs
        self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))


@slow
@require_torch
class JAISIntegrationTest(unittest.TestCase):
    def test_model_jais_family_590m_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = JAISLMHeadModel.from_pretrained("inceptionai/jais-family-590m").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[0.5481, 1.8174, 4.4284, 0.9861, 2.2728, 1.0478, 2.3018, 4.5767, 4.3964, 5.4515, 3.7641, 1.2898, 7.0074, 5.5592, 6.7922, 3.8314, 2.1620, 4.1629, 4.5003, 3.2029, 2.1505, 1.8232, 1.6246, 1.9965, 2.1022, 1.0454, 5.4607, 5.3709, 2.6574, 2.5886], [0.7051, 3.6388, 6.9868, 1.4463, 2.2418, 4.2136, 1.6641, 2.8373, 4.1799, 7.0167, 2.7028, 1.4825, 7.2492, 4.9976, 7.6055, 4.4547, 2.0052, 2.3634, 3.0697, 1.8391, 2.1821, 1.5839, 0.8878, 1.5438, 1.8150, 0.9438, 6.3197, 4.3480, 2.8423, 2.2940]]).to(torch_device)  # fmt: skip

        self.assertTrue(torch.allclose(EXPECTED_OUTPUT, output[0, :2, :30], atol=1e-4, rtol=1e-4))

    def test_jais_family_590m_chat_generation(self):
        model = JAISLMHeadModel.from_pretrained("inceptionai/jais-family-590m-chat")
        tokenizer = AutoTokenizer.from_pretrained("inceptionai/jais-family-590m-chat")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(inputs, max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "### Instruction: You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] Can you provide ways to eat combinations of bananas and dragonfruits? \n[|AI|]\n### Response:Absolutely, there are several ways to incorporate bananas and dragonfruits into a balanced diet. Here are some suggestions:\n\n1. **Ban"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)
