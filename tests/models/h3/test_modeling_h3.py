# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import datetime
import math
import unittest

from transformers import H3Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import H3_PRETRAINED_MODEL_ARCHIVE_LIST, GPT2Tokenizer, H3ForCausalLM, H3Model


class H3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
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
        scope=None,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def get_large_model_config(self):
        return H3Config.from_pretrained("h3")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return H3Config(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.intermediate_size,
            activation_function=self.hidden_act,
            resid_pdrop=self.hidden_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_h3_model(self, config, input_ids, input_mask, *args):
        model = H3Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask):
        model = H3ForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids,}

        return config, inputs_dict


@require_torch
class H3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            H3Model,
            H3ForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (H3ForCausalLM,) if is_torch_available() else ()
    fx_compatible = False
    test_missing_keys = False
    test_model_parallel = True

    def setUp(self):
        self.model_tester = H3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=H3Config, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_h3_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_h3_model(*config_and_inputs)

    def test_h3_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_h3_model_past(*config_and_inputs)

    def test_h3_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_h3_model_attention_mask_past(*config_and_inputs)

    def test_h3_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_h3_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    def test_h3_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_h3_weight_initialization(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        model = H3ForCausalLM.from_pretrained("h3")
        model.to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("h3")

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().cpu().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a mess. I'm not sure if he's going",
            "Today, I'm going to be doing a lot of research on this. I",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        for model_name in H3_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = H3Model.from_pretrained(model_name)
            self.assertIsNotNone(model)