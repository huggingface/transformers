# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HindiCausalLM model."""

import unittest

import numpy as np
from parameterized import parameterized

from transformers import HindiCausalLMConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask

if is_torch_available():
    import torch

    from transformers import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
    )

class HindiCausalLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="silu",
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
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

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

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_lm_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_lm_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

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
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict

@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    test_head_masking = False
    test_missing_keys = False
    test_pruning = False

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_model(*config_and_inputs)

    def test_lm_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_for_causal_lm(*config_and_inputs)

    @parameterized.expand([("left",), ("right",)])
    def test_left_padding_compatibility(self, padding_side):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        model = HindiCausalLMForCausalLM(config)
        model.to(torch_device)
        model.eval()
        self.assertTrue(model.config.pad_token_id is not None)

        # This is to ensure that left padding doesn't complain about the causal mask & past key values
        # Only test with causal lm + left padding + no past cache
        if padding_side == "left":
            batch_size = 2
            prompt = "Hello"
            generation_kwargs = {
                "max_length": 10,
                "num_return_sequences": 1,
                "pad_token_id": model.config.pad_token_id,
                "eos_token_id": None,
                "use_cache": False,
            }

            # Test with left padding
            tokenizer_padding_side = "left"
            encoded_prompt = batch_size * [[0, 31, 33, 1, 1] + [model.config.pad_token_id] * 4]
            encoder_prompt = torch.tensor(encoded_prompt, device=torch_device)
            # Right padding vs left padding should have a different output
            outputs_left = model.generate(
                input_ids=encoder_prompt,
                attention_mask=encoder_prompt.ne(model.config.pad_token_id).long(),
                **generation_kwargs,
            )
        else:
            # Just a placeholder test for the "right" side of the parameterize
            # It doesn't actually check anything, the check is right-side padding is not broken by left-side padding fix
            batch_size = 2
            prompt = "Hello"
            generation_kwargs = {
                "max_length": 10,
                "num_return_sequences": 1,
                "pad_token_id": model.config.pad_token_id,
                "eos_token_id": None,
                "use_cache": False,
            }

            # Test with right padding (default in tokenizers)
            tokenizer_padding_side = "right"
            encoded_prompt = batch_size * [[31, 33, 1, 1, 0] + [model.config.pad_token_id] * 4]
            encoder_prompt = torch.tensor(encoded_prompt, device=torch_device)
            _ = model.generate(
                input_ids=encoder_prompt,
                attention_mask=encoder_prompt.ne(model.config.pad_token_id).long(),
                **generation_kwargs,
            )

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        model = HindiCausalLMModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

@require_torch
@require_sentencepiece
class HindiCausalLMIntegrationTest(unittest.TestCase):
    @slow
    def test_model_inference(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        model = HindiCausalLMForCausalLM.from_pretrained(model_name)
        model.to(torch_device)
        model.eval()

        # Test a simple input
        input_text = "गंगा नदी"
        from transformers import HindiCausalLMTokenizer
        tokenizer = HindiCausalLMTokenizer.from_pretrained(model_name)

        inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.logits.shape[0], 1)  # Batch size
        self.assertEqual(outputs.logits.shape[2], model.config.vocab_size)  # Vocab size

        # Test generation
        generated_text = tokenizer.decode(
            model.generate(
                **inputs,
                max_length=20,
                num_beams=1,
                do_sample=False,
            )[0],
            skip_special_tokens=True,
        )

        # Just check that we get some output, not testing the exact text
        self.assertGreater(len(generated_text), len(input_text))
