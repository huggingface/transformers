# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the MIT License.
#

import unittest

from transformers import (
    HindiCausalLMConfig,
    HindiCausalLMForCausalLM,
    HindiCausalLMModel,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch


class HindiCausalLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=16000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        positional_encoding_type="rope",
        normalization_layer="rmsnorm",
        num_labels=3,
        num_choices=4,
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
        self.positional_encoding_type = positional_encoding_type
        self.normalization_layer = normalization_layer
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

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

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
            positional_encoding_type=self.positional_encoding_type,
            normalization_layer=self.normalization_layer,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
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
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        model = HindiCausalLMModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class HindiCausalLMModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_lm_head_model(self):
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-foundational-model-base")
        model.to(torch_device)
        model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=torch_device)
        with torch.no_grad():
            output = model(input_ids)

        vocab_size = model.config.vocab_size
        self.assertEqual(output.logits.shape, torch.Size((1, 6, vocab_size)))

    @slow
    def test_generation(self):
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-foundational-model-base")
        model.to(torch_device)
        model.eval()

        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-foundational-model-base")

        input_text = "हिंदी भाषा"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        # Test greedy generation
        outputs = model.generate(input_ids, max_length=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertGreater(len(generated_text), len(input_text))

        # Test sampling
        outputs = model.generate(input_ids, max_length=20, do_sample=True, top_k=50, top_p=0.95)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertGreater(len(generated_text), len(input_text))
