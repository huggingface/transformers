# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Opf model."""

import unittest

from transformers import (
    OPF_NER_LABELS,
    GptOssConfig,
    OpfConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    from transformers import (
        OpfForTokenClassification,
        OpfModel,
    )


class OpfModelTester:
    base_model_class = None
    config_class = OpfConfig
    token_classification_class = None

    if is_torch_available():
        base_model_class = OpfModel
        token_classification_class = OpfForTokenClassification

    @property
    def all_model_classes(self):
        return tuple(
            model_class
            for model_class in (
                self.base_model_class,
                self.token_classification_class,
            )
            if model_class is not None
        )

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        bidirectional_left_context=2,
        bidirectional_right_context=2,
        initial_context_length=64,
        max_position_embeddings=64,
        initializer_range=0.02,
        num_labels=len(OPF_NER_LABELS),
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.bidirectional_left_context = bidirectional_left_context
        self.bidirectional_right_context = bidirectional_right_context
        self.initial_context_length = initial_context_length
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

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
        return self.config_class(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            sliding_window=self.bidirectional_left_context + self.bidirectional_right_context + 1,
            bidirectional_left_context=self.bidirectional_left_context,
            bidirectional_right_context=self.bidirectional_right_context,
            initial_context_length=self.initial_context_length,
            max_position_embeddings=self.max_position_embeddings,
            default_n_ctx=self.max_position_embeddings,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 150000.0,
                "factor": 1.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": self.initial_context_length,
            },
            initializer_range=self.initializer_range,
            classifier_dropout=0.0,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = self.base_model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_token_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = self.token_classification_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, token_labels = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class OpfModelTest(ModelTesterMixin, unittest.TestCase):
    has_attentions = False
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.6]
    model_tester_class = OpfModelTester
    all_model_classes = None

    def setUp(self):
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, hidden_size=32)

        if self.all_model_classes is None:
            self.all_model_classes = self.model_tester.all_model_classes

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_inherits_gpt_oss_defaults_without_causal_overrides(self):
        config = OpfConfig()

        self.assertIsInstance(config, GptOssConfig)
        self.assertEqual(config.use_cache, False)
        self.assertEqual(config.router_aux_loss_coef, 0.0)
        self.assertEqual(config.base_model_ep_plan, {})
        self.assertEqual(config.layer_types, ["sliding_attention"] * config.num_hidden_layers)
        self.assertEqual(config.rope_parameters["rope_theta"], config.rope_theta)
        self.assertNotIn("mup_scale", config.to_dict())
        self.assertNotIn("logit_multiplier", config.to_dict())

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_set_use_kernels(self):
        config, _, _, _ = self.model_tester.prepare_config_and_inputs()
        model = OpfModel(config)

        self.assertIsNone(model.set_use_kernels(False))
        self.assertFalse(model.use_kernels)
        with self.assertRaisesRegex(ValueError, "OPF does not support kernelized layers."):
            model.set_use_kernels(True)
