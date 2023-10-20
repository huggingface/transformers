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
""" Testing suite for the PyTorch Molformer model. """


import unittest

from transformers import MolformerConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MolformerForMaskedLM,
        MolformerForSequenceClassification,
        MolformerModel,
    )
    from transformers.models.molformer.modeling_molformer import (
        MOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class MolformerModelTester:
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
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        num_random_features=8,
        deterministic_eval=True,
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
        self.num_random_features = num_random_features
        self.deterministic_eval = deterministic_eval
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

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return MolformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            num_random_features=self.num_random_features,
            deterministic_eval=self.deterministic_eval,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MolformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MolformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MolformerForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

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
class MolformerModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MolformerModel,
            MolformerForMaskedLM,
            MolformerForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MolformerModel,
            "fill-mask": MolformerForMaskedLM,
            "text-classification": MolformerForSequenceClassification,
            "zero-shot": MolformerForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MolformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MolformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in MOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MolformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class MolformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = MolformerForMaskedLM.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True)

        input_ids = torch.tensor([[0, 4, 4, 6, 12, 9, 7, 9, 5, 8, 5, 5, 5, 5, 5, 8, 4, 6, 12, 9, 7, 9, 1]])
        with torch.no_grad():
            output = model(input_ids)[0]

        expected_shape = torch.Size((1, 23, 2362))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-12.1811, -12.2346, -12.3054], [-9.0371, -11.0436, -10.6556], [-8.2395, -9.6526, -9.6713]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = MolformerModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True)

        input_ids = torch.tensor([[0, 4, 4, 6, 12, 9, 7, 9, 5, 8, 5, 5, 5, 5, 5, 8, 4, 6, 12, 9, 7, 9, 1]])
        with torch.no_grad():
            output = model(input_ids)[1]

        expected_shape = torch.Size((1, 768))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[5.9427e-01, 4.5265e-01, 3.4374e-01, 6.6782e-01, -6.9957e-01]])

        self.assertTrue(torch.allclose(output[:, :5], expected_slice, atol=1e-4))
