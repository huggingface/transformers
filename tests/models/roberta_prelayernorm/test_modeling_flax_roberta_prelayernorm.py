# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np

from transformers import RobertaPreLayerNormConfig, is_flax_available
from transformers.testing_utils import require_flax, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_flax_available():
    import jax.numpy as jnp

    from transformers.models.roberta_prelayernorm.modeling_flax_roberta_prelayernorm import (
        FlaxRobertaPreLayerNormForCausalLM,
        FlaxRobertaPreLayerNormForMaskedLM,
        FlaxRobertaPreLayerNormForMultipleChoice,
        FlaxRobertaPreLayerNormForQuestionAnswering,
        FlaxRobertaPreLayerNormForSequenceClassification,
        FlaxRobertaPreLayerNormForTokenClassification,
        FlaxRobertaPreLayerNormModel,
    )


# Copied from tests.models.roberta.test_modeling_flax_roberta.FlaxRobertaModelTester with Roberta->RobertaPreLayerNorm
class FlaxRobertaPreLayerNormModelTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_attention_mask=True,
        use_token_type_ids=True,
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
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
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
        self.num_choices = num_choices
        super().__init__()

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = RobertaPreLayerNormConfig(
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
        )

        return config, input_ids, token_type_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_decoder(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, attention_mask = config_and_inputs

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )


@require_flax
# Copied from tests.models.roberta.test_modeling_flax_roberta.FlaxRobertaModelTest with ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,FacebookAI/roberta-base->andreasmadsen/efficient_mlm_m0.40
class FlaxRobertaPreLayerNormModelTest(FlaxModelTesterMixin, unittest.TestCase):
    test_head_masking = True

    all_model_classes = (
        (
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
        )
        if is_flax_available()
        else ()
    )

    def setUp(self):
        self.model_tester = FlaxRobertaPreLayerNormModelTester(self)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("andreasmadsen/efficient_mlm_m0.40", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)


@require_flax
class TFRobertaPreLayerNormModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = FlaxRobertaPreLayerNormForMaskedLM.from_pretrained("andreasmadsen/efficient_mlm_m0.40", from_pt=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]], dtype=jnp.int32)
        output = model(input_ids)[0]
        expected_shape = [1, 11, 50265]
        self.assertEqual(list(output.shape), expected_shape)
        # compare the actual values for a slice.
        EXPECTED_SLICE = np.array(
            [[[40.4880, 18.0199, -5.2367], [-1.8877, -4.0885, 10.7085], [-2.2613, -5.6110, 7.2665]]], dtype=np.float32
        )
        self.assertTrue(np.allclose(output[:, :3, :3], EXPECTED_SLICE, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = FlaxRobertaPreLayerNormModel.from_pretrained("andreasmadsen/efficient_mlm_m0.40", from_pt=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]], dtype=jnp.int32)
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        EXPECTED_SLICE = np.array(
            [[[0.0208, -0.0356, 0.0237], [-0.1569, -0.0411, -0.2626], [0.1879, 0.0125, -0.0089]]], dtype=np.float32
        )
        self.assertTrue(np.allclose(output[:, :3, :3], EXPECTED_SLICE, atol=1e-4))
