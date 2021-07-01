# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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

from transformers import FunnelConfig, is_tf_available
from transformers.testing_utils import require_tf

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFFunnelBaseModel,
        TFFunnelForMaskedLM,
        TFFunnelForMultipleChoice,
        TFFunnelForPreTraining,
        TFFunnelForQuestionAnswering,
        TFFunnelForSequenceClassification,
        TFFunnelForTokenClassification,
        TFFunnelModel,
    )


class TFFunnelModelTester:
    """You can also import this e.g, from .test_modeling_funnel import FunnelModelTester"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        block_sizes=[1, 1, 2],
        num_decoder_layers=1,
        d_model=32,
        n_head=4,
        d_head=8,
        d_inner=37,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=3,
        num_labels=3,
        num_choices=4,
        scope=None,
        base=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = 2
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

        # Used in the tests to check the size of the first attention layer
        self.num_attention_heads = n_head
        # Used in the tests to check the size of the first hidden state
        self.hidden_size = self.d_model
        # Used in the tests to check the number of output hidden states/attentions
        self.num_hidden_layers = sum(self.block_sizes) + (0 if base else self.num_decoder_layers)
        # FunnelModel adds two hidden layers: input embeddings and the sum of the upsampled encoder hidden state with
        # the last hidden state of the first block (which is the first hidden state of the decoder).
        if not base:
            self.expected_num_hidden_layers = self.num_hidden_layers + 2

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

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

        config = FunnelConfig(
            vocab_size=self.vocab_size,
            block_sizes=self.block_sizes,
            num_decoder_layers=self.num_decoder_layers,
            d_model=self.d_model,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_inner,
            hidden_act=self.hidden_act,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
        )

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = TFFunnelModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

        config.truncate_seq = False
        model = TFFunnelModel(config=config)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

        config.separate_cls = False
        model = TFFunnelModel(config=config)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

    def create_and_check_base_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = TFFunnelBaseModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 2, self.d_model))

        config.truncate_seq = False
        model = TFFunnelBaseModel(config=config)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 3, self.d_model))

        config.separate_cls = False
        model = TFFunnelBaseModel(config=config)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 2, self.d_model))

    def create_and_check_for_pretraining(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = TFFunnelForPreTraining(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = TFFunnelForMaskedLM(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_labels = self.num_labels
        model = TFFunnelForSequenceClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_choices = self.num_choices
        model = TFFunnelForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_labels = self.num_labels
        model = TFFunnelForTokenClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = TFFunnelForQuestionAnswering(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

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


@require_tf
class TFFunnelModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFFunnelModel,
            TFFunnelForMaskedLM,
            TFFunnelForPreTraining,
            TFFunnelForQuestionAnswering,
            TFFunnelForTokenClassification,
        )
        if is_tf_available()
        else ()
    )
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFFunnelModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FunnelConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_saved_model_creation(self):
        # This test is too long (>30sec) and makes fail the CI
        pass

    def test_compile_tf_model(self):
        # This test fails the CI. TODO Lysandre re-enable it
        pass


@require_tf
class TFFunnelBaseModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFFunnelBaseModel, TFFunnelForMultipleChoice, TFFunnelForSequenceClassification) if is_tf_available() else ()
    )
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFFunnelModelTester(self, base=True)
        self.config_tester = ConfigTester(self, config_class=FunnelConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_base_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_base_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_saved_model_creation(self):
        # This test is too long (>30sec) and makes fail the CI
        pass
