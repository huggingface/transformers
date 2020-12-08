# coding=utf-8
# Copyright {{cookiecutter.authors}} and The HuggingFace Inc. team. All rights reserved.
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

from transformers import {{cookiecutter.camelcase_modelname}}Config, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
        TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
        TF{{cookiecutter.camelcase_modelname}}ForCausalLM,
        TF{{cookiecutter.camelcase_modelname}}Model,
    )


class TF{{cookiecutter.camelcase_modelname}}ModelTester:
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
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

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

        config = {{cookiecutter.camelcase_modelname}}Config(
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
            initializer_range=self.initializer_range,
            return_dict=True,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_lm_head(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.is_decoder = True
        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        prediction_scores = model(inputs)["logits"]
        self.parent.assertListEqual(
            list(prediction_scores.numpy().shape), [self.batch_size, self.seq_length, self.vocab_size]
        )

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TF{{cookiecutter.camelcase_modelname}}ForMaskedLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice(config=config)
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
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TF{{cookiecutter.camelcase_modelname}}ForTokenClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

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
class TF{{cookiecutter.camelcase_modelname}}ModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TF{{cookiecutter.camelcase_modelname}}Model,
            TF{{cookiecutter.camelcase_modelname}}ForCausalLM,
            TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
            TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
            TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        )
        if is_tf_available()
        else ()
    )

    def setUp(self):
        self.model_tester = TF{{cookiecutter.camelcase_modelname}}ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = TF{{cookiecutter.camelcase_modelname}}Model.from_pretrained("{{cookiecutter.checkpoint_identifier}}")
        self.assertIsNotNone(model)

@require_tf
class TF{{cookiecutter.camelcase_modelname}}ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TF{{cookiecutter.camelcase_modelname}}ForMaskedLM.from_pretrained("{{cookiecutter.checkpoint_identifier}}")
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 32000

        expected_shape = [1, 6, vocab_size]
        self.assertEqual(output.shape, expected_shape)

        print(output[:, :3, :3])

        # TODO Replace values below with what was printed above.
        expected_slice = tf.constant(
            [
                [
                    [-0.05243197, -0.04498899, 0.05512108],
                    [-0.07444685, -0.01064632, 0.04352357],
                    [-0.05020351, 0.05530146, 0.00700043],
                ]
            ]
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=1e-4)
