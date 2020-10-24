# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


{% if cookiecutter.is_encoder_decoder_model == "False" -%}
import unittest

from transformers import {{cookiecutter.camelcase_modelname}}Config, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers.modeling_tf_{{cookiecutter.lowercase_modelname}} import (
        TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
        TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
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

    def create_and_check_{{cookiecutter.lowercase_modelname}}_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        sequence_output = model(inputs)[0]

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_{{cookiecutter.lowercase_modelname}}_for_masked_lm(
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

    def create_and_check_{{cookiecutter.lowercase_modelname}}_for_sequence_classification(
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

    def create_and_check_{{cookiecutter.lowercase_modelname}}_for_multiple_choice(
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

    def create_and_check_{{cookiecutter.lowercase_modelname}}_for_token_classification(
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

    def create_and_check_{{cookiecutter.lowercase_modelname}}_for_question_answering(
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

    def test_{{cookiecutter.lowercase_modelname}}_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_{{cookiecutter.lowercase_modelname}}_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = TF{{cookiecutter.camelcase_modelname}}Model.from_pretrained("{{cookiecutter.checkpoint_identifier}}")
        self.assertIsNotNone(model)

{% else -%}

import tempfile
import unittest

from transformers import is_tf_available
from transformers.testing_utils import require_tf

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import {{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration, TF{{cookiecutter.camelcase_modelname}}Model


@require_tf
class ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_labels = False
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 20
        self.eos_token_ids = [2]
        self.pad_token_id = 1
        self.bos_token_id = 0

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size)
        eos_tensor = tf.expand_dims(tf.constant([2] * self.batch_size), 1)
        input_ids = tf.concat([input_ids, eos_tensor], axis=1)
        input_ids = tf.clip_by_value(input_ids, 3, self.vocab_size + 1)

        config = {{cookiecutter.camelcase_modelname}}Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_intermediate_dim=self.intermediate_size,
            decoder_intermediate_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
        )
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(config, input_ids)
        return config, inputs_dict


def prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_ids, config.pad_token_id), tf.int8)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@require_tf
class TestTF{{cookiecutter.camelcase_modelname}}(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration, TF{{cookiecutter.camelcase_modelname}}Model) if is_tf_available() else ()
    all_generative_model_classes = (TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,) if is_tf_available() else ()
    is_encoder_decoder = True
    test_pruning = False

    def setUp(self):
        self.model_tester = ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # inputs_embeds not supported
        pass

    def test_compile_tf_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        model_class = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration
        input_ids = {
            "decoder_input_ids": tf.keras.Input(batch_shape=(2, 2000), name="decoder_input_ids", dtype="int32"),
            "input_ids": tf.keras.Input(batch_shape=(2, 2000), name="input_ids", dtype="int32"),
        }

        # Prepare our model
        model = model_class(config)
        model(self._prepare_for_class(inputs_dict, model_class))  # Model must be called before saving.
        # Let's load it from the disk to be sure we can use pretrained weights
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = model_class.from_pretrained(tmpdirname)

        outputs_dict = model(input_ids)
        hidden_states = outputs_dict[0]

        # Add a dense layer on top to test integration with other keras modules
        outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

        # Compile extended model
        extendehidden_size = tf.keras.Model(inputs=[input_ids], outputs=[outputs])
        extendehidden_size.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def test_savehidden_size_with_hidden_states_output(self):
        # Should be uncommented during patrick TF refactor
        pass

    def test_savehidden_size_with_attentions_output(self):
        # Should be uncommented during patrick TF refactor
        pass


@require_tf
class TF{{cookiecutter.camelcase_modelname}}HeadTests(unittest.TestCase):

    vocab_size = 99

    def _get_config_and_data(self):
        eos_column_vector = tf.ones((4, 1), dtype=tf.int32) * 2
        input_ids = tf.concat([ids_tensor((4, 6), self.vocab_size - 3) + 3, eos_column_vector], axis=1)
        batch_size = input_ids.shape[0]
        config = {{cookiecutter.camelcase_modelname}}Config(
            vocab_size=self.vocab_size,
            hidden_size=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_intermediate_dim=32,
            decoder_intermediate_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            return_dict=True,
            decoder_start_token_id=2,
        )
        return config, input_ids, batch_size

    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        decoder_lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size)
        lm_model = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration(config)
        outputs = lm_model(inputs=input_ids, lm_labels=decoder_lm_labels, decoder_input_ids=input_ids, use_cache=False)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_lm_uneven_forward(self):
        config = {{cookiecutter.camelcase_modelname}}Config(
            vocab_size=10,
            hidden_size=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_intermediate_dim=32,
            decoder_intermediate_dim=32,
            max_position_embeddings=48,
            return_dict=True,
        )
        lm_model = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration(config)
        context = tf.fill((7, 2), 4)
        summary = tf.fill((7, 7), 6)
        outputs = lm_model(inputs=context, decoder_input_ids=summary, use_cache=False)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if tf.debugging.assert_near(a, b, atol=atol):
            return True
        raise
    except Exception:
        msg = "{} != {}".format(a, b)
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


{% endif -%}
