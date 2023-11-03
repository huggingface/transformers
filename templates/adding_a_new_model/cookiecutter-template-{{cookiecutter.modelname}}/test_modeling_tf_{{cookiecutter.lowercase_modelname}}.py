# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

{% if cookiecutter.is_encoder_decoder_model == "False" %}

import unittest

from transformers import is_tf_available, {{cookiecutter.camelcase_modelname}}Config
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF{{cookiecutter.camelcase_modelname}}ForCausalLM,
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

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_causal_lm_base_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.is_decoder = True

        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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

        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states)

        # Also check the case where encoder outputs are not passed
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_causal_lm_model(
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

    def create_and_check_causal_lm_model_as_decoder(
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

        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs, token_type_ids=token_type_ids, encoder_hidden_states=encoder_hidden_states)

        prediction_scores = result["logits"]
        self.parent.assertListEqual(
            list(prediction_scores.numpy().shape), [self.batch_size, self.seq_length, self.vocab_size]
        )


    def create_and_check_causal_lm_model_past(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True

        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)

        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs.past_key_values

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and attn_mask
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids, output_hidden_states=True).hidden_states[0]
        output_from_past = model(
            next_tokens, past_key_values=past_key_values, output_hidden_states=True
        ).hidden_states[0]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-6)

    def create_and_check_causal_lm_model_past_with_attn_mask(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True

        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)

        # create attention mask
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

        # first forward pass
        outputs = model(input_ids, attention_mask=attn_mask, use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        past_key_values = outputs.past_key_values

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
        vector_condition = tf.range(self.seq_length) == (self.seq_length - random_seq_idx_to_change)
        condition = tf.transpose(
            tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size))
        )
        input_ids = tf.where(condition, random_other_next_tokens, input_ids)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        attn_mask = tf.concat(
            [attn_mask, tf.ones((attn_mask.shape[0], 1), dtype=tf.int32)],
            axis=1,
        )

        output_from_no_past = model(
            next_input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        ).hidden_states[0]
        output_from_past = model(
            next_tokens, past_key_values=past_key_values, attention_mask=attn_mask, output_hidden_states=True
        ).hidden_states[0]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-6)

    def create_and_check_causal_lm_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True

        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)

        input_ids = input_ids[:1, :]
        input_mask = input_mask[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
        past_key_values = outputs.past_key_values

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
        ).hidden_states[0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        ).hidden_states[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

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
        config.add_cross_attention = True

        model = TF{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)

        input_ids = input_ids[:1, :]
        input_mask = input_mask[:1, :]
        encoder_hidden_states = encoder_hidden_states[:1, :, :]
        encoder_attention_mask = encoder_attention_mask[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        ).hidden_states[0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        ).hidden_states[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

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

    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TF{{cookiecutter.camelcase_modelname}}ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        """Test the base model"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Template classes interact badly with this test.")
    def test_keras_fit(self):
        pass

    def test_causal_lm_base_model(self):
        """Test the base model of the causal LM model

        is_deocder=True, no cross_attention, no encoder outputs
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_base_model(*config_and_inputs)

    def test_model_as_decoder(self):
        """Test the base model as a decoder (of an encoder-decoder architecture)

        is_deocder=True + cross_attention + pass encoder outputs
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_causal_lm(self):
        """Test the causal LM model"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_model(*config_and_inputs)

    def test_causal_lm_model_as_decoder(self):
        """Test the causal LM model as a decoder"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_causal_lm_model_as_decoder(*config_and_inputs)

    def test_causal_lm_model_past(self):
        """Test causal LM model with `past_key_values`"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_model_past(*config_and_inputs)

    def test_causal_lm_model_past_with_attn_mask(self):
        """Test the causal LM model with `past_key_values` and `attention_mask`"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_model_past_with_attn_mask(*config_and_inputs)

    def test_causal_lm_model_past_with_large_inputs(self):
        """Test the causal LM model with `past_key_values` and a longer decoder sequence length"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_model_past_large_inputs(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        """Similar to `test_causal_lm_model_past_with_large_inputs` but with cross-attention"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

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

{% else %}
import unittest

from transformers import (
    is_tf_available,
    {{cookiecutter.camelcase_modelname}}Config,
    {{cookiecutter.camelcase_modelname}}Tokenizer,
)
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
        TF{{cookiecutter.camelcase_modelname}}Model,
    )


@require_tf
class TF{{cookiecutter.camelcase_modelname}}ModelTester:
    config_cls = {{cookiecutter.camelcase_modelname}}Config
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size)
        eos_tensor = tf.expand_dims(tf.constant([self.eos_token_id] * self.batch_size), 1)
        input_ids = tf.concat([input_ids, eos_tensor], axis=1)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            **self.config_updates,
        )
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TF{{cookiecutter.camelcase_modelname}}Model(config=config).get_decoder()
        input_ids = inputs_dict["input_ids"]

        input_ids = input_ids[:1, :]
        attention_mask = inputs_dict["attention_mask"][:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)[0]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)


def prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_ids, config.pad_token_id), tf.int32)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.concat([tf.ones(decoder_input_ids[:, :1].shape, dtype=tf.int32), tf.cast(tf.math.not_equal(decoder_input_ids[:, 1:], config.pad_token_id), tf.int32)], axis=-1)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }


@require_tf
class TF{{cookiecutter.camelcase_modelname}}ModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration, TF{{cookiecutter.camelcase_modelname}}Model) if is_tf_available() else ()
    all_generative_model_classes = (TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,) if is_tf_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TF{{cookiecutter.camelcase_modelname}}ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_decoder_model_past_large_inputs(*config_and_inputs)

    @unittest.skip(reason="Template classes interact badly with this test.")
    def test_keras_fit(self):
        pass


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if tf.debugging.assert_near(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f"{prefix}: "
        raise AssertionError(f"{prefix}{a} != {b}")


def _long_tensor(tok_lst):
    return tf.constant(tok_lst, dtype=tf.int32)


TOLERANCE = 1e-4


@slow
@require_sentencepiece
@require_tokenizers
@require_tf
class TF{{cookiecutter.camelcase_modelname}}ModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head(self):
        model = TF{{cookiecutter.camelcase_modelname}}Model.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        # change to intended input here
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(model.config, input_ids, decoder_input_ids)
        output = model(**inputs_dict)[0]
        expected_shape = (1, 11, 1024)
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = tf.Tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]],
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=TOLERANCE)

    def test_inference_with_head(self):
        model = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        # change to intended input here
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(model.config, input_ids, decoder_input_ids)
        output = model(**inputs_dict)[0]
        expected_shape = (1, 11, 1024)
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = tf.Tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]],
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=TOLERANCE)

    def test_seq_to_seq_generation(self):
        hf = TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        tok = {{cookiecutter.camelcase_modelname}}Tokenizer.from_pretrained('{{cookiecutter.checkpoint_identifier}}')

        batch_input = [
            # string 1,
            # string 2,
            # string 3,
            # string 4,
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tok.batch_encode_plus(
            batch_input,
            max_length=512,
            padding="max_length",
            truncation_strategy="only_first",
            truncation=True,
            return_tensors="tf",
        )

        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=2,
        )

        EXPECTED = [
            # here expected 1,
            # here expected 2,
            # here expected 3,
            # here expected 4,
        ]

        generated = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == EXPECTED
{%- endif %}
