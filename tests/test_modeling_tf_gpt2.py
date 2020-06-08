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


import unittest

from transformers import GPT2Config, is_tf_available

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from .utils import require_tf, slow


if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_gpt2 import (
        TFGPT2Model,
        TFGPT2LMHeadModel,
        TFGPT2DoubleHeadsModel,
        TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        shape_list,
    )


@require_tf
class TFGPT2ModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFGPT2Model, TFGPT2LMHeadModel, TFGPT2DoubleHeadsModel) if is_tf_available() else ()
    all_generative_model_classes = (TFGPT2LMHeadModel,) if is_tf_available() else ()

    class TFGPT2ModelTester(object):
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
            self.bos_token_id = vocab_size - 1
            self.eos_token_id = vocab_size - 1

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            mc_token_ids = None
            if self.use_mc_token_ids:
                mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                choice_labels = ids_tensor([self.batch_size], self.num_choices)

            config = GPT2Config(
                vocab_size=self.vocab_size,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                # intermediate_size=self.intermediate_size,
                # hidden_act=self.hidden_act,
                # hidden_dropout_prob=self.hidden_dropout_prob,
                # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                n_positions=self.max_position_embeddings,
                n_ctx=self.max_position_embeddings,
                # type_vocab_size=self.type_vocab_size,
                # initializer_range=self.initializer_range
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
            )

            head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

            return (
                config,
                input_ids,
                input_mask,
                head_mask,
                token_type_ids,
                mc_token_ids,
                sequence_labels,
                token_labels,
                choice_labels,
            )

        def create_and_check_gpt2_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
            model = TFGPT2Model(config=config)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": token_type_ids,
            }
            sequence_output = model(inputs)[0]

            inputs = [input_ids, None, input_mask]  # None is the input for 'past'
            sequence_output = model(inputs)[0]

            sequence_output = model(input_ids)[0]

            result = {
                "sequence_output": sequence_output.numpy(),
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].shape), [self.batch_size, self.seq_length, self.hidden_size],
            )

        def create_and_check_gpt2_model_past(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
            model = TFGPT2Model(config=config)

            # first forward pass
            output, past = model(input_ids, token_type_ids=token_type_ids)

            # create hypothetical next token and extent to next_input_ids
            next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
            next_token_types = ids_tensor([self.batch_size, 1], self.type_vocab_size)

            # append to next input_ids and token_type_ids
            next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
            next_token_type_ids = tf.concat([token_type_ids, next_token_types], axis=-1)

            output_from_no_past, _ = model(next_input_ids, token_type_ids=next_token_type_ids)
            output_from_past, _ = model(next_tokens, token_type_ids=next_token_types, past=past)

            # select random slice
            random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
            output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
            output_from_past_slice = output_from_past[:, 0, random_slice_idx]

            # test that outputs are equal for slice
            tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-6)

        def create_and_check_gpt2_model_attention_mask_past(
            self, config, input_ids, input_mask, head_mask, token_type_ids, *args
        ):
            model = TFGPT2Model(config=config)

            # create attention mask
            half_seq_length = self.seq_length // 2
            attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
            attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
            attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

            # first forward pass
            output, past = model(input_ids, attention_mask=attn_mask)

            # create hypothetical next token and extent to next_input_ids
            next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

            # change a random masked slice from input_ids
            random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
            random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
            vector_condition = tf.range(self.seq_length) == (self.seq_length - random_seq_idx_to_change)
            condition = tf.transpose(
                tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size))
            )
            input_ids = tf.where(condition, random_other_next_tokens, input_ids)

            # append to next input_ids and attn_mask
            next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
            attn_mask = tf.concat([attn_mask, tf.ones((shape_list(attn_mask)[0], 1), dtype=tf.int32)], axis=1)

            # get two different outputs
            output_from_no_past, _ = model(next_input_ids, attention_mask=attn_mask)
            output_from_past, _ = model(next_tokens, past=past, attention_mask=attn_mask)

            # select random slice
            random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
            output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
            output_from_past_slice = output_from_past[:, 0, random_slice_idx]

            # test that outputs are equal for slice
            tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-12)

        def create_and_check_gpt2_lm_head(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
            model = TFGPT2LMHeadModel(config=config)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": token_type_ids,
            }
            prediction_scores = model(inputs)[0]
            result = {
                "prediction_scores": prediction_scores.numpy(),
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size],
            )

        def create_and_check_gpt2_double_head(
            self, config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, *args
        ):
            model = TFGPT2DoubleHeadsModel(config=config)

            multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
            multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
            multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))

            inputs = {
                "input_ids": multiple_choice_inputs_ids,
                "mc_token_ids": mc_token_ids,
                "attention_mask": multiple_choice_input_mask,
                "token_type_ids": multiple_choice_token_type_ids,
            }
            lm_logits, mc_logits = model(inputs)[:2]
            result = {"lm_logits": lm_logits.numpy(), "mc_logits": mc_logits.numpy()}
            self.parent.assertListEqual(
                list(result["lm_logits"].shape), [self.batch_size, self.num_choices, self.seq_length, self.vocab_size],
            )
            self.parent.assertListEqual(list(result["mc_logits"].shape), [self.batch_size, self.num_choices])

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()

            (
                config,
                input_ids,
                input_mask,
                head_mask,
                token_type_ids,
                mc_token_ids,
                sequence_labels,
                token_labels,
                choice_labels,
            ) = config_and_inputs

            inputs_dict = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": input_mask,
            }
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFGPT2ModelTest.TFGPT2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPT2Config, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_gpt2_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt2_model(*config_and_inputs)

    def test_gpt2_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt2_model_past(*config_and_inputs)

    def test_gpt2_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt2_model_attention_mask_past(*config_and_inputs)

    def test_gpt2_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt2_lm_head(*config_and_inputs)

    def test_gpt2_double_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt2_double_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFGPT2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFGPT2ModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gpt2(self):
        model = TFGPT2LMHeadModel.from_pretrained("gpt2")
        input_ids = tf.convert_to_tensor([[464, 3290]], dtype=tf.int32)  # The dog
        expected_output_ids = [
            464,
            3290,
            373,
            1043,
            287,
            257,
            2214,
            1474,
            262,
            16246,
            286,
            2688,
            290,
            2688,
            27262,
            13,
            198,
            198,
            464,
            3290,
        ]  # The dog was found in a field near the intersection of West and West Streets.\n\nThe dog
        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)

    @slow
    def test_lm_generate_distilgpt2(self):
        model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")
        input_ids = tf.convert_to_tensor([[464, 1893]], dtype=tf.int32)  # The president
        expected_output_ids = [
            464,
            1893,
            286,
            262,
            1578,
            1829,
            11,
            290,
            262,
            1893,
            286,
            262,
            1578,
            7526,
            11,
            423,
            587,
            287,
            262,
            2635,
        ]  # The president of the United States, and the president of the United Kingdom, have been in the White

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
