# coding=utf-8
# Copyright 2024 Google CodeReviewer Authors and HuggingFace Inc. team.
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

from __future__ import annotations

import unittest

from transformers import CodeReviewerConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFCodeReviewerEncoderModel,
        TFCodeReviewerForConditionalGeneration,
        TFCodeReviewerForQuestionAnswering,
        TFCodeReviewerForSequenceClassification,
        TFCodeReviewerModel,
    )


class TFCodeReviewerModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_labels = True
        self.vocab_size = 99
        self.n_positions = 14
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.d_ff = 37
        self.relative_attention_num_buckets = 8
        self.dropout_rate = 0.1
        self.initializer_factor = 0.002
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = CodeReviewerConfig(
            vocab_size=self.vocab_size,
            n_positions=self.n_positions,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
        )

        return (config, input_ids, input_mask, token_labels)

    def create_and_check_codereviewer_model(self, config, input_ids, input_mask, token_labels):
        model = TFCodeReviewerModel(config=config)
        inputs = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        result = model(inputs)

        result = model(input_ids, decoder_attention_mask=input_mask, decoder_input_ids=input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state
        self.parent.assertListEqual(list(encoder_output.shape), [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertListEqual(list(decoder_output.shape), [self.batch_size, self.seq_length, self.hidden_size])
        # There should be `num_layers` key value embeddings stored in decoder_past[1]
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past[1] tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_codereviewer_with_lm_head(self, config, input_ids, input_mask, token_labels):
        model = TFCodeReviewerForConditionalGeneration(config=config)
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }

        result = model(inputs_dict)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_codereviewer_decoder_model_past(self, config, input_ids, decoder_input_ids, attention_mask):
        model = TFCodeReviewerModel(config=config).get_decoder()

        input_ids = input_ids[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, use_cache=True)

        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)[0]
        output_from_past = model(next_tokens, past_key_values=outputs.past_key_values)[0]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_codereviewer_decoder_model_attention_mask_past(
        self, config, input_ids, decoder_input_ids, attention_mask
    ):
        model = TFCodeReviewerModel(config=config).get_decoder()

        # create attention mask
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

        # first forward pass
        outputs = model(input_ids, attention_mask=attn_mask, use_cache=True)

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
        attn_mask = tf.concat(
            [attn_mask, tf.ones((attn_mask.shape[0], 1), dtype=tf.int32)],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)[0]
        output_from_past = model(next_tokens, past_key_values=outputs.past_key_values, attention_mask=attn_mask)[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).numpy().item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_codereviewer_decoder_model_past_large_inputs(
        self, config, input_ids, decoder_input_ids, attention_mask
    ):
        model = TFCodeReviewerModel(config=config).get_decoder()

        input_ids = input_ids[:1, :]
        attention_mask = attention_mask[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)[0]
        output_from_past = model(
            next_tokens, attention_mask=next_attention_mask, past_key_values=outputs.past_key_values
        )[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, token_labels) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return config, inputs_dict


@require_tf
class TFCodeReviewerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    is_encoder_decoder = True
    all_model_classes = (TFCodeReviewerModel, TFCodeReviewerForConditionalGeneration) if is_tf_available() else ()
    all_generative_model_classes = (TFCodeReviewerForConditionalGeneration,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {
            "text2text-generation": TFCodeReviewerForConditionalGeneration,
            "question-answering": TFCodeReviewerForQuestionAnswering,
            "text-classification": TFCodeReviewerForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCodeReviewerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CodeReviewerConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_codereviewer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codereviewer_model(*config_and_inputs)

    def test_codereviewer_model_v1_1(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config.tie_word_embeddings = False
        config.feed_forward_proj = "gated-gelu"
        self.model_tester.create_and_check_codereviewer_model(config, *config_and_inputs[1:])

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codereviewer_with_lm_head(*config_and_inputs)

    def test_codereviewer_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codereviewer_decoder_model_past(*config_and_inputs)

    def test_codereviewer_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codereviewer_decoder_model_attention_mask_past(*config_and_inputs)

    def test_codereviewer_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()

        # `create_and_check_codereviewer_decoder_model_past_large_inputs` has special inputs:
        #     (config, input_ids, decoder_input_ids, attention_mask)
        # and we have to prepare it correctly here.
        config, input_ids, input_mask, token_labels = config_and_inputs
        config_and_inputs = (config, input_ids, None, input_mask)

        self.model_tester.create_and_check_codereviewer_decoder_model_past_large_inputs(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = TFCodeReviewerModel.from_pretrained("microsoft/codereviewer")
        self.assertIsNotNone(model)

    def test_generate_with_headmasking(self):
        # TODO: Fix head-masking according to PyTorch CodeReviewer model
        pass

    # This test is run in `TFCodeReviewerEncoderOnlyModelTest`, where the main layer has the same inputs as the model
    @unittest.skip(reason="The inputs of the Main Layer are different.")
    def test_keras_save_load(self):
        pass

    @unittest.skip("Does not support conversations.")
    def test_pipeline_conversational(self):
        pass


class TFCodeReviewerEncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = CodeReviewerConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = TFCodeReviewerEncoderModel(config=config)
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class TFCodeReviewerEncoderOnlyModelTest(TFModelTesterMixin, unittest.TestCase):
    is_encoder_decoder = False
    all_model_classes = (TFCodeReviewerEncoderModel,) if is_tf_available() else ()
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCodeReviewerEncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CodeReviewerConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # is not able to be part of a pipeline
    def test_train_pipeline_custom_model(self):
        pass
