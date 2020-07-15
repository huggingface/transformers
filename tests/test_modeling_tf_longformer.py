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

from transformers import is_tf_available
from transformers.modeling_tf_longformer import TFLongformerModel, TFLongformerSelfAttention
from transformers.modeling_tf_utils import shape_list
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf
    from transformers import (
        LongformerConfig,
        #        TFLongformerModel,
        #        TFLongformerForMaskedLM,
        #        TFLongformerForSequenceClassification,
        #        TFLongformerForTokenClassification,
        #        TFLongformerForQuestionAnswering,
        #        TFLongformerForMultipleChoice,
    )


class LongformerModelTester:
    def __init__(
        self, parent,
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
        self.attention_window = 4

        # `ModelTesterMixin.test_attention_outputs` is expecting attention tensors to be of size
        # [num_attention_heads, encoder_seq_length, encoder_key_length], but TFLongformerSelfAttention
        # returns attention of shape [num_attention_heads, encoder_seq_length, self.attention_window + 1]
        # because its local attention only attends to `self.attention_window + 1` locations
        self.key_length = self.attention_window + 1

        # because of padding `encoder_seq_length`, is different from `seq_length`. Relevant for
        # the `test_attention_outputs` and `test_hidden_states_output` tests
        self.encoder_seq_length = (
            self.seq_length + (self.attention_window - self.seq_length % self.attention_window) % self.attention_window
        )

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

        config = LongformerConfig(
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
            attention_window=self.attention_window,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_attention_mask_determinism(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLongformerModel(config=config)
        model.to(torch_device)
        model.eval()

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output_with_mask = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]
        self.parent.assertTrue(torch.allclose(output_with_mask[0, 0, :5], output_without_mask[0, 0, :5], atol=1e-4))

    def create_and_check_longformer_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_longformer_model_with_global_attention_mask(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        global_attention_mask = input_mask.clone()
        global_attention_mask[:, input_mask.shape[-1] // 2] = 0
        global_attention_mask = global_attention_mask.to(torch_device)

        sequence_output, pooled_output = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output, pooled_output = model(
            input_ids, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask
        )
        sequence_output, pooled_output = model(input_ids, global_attention_mask=global_attention_mask)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_longformer_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLongformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels
        )
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.check_loss_output(result)

    def create_and_check_longformer_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLongformerForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        loss, start_logits, end_logits = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        result = {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        self.parent.assertListEqual(list(result["start_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].size()), [self.batch_size, self.seq_length])
        self.check_loss_output(result)

    def create_and_check_longformer_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFLongformerForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_longformer_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFLongformerForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.seq_length, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_longformer_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = TFLongformerForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        loss, logits = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            global_attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_choices])
        self.check_loss_output(result)

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
        global_attention_mask = torch.zeros_like(input_ids)
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "global_attention_mask": global_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_question_answering(self):
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

        # Replace sep_token_id by some random id
        input_ids[input_ids == config.sep_token_id] = torch.randint(0, config.vocab_size, (1,)).item()
        # Make sure there are exactly three sep_token_id
        input_ids[:, -3:] = config.sep_token_id
        input_mask = torch.ones_like(input_ids)

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels


class TFLongformerModelIntegrationTest(unittest.TestCase):
    def _get_hidden_states(self):
        return tf.convert_to_tensor(
            [
                [
                    [
                        4.98332758e-01,
                        2.69175139e00,
                        -7.08081422e-03,
                        1.04915401e00,
                        -1.83476661e00,
                        7.67220476e-01,
                        2.98580543e-01,
                        2.84803992e-02,
                    ],
                    [
                        -7.58357372e-01,
                        4.20635998e-01,
                        -4.04739919e-02,
                        1.59924145e-01,
                        2.05135748e00,
                        -1.15997978e00,
                        5.37166397e-01,
                        2.62873606e-01,
                    ],
                    [
                        -1.69438001e00,
                        4.17574660e-01,
                        -1.49196962e00,
                        -1.76483717e00,
                        -1.94566312e-01,
                        -1.71183858e00,
                        7.72903565e-01,
                        -1.11557056e00,
                    ],
                    [
                        5.44028163e-01,
                        2.05466114e-01,
                        -3.63045868e-01,
                        2.41865062e-01,
                        3.20348382e-01,
                        -9.05611176e-01,
                        -1.92690727e-01,
                        -1.19917547e00,
                    ],
                ]
            ],
            dtype=tf.float32,
        )

    def test_diagonalize(self):
        hidden_states = self._get_hidden_states()
        hidden_states = tf.reshape(hidden_states, (1, 8, 4))  # set seq length = 8, hidden dim = 4
        chunked_hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)
        window_overlap_size = shape_list(chunked_hidden_states)[2]
        self.assertTrue(window_overlap_size == 4)

        padded_hidden_states = TFLongformerSelfAttention._pad_and_diagonalize(chunked_hidden_states)

        self.assertTrue(
            shape_list(padded_hidden_states)[-1] == shape_list(chunked_hidden_states)[-1] + window_overlap_size - 1
        )

        # first row => [0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000]
        tf.debugging.assert_near(padded_hidden_states[0, 0, 0, :4], chunked_hidden_states[0, 0, 0], rtol=1e-3)
        tf.debugging.assert_near(padded_hidden_states[0, 0, 0, 4:], tf.zeros((3,), dtype=tf.dtypes.float32), rtol=1e-3)

        # last row => [0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629]
        tf.debugging.assert_near(padded_hidden_states[0, 0, -1, 3:], chunked_hidden_states[0, 0, -1], rtol=1e-3)
        tf.debugging.assert_near(
            padded_hidden_states[0, 0, -1, :3], tf.zeros((3,), dtype=tf.dtypes.float32), rtol=1e-3
        )

    def test_pad_and_transpose_last_two_dims(self):
        hidden_states = self._get_hidden_states()
        self.assertTrue(shape_list(hidden_states), [1, 8, 4])

        # pad along seq length dim
        paddings = tf.constant([[0, 0], [0, 1], [0, 0]], dtype=tf.dtypes.int32)

        padded_hidden_states = TFLongformerSelfAttention._pad_and_transpose_last_two_dims(hidden_states, paddings)
        self.assertTrue(shape_list(padded_hidden_states) == [1, 8, 5])

        expected_added_dim = tf.zeros((5,), dtype=tf.dtypes.float32)
        tf.debugging.assert_near(expected_added_dim, padded_hidden_states[0, -1, :], rtol=1e-6)
        tf.debugging.assert_near(
            hidden_states[0, -1, :], tf.reshape(padded_hidden_states, (1, -1))[0, 24:32], rtol=1e-6
        )

    def test_chunk(self):
        hidden_states = self._get_hidden_states()
        batch_size = 1
        seq_length = 8
        hidden_size = 4
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length, hidden_size))

        chunked_hidden_states = TFLongformerSelfAttention._chunk(hidden_states, window_overlap=2)

        # expected slices across chunk and seq length dim
        expected_slice_along_seq_length = tf.convert_to_tensor([0.4983, -0.7584, -1.6944], dtype=tf.dtypes.float32)
        expected_slice_along_chunk = tf.convert_to_tensor([0.4983, -1.8348, -0.7584, 2.0514], dtype=tf.dtypes.float32)

        self.assertTrue(shape_list(chunked_hidden_states) == [1, 3, 4, 4])
        tf.debugging.assert_near(chunked_hidden_states[0, :, 0, 0], expected_slice_along_seq_length, rtol=1e-3)
        tf.debugging.assert_near(chunked_hidden_states[0, 0, :, 0], expected_slice_along_chunk, rtol=1e-3)

    def test_layer_local_attn(self):
        model = TFLongformerModel.from_pretrained("patrickvonplaten/longformer-random-tiny")
        layer = model.longformer.encoder.layer[0].attention.self_attention
        hidden_states = self._get_hidden_states()
        batch_size, seq_length, hidden_size = hidden_states.shape

        attention_mask = tf.zeros((batch_size, 1, 1, seq_length), dtype=tf.dtypes.float32)
        attention_mask = tf.where(tf.range(4)[None, None, None, :] > 1, -10000.0, attention_mask)

        output_hidden_states = layer([hidden_states, attention_mask, None])[0]

        expected_slice = tf.convert_to_tensor(
            [0.0019, 0.0122, -0.0171, -0.0256, -0.0300, 0.0173, -0.0115, 0.0048], dtype=tf.dtypes.float32
        )

        self.assertTrue(output_hidden_states.shape, (1, 4, 8))
        tf.debugging.assert_near(output_hidden_states[0, 1], expected_slice, rtol=1e-3)

    def _test_layer_global_attn(self):
        model = TFLongformerModel.from_pretrained("patrickvonplaten/longformer-random-tiny")
        layer = model.encoder.layer[0].attention.self.to(torch_device)
        hidden_states = self._get_hidden_states()
        batch_size, seq_length, hidden_size = hidden_states.size()
        attention_mask = torch.zeros((batch_size, 1, 1, seq_length), dtype=torch.float32, device=torch_device)
        attention_mask[:, :, :, -2:] = 10000
        output_hidden_states = layer(hidden_states, attention_mask)[0]

        self.assertTrue(output_hidden_states.shape, (1, 4, 8))
        self.assertTrue(
            torch.allclose(
                output_hidden_states[0, -1],
                torch.tensor(
                    [-0.0429, -0.0341, 0.0231, -0.0335, -0.0111, -0.0131, -0.0186, -0.0403],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                atol=1e-3,
            )
        )
