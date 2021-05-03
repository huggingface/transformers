# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import timeout_decorator  # noqa

from transformers import BartConfig, is_flax_available
from transformers.testing_utils import require_flax, slow

from .test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    from transformers.models.bart.modeling_flax_bart import (
        FlaxBartForConditionalGeneration,
        FlaxBartForQuestionAnswering,
        FlaxBartForSequenceClassification,
        FlaxBartModel,
        shift_tokens_right,
    )


def prepare_bart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = np.where(input_ids != config.pad_token_id, 1, 0)
    if decoder_attention_mask is None:
        decoder_attention_mask = np.where(decoder_input_ids != config.pad_token_id, 1, 0)
    if head_mask is None:
        head_mask = np.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class FlaxBartModelTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        initializer_range=0.02,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        input_ids = np.clip(ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size), 3, self.vocab_size)
        input_ids = np.concatenate((input_ids, 2 * np.ones((self.batch_size, 1), dtype=np.int64)), -1)

        decoder_input_ids = shift_tokens_right(input_ids, 1, 2)

        config = BartConfig(
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
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
        )
        inputs_dict = prepare_bart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_flax
class BartHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        input_ids = np.array(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ],
            dtype=np.int64,
        )

        batch_size = input_ids.shape[0]
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        return config, input_ids, batch_size

    def test_sequence_classification_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        model = FlaxBartForSequenceClassification(config)
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
        expected_shape = (batch_size, config.num_labels)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_question_answering_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        model = FlaxBartForQuestionAnswering(config)
        outputs = model(input_ids=input_ids)

        self.assertEqual(outputs["start_logits"].shape, input_ids.shape)
        self.assertEqual(outputs["end_logits"].shape, input_ids.shape)

    # @timeout_decorator.timeout(1)  # not working with the decorator so far
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_model = FlaxBartForConditionalGeneration(config)
        outputs = lm_model(input_ids=input_ids)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_lm_uneven_forward(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=14,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=48,
        )
        lm_model = FlaxBartForConditionalGeneration(config)
        context = np.array([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], dtype=np.int64)
        summary = np.array([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], dtype=np.int64)
        outputs = lm_model(input_ids=context, decoder_input_ids=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_shift_tokens_right(self):
        input_ids = np.array([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]], dtype=np.int64)
        shifted = shift_tokens_right(input_ids, 1, 2)
        n_pad_before = np.equal(input_ids, 1).astype(np.float32).sum()
        n_pad_after = np.equal(shifted, 1).astype(np.float32).sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(np.equal(shifted[:, 0], 2).all())


@require_flax
class FlaxBartModelTest(FlaxModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FlaxBartModel,
            FlaxBartForConditionalGeneration,
            FlaxBartForSequenceClassification,
            FlaxBartForQuestionAnswering,
        )
        if is_flax_available()
        else ()
    )
    test_head_masking = True

    def setUp(self):
        self.model_tester = FlaxBartModelTester(self)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("facebook/bart-base", from_pt=True)
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)
