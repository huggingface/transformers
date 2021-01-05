# coding=utf-8
# Copyright 2020 Huggingface
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

import tempfile
import unittest

from transformers import is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import numpy
    import tensorflow as tf

    from transformers import (
        TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertConfig,
        DPRConfig,
        TFDPRContextEncoder,
        TFDPRQuestionEncoder,
        TFDPRReader,
    )


class TFDPRModelTester:
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
        projection_dim=0,
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
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.projection_dim = projection_dim

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor(
                [self.batch_size, self.seq_length], vocab_size=2
            )  # follow test_modeling_tf_ctrl.py

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

        config = BertConfig(
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
        config = DPRConfig(projection_dim=self.projection_dim, **config.to_dict())

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_dpr_context_encoder(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDPRContextEncoder(config=config)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.projection_dim or self.hidden_size))

    def create_and_check_dpr_question_encoder(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDPRQuestionEncoder(config=config)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.projection_dim or self.hidden_size))

    def create_and_check_dpr_reader(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDPRReader(config=config)
        result = model(input_ids, attention_mask=input_mask)

        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.relevance_logits.shape, (self.batch_size,))

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
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_tf
class TFDPRModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TFDPRContextEncoder,
            TFDPRQuestionEncoder,
            TFDPRReader,
        )
        if is_tf_available()
        else ()
    )

    test_resize_embeddings = False
    test_missing_keys = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = TFDPRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DPRConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_dpr_context_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_context_encoder(*config_and_inputs)

    def test_dpr_question_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_question_encoder(*config_and_inputs)

    def test_dpr_reader_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_reader(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFDPRContextEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFDPRContextEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFDPRQuestionEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFDPRReader.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_saved_model_with_attentions_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_attentions = True

        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", self.model_tester.seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            print(model_class)
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            num_out = len(model(class_inputs_dict))
            model._saved_model_inputs_spec = None
            model._set_save_spec(class_inputs_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                tf.saved_model.save(model, tmpdirname)
                model = tf.keras.models.load_model(tmpdirname)
                outputs = model(class_inputs_dict)

                if self.is_encoder_decoder:
                    output = outputs["encoder_attentions"] if isinstance(outputs, dict) else outputs[-1]
                else:
                    output = outputs["attentions"] if isinstance(outputs, dict) else outputs[-1]

                attentions = [t.numpy() for t in output]
                self.assertEqual(len(outputs), num_out)
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )


@require_tf
class TFDPRModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = TFDPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        input_ids = tf.constant(
            [[101, 7592, 1010, 2003, 2026, 3899, 10140, 1029, 102]]
        )  # [CLS] hello, is my dog cute? [SEP]
        output = model(input_ids)[0]  # embedding shape = (1, 768)
        # compare the actual values for a slice.
        expected_slice = tf.constant(
            [
                [
                    0.03236253,
                    0.12753335,
                    0.16818509,
                    0.00279786,
                    0.3896933,
                    0.24264945,
                    0.2178971,
                    -0.02335227,
                    -0.08481959,
                    -0.14324117,
                ]
            ]
        )
        self.assertTrue(numpy.allclose(output[:, :10].numpy(), expected_slice.numpy(), atol=1e-4))
