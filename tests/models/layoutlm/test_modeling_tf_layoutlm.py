# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors, The Hugging Face Team.
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

from transformers import LayoutLMConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf

    from transformers.models.layoutlm.modeling_tf_layoutlm import (
        TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLayoutLMForMaskedLM,
        TFLayoutLMForQuestionAnswering,
        TFLayoutLMForSequenceClassification,
        TFLayoutLMForTokenClassification,
        TFLayoutLMModel,
    )


class TFLayoutLMModelTester:
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
        range_bbox=1000,
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
        self.range_bbox = range_bbox

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # convert bbox to numpy since TF does not support item assignment
        bbox = ids_tensor([self.batch_size, self.seq_length, 4], self.range_bbox).numpy()
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    t = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = t
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    t = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = t
        bbox = tf.convert_to_tensor(bbox)

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

        config = LayoutLMConfig(
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
        )

        return config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLayoutLMModel(config=config)

        result = model(input_ids, bbox, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, bbox, token_type_ids=token_type_ids)
        result = model(input_ids, bbox)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLayoutLMForMaskedLM(config=config)

        result = model(input_ids, bbox, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFLayoutLMForSequenceClassification(config=config)

        result = model(input_ids, bbox, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFLayoutLMForTokenClassification(config=config)

        result = model(input_ids, bbox, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFLayoutLMForQuestionAnswering(config=config)

        result = model(input_ids, bbox, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_tf
class TFLayoutLMModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFLayoutLMModel,
            TFLayoutLMForMaskedLM,
            TFLayoutLMForTokenClassification,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForQuestionAnswering,
        )
        if is_tf_available()
        else ()
    )
    test_head_masking = False
    test_onnx = True
    onnx_min_opset = 10

    def setUp(self):
        self.model_tester = TFLayoutLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFLayoutLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    # TODO (Joao): fix me
    @unittest.skip("Onnx compliancy broke with TF 2.10")
    def test_onnx_compliancy(self):
        pass


def prepare_layoutlm_batch_inputs():
    # Here we prepare a batch of 2 sequences to test a LayoutLM forward pass on:
    # fmt: off
    input_ids = tf.convert_to_tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]])  # noqa: E231
    attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],])  # noqa: E231
    bbox = tf.convert_to_tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]])  # noqa: E231
    token_type_ids = tf.convert_to_tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])  # noqa: E231
    # these are sequence labels (i.e. at the token level)
    labels = tf.convert_to_tensor([[-100,10,10,10,9,1,-100,7,7,-100,7,7,4,2,5,2,8,8,-100,-100,5,0,3,2,-100],[-100,12,12,12,-100,12,10,-100,-100,-100,-100,10,12,9,-100,-100,-100,10,10,10,9,12,-100,10,-100]])  # noqa: E231
    # fmt: on

    return input_ids, attention_mask, bbox, token_type_ids, labels


@require_tf
class TFLayoutLMModelIntegrationTest(unittest.TestCase):
    @slow
    def test_forward_pass_no_head(self):
        model = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

        input_ids, attention_mask, bbox, token_type_ids, labels = prepare_layoutlm_batch_inputs()

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # test the sequence output on [0, :3, :3]
        expected_slice = tf.convert_to_tensor(
            [[0.1785, -0.1947, -0.0425], [-0.3254, -0.2807, 0.2553], [-0.5391, -0.3322, 0.3364]],
        )

        self.assertTrue(np.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-3))

        # test the pooled output on [1, :3]
        expected_slice = tf.convert_to_tensor([-0.6580, -0.0214, 0.8552])

        self.assertTrue(np.allclose(outputs.pooler_output[1, :3], expected_slice, atol=1e-3))

    @slow
    def test_forward_pass_sequence_classification(self):
        # initialize model with randomly initialized sequence classification head
        model = TFLayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=2)

        input_ids, attention_mask, bbox, token_type_ids, _ = prepare_layoutlm_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=tf.convert_to_tensor([1, 1]),
        )

        # test whether we get a loss as a scalar
        loss = outputs.loss
        expected_shape = (2,)
        self.assertEqual(loss.shape, expected_shape)

        # test the shape of the logits
        logits = outputs.logits
        expected_shape = (2, 2)
        self.assertEqual(logits.shape, expected_shape)

    @slow
    def test_forward_pass_token_classification(self):
        # initialize model with randomly initialized token classification head
        model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=13)

        input_ids, attention_mask, bbox, token_type_ids, labels = prepare_layoutlm_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
        )

        # test the shape of the logits
        logits = outputs.logits
        expected_shape = tf.convert_to_tensor((2, 25, 13))
        self.assertEqual(logits.shape, expected_shape)

    @slow
    def test_forward_pass_question_answering(self):
        # initialize model with randomly initialized token classification head
        model = TFLayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")

        input_ids, attention_mask, bbox, token_type_ids, labels = prepare_layoutlm_batch_inputs()

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # test the shape of the logits
        expected_shape = tf.convert_to_tensor((2, 25))
        self.assertEqual(outputs.start_logits.shape, expected_shape)
        self.assertEqual(outputs.end_logits.shape, expected_shape)
