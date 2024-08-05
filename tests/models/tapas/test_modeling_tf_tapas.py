# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import copy
import unittest

import numpy as np
import pandas as pd

from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TapasConfig,
    TapasTokenizer,
    is_tf_available,
)
from transformers.models.auto import get_values
from transformers.testing_utils import require_tensorflow_probability, require_tf, slow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFTapasForMaskedLM,
        TFTapasForQuestionAnswering,
        TFTapasForSequenceClassification,
        TFTapasModel,
    )
    from transformers.models.tapas.modeling_tf_tapas import (
        IndexMap,
        ProductIndexMap,
        flatten,
        gather,
        range_index_map,
        reduce_max,
        reduce_mean,
        reduce_sum,
    )


class TFTapasModelTester:
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
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=512,
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],
        type_sequence_label_size=2,
        positive_weight=10.0,
        num_aggregation_labels=4,
        num_labels=2,
        aggregation_loss_importance=0.8,
        use_answer_as_supervision=True,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function="ratio",
        cell_selection_preference=0.5,
        answer_loss_cutoff=100,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=True,
        reset_position_index_per_cell=True,
        disable_per_token_loss=False,
        scope=None,
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
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_sizes = type_vocab_sizes
        self.type_sequence_label_size = type_sequence_label_size
        self.positive_weight = positive_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.num_labels = num_labels
        self.aggregation_loss_importance = aggregation_loss_importance
        self.use_answer_as_supervision = use_answer_as_supervision
        self.answer_loss_importance = answer_loss_importance
        self.use_normalized_answer_loss = use_normalized_answer_loss
        self.huber_loss_delta = huber_loss_delta
        self.temperature = temperature
        self.agg_temperature = agg_temperature
        self.use_gumbel_for_cells = use_gumbel_for_cells
        self.use_gumbel_for_agg = use_gumbel_for_agg
        self.average_approximation_function = average_approximation_function
        self.cell_selection_preference = cell_selection_preference
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = []
        for type_vocab_size in self.type_vocab_sizes:
            token_type_ids.append(ids_tensor(shape=[self.batch_size, self.seq_length], vocab_size=type_vocab_size))
        token_type_ids = tf.stack(token_type_ids, axis=2)

        sequence_labels = None
        token_labels = None
        labels = None
        numeric_values = None
        numeric_values_scale = None
        float_answer = None
        aggregation_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            labels = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
            numeric_values = ids_tensor([self.batch_size, self.seq_length], vocab_size=2, dtype=tf.float32)
            numeric_values_scale = ids_tensor([self.batch_size, self.seq_length], vocab_size=2, dtype=tf.float32)
            float_answer = ids_tensor([self.batch_size], vocab_size=2, dtype=tf.float32)
            aggregation_labels = ids_tensor([self.batch_size], self.num_aggregation_labels)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            labels,
            numeric_values,
            numeric_values_scale,
            float_answer,
            aggregation_labels,
        )

    def get_config(self):
        return TapasConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_sizes=self.type_vocab_sizes,
            initializer_range=self.initializer_range,
            positive_weight=self.positive_weight,
            num_aggregation_labels=self.num_aggregation_labels,
            num_labels=self.num_labels,
            aggregation_loss_importance=self.aggregation_loss_importance,
            use_answer_as_supervision=self.use_answer_as_supervision,
            answer_loss_importance=self.answer_loss_importance,
            use_normalized_answer_loss=self.use_normalized_answer_loss,
            huber_loss_delta=self.huber_loss_delta,
            temperature=self.temperature,
            agg_temperature=self.agg_temperature,
            use_gumbel_for_cells=self.use_gumbel_for_cells,
            use_gumbel_for_agg=self.use_gumbel_for_agg,
            average_approximation_function=self.average_approximation_function,
            cell_selection_preference=self.cell_selection_preference,
            answer_loss_cutoff=self.answer_loss_cutoff,
            max_num_rows=self.max_num_rows,
            max_num_columns=self.max_num_columns,
            average_logits_per_cell=self.average_logits_per_cell,
            select_one_column=self.select_one_column,
            allow_empty_column_selection=self.allow_empty_column_selection,
            init_cell_selection_weights_to_zero=self.init_cell_selection_weights_to_zero,
            reset_position_index_per_cell=self.reset_position_index_per_cell,
            disable_per_token_loss=self.disable_per_token_loss,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        model = TFTapasModel(config=config)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        inputs.pop("attention_mask")
        result = model(inputs)
        inputs.pop("token_type_ids")
        result = model(inputs)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        model = TFTapasForMaskedLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        config.num_labels = self.num_labels
        model = TFTapasForSequenceClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": sequence_labels,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        labels,
        numeric_values,
        numeric_values_scale,
        float_answer,
        aggregation_labels,
    ):
        # inference: without aggregation head (SQA). Model only returns logits
        sqa_config = copy.copy(config)
        sqa_config.num_aggregation_labels = 0
        sqa_config.use_answer_as_supervision = False
        model = TFTapasForQuestionAnswering(config=sqa_config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

        # inference: with aggregation head (WTQ, WikiSQL-supervised). Model returns logits and aggregation logits
        model = TFTapasForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

        # training: can happen in 3 main ways
        # case 1: conversational (SQA)
        model = TFTapasForQuestionAnswering(config=sqa_config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        result = model(inputs)
        self.parent.assertEqual(result.loss.shape, (1,))
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

        # case 2: weak supervision for aggregation (WTQ)
        model = TFTapasForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "numeric_values": numeric_values,
            "numeric_values_scale": numeric_values_scale,
            "float_answer": float_answer,
        }
        result = model(inputs)
        self.parent.assertEqual(result.loss.shape, (1,))
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

        # case 3: strong supervision for aggregation (WikiSQL-supervised)
        wikisql_config = copy.copy(config)
        wikisql_config.use_answer_as_supervision = False
        model = TFTapasForQuestionAnswering(config=wikisql_config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "aggregation_labels": aggregation_labels,
        }
        result = model(inputs)
        self.parent.assertEqual(result.loss.shape, (1,))
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            labels,
            numeric_values,
            numeric_values_scale,
            float_answer,
            aggregation_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tensorflow_probability
@require_tf
class TFTapasModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFTapasModel,
            TFTapasForMaskedLM,
            TFTapasForSequenceClassification,
            TFTapasForQuestionAnswering,
        )
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TFTapasModel,
            "fill-mask": TFTapasForMaskedLM,
            "text-classification": TFTapasForSequenceClassification,
            "zero-shot": TFTapasForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False) -> dict:
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: tf.tile(tf.expand_dims(v, 1), (1, self.model_tester.num_choices) + (1,) * (v.ndim - 1))
                if isinstance(v, tf.Tensor) and v.ndim > 0
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = tf.ones(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING):
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )
                inputs_dict["aggregation_labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
                inputs_dict["numeric_values"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.float32
                )
                inputs_dict["numeric_values_scale"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.float32
                )
                inputs_dict["float_answer"] = tf.zeros(self.model_tester.batch_size, dtype=tf.float32)
            elif model_class in get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING):
                inputs_dict["next_sentence_label"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(TF_MODEL_FOR_PRETRAINING_MAPPING),
                *get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = TFTapasModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TapasConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    @unittest.skip(reason="The default test gets NaN losses with the test-generated inputs")
    def test_dataset_conversion(self):
        pass

    @unittest.skip(reason="The default test gets NaN losses with the test-generated inputs")
    def test_keras_fit(self):
        pass

    @unittest.skip(reason="The default test gets NaN losses with the test-generated inputs")
    def test_loss_computation(self):
        pass

    @unittest.skip("tfp is not defined even if installed. FIXME @Arthur in a followup PR!")
    def test_pt_tf_model_equivalence(self):
        pass


def prepare_tapas_single_inputs_for_inference():
    # Here we prepare a single table-question pair to test TAPAS inference on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
    }
    queries = "Which footballer is 33 years old?"
    table = pd.DataFrame.from_dict(data)

    return table, queries


def prepare_tapas_batch_inputs_for_inference():
    # Here we prepare a batch of 2 table-question pairs to test TAPAS inference on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
        "Number of goals": ["712", "750"],
    }
    queries = ["Which footballer is 33 years old?", "How many goals does Ronaldo have?"]
    table = pd.DataFrame.from_dict(data)

    return table, queries


def prepare_tapas_batch_inputs_for_training():
    # Here we prepare a DIFFERENT batch of 2 table-question pairs to test TAPAS training on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
        "Number of goals": ["712", "750"],
    }
    queries = ["Which footballer is 33 years old?", "What's the total number of goals?"]
    table = pd.DataFrame.from_dict(data)

    answer_coordinates = [[(0, 0)], [(0, 2), (1, 2)]]
    answer_text = [["Lionel Messi"], ["1462"]]
    float_answer = [float("NaN"), float("1462")]

    return table, queries, answer_coordinates, answer_text, float_answer


@require_tensorflow_probability
@require_tf
class TFTapasModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

    @slow
    def test_inference_no_head(self):
        # ideally we want to test this with the weights of tapas_inter_masklm_base_reset,
        # but since it's not straightforward to do this with the TF 1 implementation, we test it with
        # the weights of the WTQ base model (i.e. tapas_wtq_wikisql_sqa_inter_masklm_base_reset)
        model = TFTapasModel.from_pretrained("google/tapas-base-finetuned-wtq")
        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="tf")
        outputs = model(**inputs)

        # test the sequence output
        expected_slice = tf.constant(
            [
                [
                    [-0.141581565, -0.599805772, 0.747186482],
                    [-0.143664181, -0.602008104, 0.749218345],
                    [-0.15169853, -0.603363097, 0.741370678],
                ]
            ]
        )
        tf.debugging.assert_near(outputs.last_hidden_state[:, :3, :3], expected_slice, atol=0.0005)

        # test the pooled output
        expected_slice = tf.constant([[0.987518311, -0.970520139, -0.994303405]])

        tf.debugging.assert_near(outputs.pooler_output[:, :3], expected_slice, atol=0.0005)

    @unittest.skip(reason="Model not available yet")
    def test_inference_masked_lm(self):
        pass

    # TapasForQuestionAnswering has 3 possible ways of being fine-tuned:
    # - conversational set-up (SQA)
    # - weak supervision for aggregation (WTQ, WikiSQL)
    # - strong supervision for aggregation (WikiSQL-supervised)
    # We test all of them:
    @slow
    def test_inference_question_answering_head_conversational(self):
        # note that google/tapas-base-finetuned-sqa should correspond to tapas_sqa_inter_masklm_base_reset
        model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-sqa")
        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="tf")
        outputs = model(**inputs)

        # test the logits
        logits = outputs.logits
        expected_shape = tf.TensorShape([1, 21])
        tf.debugging.assert_equal(logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -9997.274,
                    -16.262585,
                    -10004.089,
                    15.435196,
                    15.435196,
                    15.435196,
                    -9990.443,
                    -16.327433,
                    -16.327433,
                    -16.327433,
                    -16.327433,
                    -16.327433,
                    -10004.84,
                ]
            ]
        )

        tf.debugging.assert_near(logits, expected_slice, atol=0.015)

    @slow
    def test_inference_question_answering_head_conversational_absolute_embeddings(self):
        # note that google/tapas-small-finetuned-sqa should correspond to tapas_sqa_inter_masklm_small_reset
        # however here we test the version with absolute position embeddings
        model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-small-finetuned-sqa")
        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="tf")
        outputs = model(**inputs)

        # test the logits
        logits = outputs.logits
        expected_shape = tf.TensorShape([1, 21])
        tf.debugging.assert_equal(logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -10000.041,
                    -18.369339,
                    -10014.692,
                    17.730324,
                    17.730324,
                    17.730324,
                    -9984.974,
                    -18.322773,
                    -18.322773,
                    -18.322773,
                    -18.322773,
                    -18.322773,
                    -10007.267,
                ]
            ]
        )

        tf.debugging.assert_near(logits, expected_slice, atol=0.01)

    @slow
    def test_inference_question_answering_head_weak_supervision(self):
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        tokenizer = self.default_tokenizer
        # let's test on a batch
        table, queries = prepare_tapas_batch_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, padding="longest", return_tensors="tf")
        outputs = model(**inputs)

        # test the logits
        logits = outputs.logits
        expected_shape = tf.TensorShape([2, 28])
        tf.debugging.assert_equal(logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [-160.375504, -160.375504, -160.375504, -10072.3965, -10070.9414, -10094.9736],
                [-9861.6123, -9861.6123, -9861.6123, -9861.6123, -9891.01172, 146.600677],
            ]
        )

        tf.debugging.assert_near(logits[:, -6:], expected_slice, atol=0.4)

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = tf.TensorShape([2, 4])
        tf.debugging.assert_equal(logits_aggregation.shape, expected_shape)
        expected_tensor = tf.constant(
            [[18.8545208, -9.76614857, -6.3128891, -2.93525243], [-4.05782509, 40.0351, -5.35329962, 23.3978653]]
        )
        tf.debugging.assert_near(logits_aggregation, expected_tensor, atol=0.001)

        # test the predicted answer coordinates and aggregation indices
        EXPECTED_PREDICTED_ANSWER_COORDINATES = [[(0, 0)], [(1, 2)]]
        EXPECTED_PREDICTED_AGGREGATION_INDICES = [0, 1]

        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits, outputs.logits_aggregation
        )
        tf.debugging.assert_equal(EXPECTED_PREDICTED_ANSWER_COORDINATES, predicted_answer_coordinates)
        tf.debugging.assert_equal(EXPECTED_PREDICTED_AGGREGATION_INDICES, predicted_aggregation_indices)

    @slow
    def test_training_question_answering_head_weak_supervision(self):
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
        tokenizer = self.default_tokenizer
        # let's test on a batch
        table, queries, answer_coordinates, answer_text, float_answer = prepare_tapas_batch_inputs_for_training()
        inputs = tokenizer(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            padding="longest",
            return_tensors="tf",
        )
        # the answer should be prepared by the user
        float_answer = tf.constant(float_answer, dtype=tf.float32)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            labels=inputs["labels"],
            numeric_values=inputs["numeric_values"],
            numeric_values_scale=inputs["numeric_values_scale"],
            float_answer=float_answer,
        )

        # test the loss
        loss = outputs.loss
        expected_loss = tf.constant(3.3527612686157227e-08)
        tf.debugging.assert_near(loss, expected_loss, atol=1e-6)

        # test the logits on the first example
        logits = outputs.logits
        expected_shape = tf.TensorShape([2, 29])
        tf.debugging.assert_equal(logits.shape, expected_shape)
        expected_slice = tf.constant(
            [
                -160.0156,
                -160.0156,
                -160.0156,
                -160.0156,
                -160.0156,
                -10072.2266,
                -10070.8896,
                -10092.6006,
                -10092.6006,
            ]
        )
        tf.debugging.assert_near(logits[0, -9:], expected_slice, atol=1e-6)

        # test the aggregation logits on the second example
        logits_aggregation = outputs.logits_aggregation
        expected_shape = tf.TensorShape([2, 4])
        tf.debugging.assert_equal(logits_aggregation.shape, expected_shape)
        expected_tensor = tf.constant([-4.0538, 40.0304, -5.3554, 23.3965])
        tf.debugging.assert_near(logits_aggregation[1, -4:], expected_tensor, atol=1e-4)

    @slow
    def test_inference_question_answering_head_strong_supervision(self):
        # note that google/tapas-base-finetuned-wikisql-supervised should correspond to tapas_wikisql_sqa_inter_masklm_base_reset
        model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")
        tokenizer = self.default_tokenizer

        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="tf")
        outputs = model(**inputs)

        # test the logits
        logits = outputs.logits
        expected_shape = tf.TensorShape([1, 21])
        tf.debugging.assert_equal(logits.shape, expected_shape)
        expected_slice = tf.constant(
            [
                [
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -10011.1084,
                    -18.6185989,
                    -10008.7969,
                    17.6355762,
                    17.6355762,
                    17.6355762,
                    -10002.4404,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -18.7111301,
                    -10007.0977,
                ]
            ]
        )
        tf.debugging.assert_near(logits, expected_slice, atol=0.02)

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = tf.TensorShape([1, 4])
        tf.debugging.assert_equal(logits_aggregation.shape, expected_shape)
        expected_tensor = tf.constant([[16.5659733, -3.06624889, -2.34152961, -0.970244825]])
        tf.debugging.assert_near(logits_aggregation, expected_tensor, atol=0.003)

    @slow
    def test_inference_classification_head(self):
        # note that google/tapas-base-finetuned-tabfact should correspond to tapas_tabfact_inter_masklm_base_reset
        model = TFTapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")
        tokenizer = self.default_tokenizer

        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="tf")
        outputs = model(**inputs)

        # test the classification logits
        logits = outputs.logits
        expected_shape = tf.TensorShape([1, 2])
        tf.debugging.assert_equal(logits.shape, expected_shape)
        expected_slice = tf.constant([[0.795137286, 9.5572]])
        tf.debugging.assert_near(logits, expected_slice, atol=0.05)


# Below: tests for Tapas utilities which are defined in modeling_tf_tapas.py.
# These are based on segmented_tensor_test.py of the original implementation.
# URL: https://github.com/google-research/tapas/blob/master/tapas/models/segmented_tensor_test.py
@require_tensorflow_probability
class TFTapasUtilsTest(unittest.TestCase):
    def _prepare_tables(self):
        """Prepares two tables, both with three distinct rows.
        The first table has two columns:
        1.0, 2.0 | 3.0
        2.0, 0.0 | 1.0
        1.0, 3.0 | 4.0
        The second table has three columns:
        1.0 | 2.0 | 3.0
        2.0 | 0.0 | 1.0
        1.0 | 3.0 | 4.0
        Returns:
        SegmentedTensors with the tables.
        """
        values = tf.constant(
            [
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
            ]
        )
        row_index = IndexMap(
            indices=[
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            ],
            num_segments=3,
            batch_dims=1,
        )
        col_index = IndexMap(
            indices=[
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
            ],
            num_segments=3,
            batch_dims=1,
        )
        return values, row_index, col_index

    def test_product_index(self):
        _, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_index_proj = cell_index.project_outer(cell_index)
        col_index_proj = cell_index.project_inner(cell_index)

        ind = cell_index.indices
        self.assertEqual(cell_index.num_segments, 9)

        # Projections should give back the original indices.
        # we use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(row_index.indices.numpy(), row_index_proj.indices.numpy())
        self.assertEqual(row_index.num_segments, row_index_proj.num_segments)
        self.assertEqual(row_index.batch_dims, row_index_proj.batch_dims)
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(col_index.indices.numpy(), col_index_proj.indices.numpy())
        self.assertEqual(col_index.batch_dims, col_index_proj.batch_dims)

        # The first and second "column" are identified in the first table.
        for i in range(3):
            self.assertEqual(ind[0, i, 0], ind[0, i, 1])
            self.assertNotEqual(ind[0, i, 0], ind[0, i, 2])

        # All rows are distinct in the first table.
        for i, i_2 in zip(range(3), range(3)):
            for j, j_2 in zip(range(3), range(3)):
                if i != i_2 and j != j_2:
                    self.assertNotEqual(ind[0, i, j], ind[0, i_2, j_2])

        # All cells are distinct in the second table.
        for i, i_2 in zip(range(3), range(3)):
            for j, j_2 in zip(range(3), range(3)):
                if i != i_2 or j != j_2:
                    self.assertNotEqual(ind[1, i, j], ind[1, i_2, j_2])

    def test_flatten(self):
        _, row_index, col_index = self._prepare_tables()
        row_index_flat = flatten(row_index)
        col_index_flat = flatten(col_index)

        shape = [3, 4, 5]
        batched_index = IndexMap(indices=tf.zeros(shape, dtype=tf.int32), num_segments=1, batch_dims=3)
        batched_index_flat = flatten(batched_index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(
            row_index_flat.indices.numpy(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        )
        np.testing.assert_array_equal(
            col_index_flat.indices.numpy(), [0, 0, 1, 0, 0, 1, 0, 0, 1, 3, 4, 5, 3, 4, 5, 3, 4, 5]
        )
        self.assertEqual(batched_index_flat.num_segments.numpy(), np.prod(shape))
        np.testing.assert_array_equal(batched_index_flat.indices.numpy(), range(np.prod(shape)))

    def test_range_index_map(self):
        batch_shape = [3, 4]
        num_segments = 5
        index = range_index_map(batch_shape, num_segments)

        self.assertEqual(num_segments, index.num_segments)
        self.assertEqual(2, index.batch_dims)
        indices = index.indices
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(list(indices.shape), [3, 4, 5])
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
                np.testing.assert_array_equal(indices[i, j, :].numpy(), range(num_segments))

    def test_reduce_sum(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_sum, _ = reduce_sum(values, row_index)
        col_sum, _ = reduce_sum(values, col_index)
        cell_sum, _ = reduce_sum(values, cell_index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(row_sum.numpy(), [[6.0, 3.0, 8.0], [6.0, 3.0, 8.0]])
        np.testing.assert_allclose(col_sum.numpy(), [[9.0, 8.0, 0.0], [4.0, 5.0, 8.0]])
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 4.0, 4.0, 0.0], [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0]],
        )

    def test_reduce_mean(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)
        row_mean, _ = reduce_mean(values, row_index)
        col_mean, _ = reduce_mean(values, col_index)
        cell_mean, _ = reduce_mean(values, cell_index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(
            row_mean.numpy(), [[6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0], [6.0 / 3.0, 3.0 / 3.0, 8.0 / 3.0]]
        )
        np.testing.assert_allclose(col_mean.numpy(), [[9.0 / 6.0, 8.0 / 3.0, 0.0], [4.0 / 3.0, 5.0 / 3.0, 8.0 / 3.0]])
        np.testing.assert_allclose(
            cell_mean.numpy(),
            [
                [3.0 / 2.0, 3.0, 0.0, 2.0 / 2.0, 1.0, 0.0, 4.0 / 2.0, 4.0, 0.0],
                [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0],
            ],
        )

    def test_reduce_max(self):
        values = tf.convert_to_tensor([2.0, 1.0, 0.0, 3.0])
        index = IndexMap(indices=tf.convert_to_tensor([0, 1, 0, 1]), num_segments=2)
        maximum, _ = reduce_max(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(maximum.numpy(), [2, 3])

    def test_reduce_sum_vectorized(self):
        values = tf.convert_to_tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        index = IndexMap(indices=tf.convert_to_tensor([0, 0, 1]), num_segments=2, batch_dims=0)
        sums, new_index = reduce_sum(values, index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(sums.numpy(), [[3.0, 5.0, 7.0], [3.0, 4.0, 5.0]])
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(new_index.indices.numpy(), [0, 1])
        np.testing.assert_array_equal(new_index.num_segments.numpy(), 2)
        np.testing.assert_array_equal(new_index.batch_dims, 0)

    def test_gather(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = ProductIndexMap(row_index, col_index)

        # Compute sums and then gather. The result should have the same shape as
        # the original table and each element should contain the sum the values in
        # its cell.
        sums, _ = reduce_sum(values, cell_index)
        cell_sum = gather(sums, cell_index)
        assert cell_sum.shape == values.shape

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[[3.0, 3.0, 3.0], [2.0, 2.0, 1.0], [4.0, 4.0, 4.0]], [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]]],
        )

    def test_gather_vectorized(self):
        values = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        index = IndexMap(indices=tf.convert_to_tensor([[0, 1], [1, 0]]), num_segments=2, batch_dims=1)
        result = gather(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(result.numpy(), [[[1, 2], [3, 4]], [[7, 8], [5, 6]]])
