# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
        TapasConfig,
        TapasForMaskedLM,
        TapasForQuestionAnswering,
        TapasForSequenceClassification,
        TapasModel,
    )


class TapasModelTester:
    """You can also import this e.g from .test_modeling_tapas import TapasModelTester """

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
        cell_select_pref=0.5,
        answer_loss_cutoff=100,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=False,
        reset_position_index_per_cell=False,
        disable_per_token_loss=False,
        span_prediction="none",
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
        self.cell_select_pref = cell_select_pref
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.span_prediction = span_prediction
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = []
        for type_vocab_size in self.type_vocab_sizes:
            token_type_ids.append(ids_tensor(shape=[self.batch_size, self.seq_length], vocab_size=type_vocab_size))
        token_type_ids = torch.stack(token_type_ids, dim=2)

        sequence_labels = None
        token_labels = None
        label_ids = None
        answer = None
        numeric_values = None
        numeric_values_scale = None
        aggregation_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            label_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
            answer = floats_tensor([self.batch_size])
            numeric_values = floats_tensor([self.batch_size, self.seq_length])
            numeric_values_scale = floats_tensor([self.batch_size, self.seq_length])
            aggregation_labels = ids_tensor([self.batch_size], self.num_aggregation_labels)

        config = TapasConfig(
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
            cell_select_pref=self.cell_select_pref,
            answer_loss_cutoff=self.answer_loss_cutoff,
            max_num_rows=self.max_num_rows,
            max_num_columns=self.max_num_columns,
            average_logits_per_cell=self.average_logits_per_cell,
            select_one_column=self.select_one_column,
            allow_empty_column_selection=self.allow_empty_column_selection,
            init_cell_selection_weights_to_zero=self.init_cell_selection_weights_to_zero,
            reset_position_index_per_cell=self.reset_position_index_per_cell,
            disable_per_token_loss=self.disable_per_token_loss,
            span_prediction=self.span_prediction,
            return_dict=True,
        )

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            label_ids,
            answer,
            numeric_values,
            numeric_values_scale,
            aggregation_labels,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        label_ids,
        answer,
        numeric_values,
        numeric_values_scale,
        aggregation_labels,
    ):
        model = TapasModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
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
        label_ids,
        answer,
        numeric_values,
        numeric_values_scale,
        aggregation_labels,
    ):
        model = TapasForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        label_ids,
        answer,
        numeric_values,
        numeric_values_scale,
        aggregation_labels,
    ):
        model = TapasForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
            answer=answer,
            numeric_values=numeric_values,
            numeric_values_scale=numeric_values_scale,
            aggregation_labels=aggregation_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.logits_aggregation.shape, (self.batch_size, self.num_aggregation_labels))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        token_type_ids,
        sequence_labels,
        token_labels,
        label_ids,
        answer,
        numeric_values,
        numeric_values_scale,
        aggregation_labels,
    ):
        config.num_labels = self.num_labels
        model = TapasForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            label_ids,
            answer,
            numeric_values,
            numeric_values_scale,
            aggregation_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class TapasModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TapasModel,
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
        )
        if is_torch_available()
        else None
    )
    test_pruning = False
    test_torchscript = True
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = TapasModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TapasConfig, dim=37)

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

    # @slow
    # def test_lm_outputs_same_as_reference_model(self):
    #     """Write something that could help someone fixing this here."""
    #     checkpoint_path = "XXX/bart-large"
    #     model = self.big_model
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         checkpoint_path
    #     )  # same with AutoTokenizer (see tokenization_auto.py). This is not mandatory
    #     # MODIFY THIS DEPENDING ON YOUR MODELS RELEVANT TASK.
    #     batch = tokenizer(["I went to the <mask> yesterday"]).to(torch_device)
    #     desired_mask_result = tokenizer.decode("store")  # update this
    #     logits = model(**batch).logits
    #     masked_index = (batch.input_ids == self.tokenizer.mask_token_id).nonzero()
    #     assert model.num_parameters() == 175e9  # a joke
    #     mask_entry_logits = logits[0, masked_index.item(), :]
    #     probs = mask_entry_logits.softmax(dim=0)
    #     _, predictions = probs.topk(1)
    #     self.assertEqual(tokenizer.decode(predictions), desired_mask_result)

    # @cached_property
    # def big_model(self):
    #     """Cached property means this code will only be executed once."""
    #     checkpoint_path = "XXX/bart-large"
    #     model = AutoModelForMaskedLM.from_pretrained(checkpoint_path).to(
    #         torch_device
    #     )  # test whether AutoModel can determine your model_class from checkpoint name
    #     if torch_device == "cuda":
    #         model.half()

    # optional: do more testing! This will save you time later!
    # @slow
    # def test_that_XXX_can_be_used_in_a_pipeline(self):
    #     """We can use self.big_model here without calling __init__ again."""
    #     pass

    # def test_XXX_loss_doesnt_change_if_you_add_padding(self):
    #     pass

    # def test_XXX_bad_args(self):
    #     pass

    # def test_XXX_backward_pass_reduces_loss(self):
    #     """Test loss/gradients same as reference implementation, for example."""
    #     pass

    # @require_torch_and_cuda
    # def test_large_inputs_in_fp16_dont_cause_overflow(self):
    #     pass


# Below: tests for Tapas utilities, based on segmented_tensor_test.py of the original implementation.
# These test the operations on segmented tensors.
class TapasUtilitiesTest(unittest.TestCase):
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
        values = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
                [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]],
            ]
        )
        row_index = utils.IndexMap(
            indices=torch.tensor(
                [
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                ]
            ),
            num_segments=3,
            batch_dims=1,
        )
        col_index = utils.IndexMap(
            indices=torch.tensor(
                [
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                ]
            ),
            num_segments=3,
            batch_dims=1,
        )
        return values, row_index, col_index

    def test_product_index(self):
        _, row_index, col_index = self._prepare_tables()
        cell_index = utils.ProductIndexMap(row_index, col_index)
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
        row_index_flat = utils.flatten(row_index)
        col_index_flat = utils.flatten(col_index)

        shape = [3, 4, 5]
        batched_index = utils.IndexMap(indices=torch.zeros(shape).type(torch.LongTensor), num_segments=1, batch_dims=3)
        batched_index_flat = utils.flatten(batched_index)

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
        index = utils.range_index_map(batch_shape, num_segments)

        self.assertEqual(num_segments, index.num_segments)
        self.assertEqual(2, index.batch_dims)
        indices = index.indices
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(list(indices.size()), [3, 4, 5])
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
                np.testing.assert_array_equal(indices[i, j, :].numpy(), range(num_segments))

    def test_reduce_sum(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = utils.ProductIndexMap(row_index, col_index)
        row_sum, _ = utils.reduce_sum(values, row_index)
        col_sum, _ = utils.reduce_sum(values, col_index)
        cell_sum, _ = utils.reduce_sum(values, cell_index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(row_sum.numpy(), [[6.0, 3.0, 8.0], [6.0, 3.0, 8.0]])
        np.testing.assert_allclose(col_sum.numpy(), [[9.0, 8.0, 0.0], [4.0, 5.0, 8.0]])
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 4.0, 4.0, 0.0], [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0]],
        )

    def test_reduce_mean(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = utils.ProductIndexMap(row_index, col_index)
        row_mean, _ = utils.reduce_mean(values, row_index)
        col_mean, _ = utils.reduce_mean(values, col_index)
        cell_mean, _ = utils.reduce_mean(values, cell_index)

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
        values = torch.as_tensor([2.0, 1.0, 0.0, 3.0])
        index = utils.IndexMap(indices=torch.as_tensor([0, 1, 0, 1]), num_segments=2)
        maximum, _ = utils.reduce_max(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(maximum.numpy(), [2, 3])

    def test_reduce_sum_vectorized(self):
        values = torch.as_tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        index = utils.IndexMap(indices=torch.as_tensor([0, 0, 1]), num_segments=2, batch_dims=0)
        sums, new_index = utils.reduce_sum(values, index)

        # We use np.testing.assert_allclose rather than Tensorflow's assertAllClose
        np.testing.assert_allclose(sums.numpy(), [[3.0, 5.0, 7.0], [3.0, 4.0, 5.0]])
        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(new_index.indices.numpy(), [0, 1])
        np.testing.assert_array_equal(new_index.num_segments.numpy(), 2)
        np.testing.assert_array_equal(new_index.batch_dims, 0)

    def test_gather(self):
        values, row_index, col_index = self._prepare_tables()
        cell_index = utils.ProductIndexMap(row_index, col_index)

        # Compute sums and then gather. The result should have the same shape as
        # the original table and each element should contain the sum the values in
        # its cell.
        sums, _ = utils.reduce_sum(values, cell_index)
        cell_sum = utils.gather(sums, cell_index)
        assert cell_sum.size() == values.size()

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_allclose(
            cell_sum.numpy(),
            [[[3.0, 3.0, 3.0], [2.0, 2.0, 1.0], [4.0, 4.0, 4.0]], [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [1.0, 3.0, 4.0]]],
        )

    def test_gather_vectorized(self):
        values = torch.as_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        index = utils.IndexMap(indices=torch.as_tensor([[0, 1], [1, 0]]), num_segments=2, batch_dims=1)
        result = utils.gather(values, index)

        # We use np.testing.assert_array_equal rather than Tensorflow's assertAllEqual
        np.testing.assert_array_equal(result.numpy(), [[[1, 2], [3, 4]], [[7, 8], [5, 6]]])
