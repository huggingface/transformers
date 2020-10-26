# coding=utf-8
# Copyright (...) and the HuggingFace Inc. team.
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
""" Utilities for PyTorch Tapas model. This includes: 
- operations on segmented tensors using the torch_scatter PyTorch package
- functions to calculate the various logits and losses defined by the authors of TAPAS 
(cell selection loss, aggregation loss, regression loss). These make use of these operations.

For the operations on segmented tensors, first 2 classes are defined that actually segment tensors:
IndexMap and ProductIndexMap.
- An IndexMap is typically used to represent the column_ids or row_ids of the various tokens of a batch. 
Each token belongs to a particular segment, i.e., a particular column or row.  
- A ProductIndexMap is typically used to represent the cell of the various tokens of a batch. 
Each token belongs to a particular segment, i.e. a particular cell. 
Next, the operations themselves are defined as separate functions: gather, reduce_sum, reduce_mean,
reduce_max and reduce_min. The functions flatten and range_index_map are used within these functions. 

For the calculations of the custom losses, the following is defined:
- calculation of the column logits
- calculation of the hierarchical log likelihood loss for cell selection
- calculation of the token logits, classification and aggregation logits
- calculation of the aggregation loss (this function makes use of _calculate_aggregate_mask,
_calculate_aggregation_loss_known and _calculate_aggregation_loss_unknown)
- calculation of the regression loss (this function makes use of _calculate_expected_result)
"""

import enum

import torch

from torch_scatter import scatter


EPSILON_ZERO_DIVISION = 1e-10
CLOSE_ENOUGH_TO_LOG_ZERO = -10000.0


class AverageApproximationFunction(str, enum.Enum):
    RATIO = "ratio"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


### Beginning of everything related to segmented tensors ###


class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """Creates an index.
        Args:
            indices (:obj:`torch.LongTensor`, same shape as `values`):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor
                must have the same number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor
                are treated as batch dimensions. Segments in different batch elements are always distinct
                even if they have the same index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object


class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """Combines indices i and j into pairs (i, j).
        The result is an index where each segment (i, j) is the intersection of
        segments i and j. For example if the inputs represent table cells indexed by
        respectively rows and columns the output will be a table indexed by
        (row, column) pairs, i.e. by cell.
        The implementation combines indices {0, .., n - 1} and {0, .., m - 1} into
        {0, .., nm - 1}. The output has `num_segments` equal to
            `outer_index.num_segments` * `inner_index.num_segments`.
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def gather(values, index, name="segmented_gather"):
    """Gathers from `values` using the index map.
    For each element in the domain of the index map this operation looks up a
    value for that index in `values`. Two elements from the same segment always
    get assigned the same value.
    Args:
        values (:obj:`torch.Tensor` of shape (B1, ..., Bn, num_segments, V1, ...)):
            Tensor with segment values.
        index (:obj:`IndexMap` of shape (B1, ..., Bn, I1, ..., Ik)):
            IndexMap.
        name (:obj:`str`, `optional`, defaults to 'segmented_gather'):
            Name for the operation. Currently not used.
    Returns:
        :obj:`tuple(torch.Tensor)`: Tensor of shape (B1, ..., Bn, I1, ..., Ik, V1, ...) with the gathered values.
    """
    indices = index.indices
    # first, check whether the indices of the index represent scalar values (i.e. not vectorized)
    if len(values.shape[index.batch_dims :]) < 2:
        return torch.gather(
            values,
            index.batch_dims,
            indices.view(
                values.size()[0], -1
            ),  # torch.gather expects index to have the same number of dimensions as values
        ).view(indices.size())
    else:
        # this means we have a vectorized version
        # we have to adjust the index
        indices = indices.unsqueeze(-1).expand(values.shape)
        return torch.gather(values, index.batch_dims, indices)


def flatten(index, name="segmented_flatten"):
    """Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map.
    This operation relabels the segments to keep batch elements distinct. The k-th
    batch element will have indices shifted by `num_segments` * (k - 1). The
    result is a tensor with `num_segments` multiplied by the number of elements
    in the batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used.
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)


def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used.
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used.
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index


def reduce_sum(values, index, name="segmented_reduce_sum"):
    """Sums a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the sum over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be a sum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the sum must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used.
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]):
            Tensor containing the output values.
        output_index (:obj:`IndexMap`):
            IndexMap with shape [B1, B2, ..., Bn, num_segments].       .
    """
    return _segment_reduce(values, index, "sum", name)


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be a mean of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used.
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]):
            Tensor containing the output values.
        output_index (:obj:`IndexMap`):
            IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)


def reduce_max(values, index, name="segmented_reduce_max"):
    """Computes the maximum over segments.
    This operations computes the maximum over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be an element-wise maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the max must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used.
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]):
            Tensor containing the output values.
        output_index (:obj:`IndexMap`):
            IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "max", name)


def reduce_min(values, index, name="segmented_reduce_min"):
    """Computes the minimum over segments.
    This operations computes the maximum over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in
        a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present
        the output will be an element-wise maximum of vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the min must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used.
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]):
            Tensor containing the output values.
        output_index (:obj:`IndexMap`):
            IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "min", name)


### End of everything related to segmented tensors ###


def compute_column_logits(
    sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection
):
    """Computes the column logits.

    Args:
        sequence_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        column_output_weights (:obj:`torch.FloatTensor` of shape :obj:`(hidden_size)`):
            Weights of the linear layer for column selection.
        column_output_bias (:obj:`torch.FloatTensor` of shape :obj:`()`):
            Bias of the linear layer for column selection.
        cell_index (:obj:`ProductIndexMap`):
            Index that groups tokens into cells.
        cell_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).
        allow_empty_column_selection (:obj:`bool`):
            Whether to allow not to select any column.
    Returns:
        column_logits (:obj:`torch.FloatTensor`of shape :obj:`(batch_size, max_num_cols)`):
            Tensor containing the column logits for every example in the batch.
    """

    # First, compute the token logits (batch_size, seq_len) - without temperature
    token_logits = torch.einsum("bsj,j->bs", sequence_output, column_output_weights) + column_output_bias

    # Next, average the logits per cell (batch_size, max_num_cols*max_num_rows)
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)

    # Finally, average the logits per column (batch_size, max_num_cols)
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)

    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION

    # Mask columns that do not appear in the example.
    is_padding = torch.logical_and(cell_count < 0.5, ~torch.eq(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
        is_padding, dtype=torch.float32, device=is_padding.device
    )

    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
            torch.eq(out_index.indices, 0), dtype=torch.float32, device=out_index.indices.device
        )

    return column_logits


def _single_column_cell_selection_loss(token_logits, column_logits, label_ids, cell_index, col_index, cell_mask):
    """Computes the loss for cell selection constrained to a single column.
    The loss is a hierarchical log-likelihood. The model first predicts a column
    and then selects cells within that column (conditioned on the column). Cells
    outside the selected column are never selected.

    Args:
        token_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Tensor containing the logits per token.
        column_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_num_cols)`):
            Tensor containing the logits per column.
        label_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Labels per token.
        cell_index (:obj:`ProductIndexMap`):
            Index that groups tokens into cells.
        col_index (:obj:`IndexMap`):
            Index that groups tokens into columns.
        cell_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).

    Returns:
        selection_loss_per_example (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Loss for each example.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            New logits which are only allowed to select cells in a single column. Logits outside of the most likely
            column according to `column_logits` will be set to a very low value (such that the probabilities are 0).
    """
    ## Part 1: column loss

    # First find the column we should select. We use the column with maximum
    # number of selected cells.
    labels_per_column, _ = reduce_sum(
        torch.as_tensor(label_ids, dtype=torch.float32, device=label_ids.device), col_index
    )
    # shape of labels_per_column is (batch_size, max_num_cols). It contains the number of label ids for every column, for every example
    column_label = torch.argmax(labels_per_column, dim=-1)  # shape (batch_size,)
    # Check if there are no selected cells in the column. In that case the model
    # should predict the special column id 0, which means "select nothing".
    no_cell_selected = torch.eq(
        torch.max(labels_per_column, dim=-1)[0], 0
    )  # no_cell_selected is of shape (batch_size,) and equals True
    # if an example of the batch has no cells selected (i.e. if there are no label_ids set to 1 for that example)
    column_label = torch.where(
        no_cell_selected.view(column_label.size()), torch.zeros_like(column_label), column_label
    )

    column_dist = torch.distributions.Categorical(logits=column_logits)  # shape (batch_size, max_num_cols)
    column_loss_per_example = -column_dist.log_prob(column_label)

    ## Part 2: cell loss

    # Reduce the labels and logits to per-cell from per-token.
    # logits_per_cell: shape (batch_size, max_num_rows*max_num_cols) i.e. (batch_size, 64*32)
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    # labels_per_cell: shape (batch_size, 64*32), indicating whether each cell should be selected (1) or not (0)
    labels_per_cell, labels_index = reduce_max(
        torch.as_tensor(label_ids, dtype=torch.long, device=label_ids.device), cell_index
    )

    # Mask for the selected column.
    # column_id_for_cells: shape (batch_size, 64*32), indicating to which column each cell belongs
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    # column_mask: shape (batch_size, 64*32), equal to 1 if cell belongs to column to be selected
    column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(column_label, dim=-1)),
        dtype=torch.float32,
        device=cell_mask.device,
    )

    # Compute the log-likelihood for cells, but only for the selected column.
    cell_dist = torch.distributions.Bernoulli(logits=logits_per_cell)  # shape (batch_size, 64*32)
    cell_log_prob = cell_dist.log_prob(labels_per_cell.type(torch.float32))  # shape(batch_size, 64*32)

    cell_loss = -torch.sum(cell_log_prob * column_mask * cell_mask, dim=1)

    # We need to normalize the loss by the number of cells in the column.
    cell_loss /= torch.sum(column_mask * cell_mask, dim=1) + EPSILON_ZERO_DIVISION

    selection_loss_per_example = column_loss_per_example
    selection_loss_per_example += torch.where(
        no_cell_selected.view(selection_loss_per_example.size()),
        torch.zeros_like(selection_loss_per_example),
        cell_loss,
    )

    # Set the probs outside the selected column (selected by the *model*)
    # to 0. This ensures backwards compatibility with models that select
    # cells from multiple columns.
    selected_column_id = torch.as_tensor(
        torch.argmax(column_logits, dim=-1), dtype=torch.long, device=column_logits.device
    )  # shape (batch_size,)

    # selected_column_mask: shape (batch_size, 64*32), equal to 1 if cell belongs to column selected by the model
    selected_column_mask = torch.as_tensor(
        torch.eq(column_id_for_cells, torch.unsqueeze(selected_column_id, dim=-1)),
        dtype=torch.float32,
        device=selected_column_id.device,
    )

    # Never select cells with the special column id 0.
    selected_column_mask = torch.where(
        torch.eq(column_id_for_cells, 0).view(selected_column_mask.size()),
        torch.zeros_like(selected_column_mask),
        selected_column_mask,
    )
    new_logits_per_cell = logits_per_cell + CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    logits = gather(new_logits_per_cell, cell_index)

    return selection_loss_per_example, logits


def compute_token_logits(sequence_output, temperature, output_weights, output_bias):
    """Computes logits per token.
    Args:
        sequence_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        temperature (:obj:`float`):
            Temperature for the Bernoulli distribution.
        output_weights (:obj:`torch.FloatTensor` of shape :obj:`(hidden_size,)`):
            Weights of the linear layer for cell selection.
        output_bias (:obj:`torch.FloatTensor` of shape :obj:`()`):
            Bias of the linear layer for cell selection.
    Returns:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Logits per token.
    """
    logits = (torch.einsum("bsj,j->bs", sequence_output, output_weights) + output_bias) / temperature

    return logits


def _calculate_aggregate_mask(answer, pooled_output, cell_select_pref, label_ids, aggregation_classifier):
    """Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select
    answers directly from the table, without any aggregation function. If the
    answer is a piece of text the case is unambiguous as aggregation functions
    only apply to numbers. If the answer is a number but does not appear in the
    table then we must use some aggregation case. The ambiguous case is when the
    answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether
    to select or aggregate. The threshold for this is a hyperparameter
    `cell_select_pref`
    Args:
        answer (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_select_pref (:obj:`float`):
            Preference for cell selection in ambiguous cases.
        label_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Labels per token.
       aggregation_classifier (:obj:`torch.nn.Linear`):
            Aggregation head. 
    Returns:
        aggregate_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
    """
    # torch.FloatTensor(batch_size,)
    aggregate_mask_init = torch.logical_not(torch.isnan(answer)).type(torch.FloatTensor).to(answer.device)
    logits_aggregation = aggregation_classifier(pooled_output)
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    # Index 0 correponds to "no aggregation".
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)

    # Cell selection examples according to current model.
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_select_pref

    # Examples with non-empty cell selection supervision.
    is_cell_supervision_available = torch.sum(label_ids, dim=1) > 0

    # torch.where is not equivalent to tf.where (in tensorflow 1)
    # hence the added .view on the condition to match the shape of the first tensor
    aggregate_mask = torch.where(
        torch.logical_and(is_pred_cell_selection, is_cell_supervision_available).view(aggregate_mask_init.size()),
        torch.zeros_like(aggregate_mask_init, dtype=torch.float32),
        aggregate_mask_init,
    )

    aggregate_mask = aggregate_mask.detach()

    return aggregate_mask


def _calculate_aggregation_loss_known(logits_aggregation, aggregate_mask, aggregation_function_id, config):
    """Calculates aggregation loss when its type is known during training.

    In the weakly supervised setting, the only known information is that for
    cell selection examples, "no aggregation" should be predicted. For other
    examples (those that require aggregation), no loss is accumulated.
    In the setting where aggregation type is always known, standard cross entropy
    loss is accumulated for all examples.
    Args:
        logits_aggregation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_function_id (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`):
            Aggregation function id for every example in the batch.
        config (:class:`~transformers.TapasConfig`):
            Model configuration class with all the parameters of the model.
    Returns:
        aggregation_loss_known (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Aggregation loss (when its type is known during training) per example.
    """
    if config.use_answer_as_supervision:
        # Prepare "no aggregation" targets for cell selection examples.
        target_aggregation = torch.zeros_like(aggregate_mask, dtype=torch.long)
    else:
        # Use aggregation supervision as the target.
        target_aggregation = aggregation_function_id

    one_hot_labels = torch.nn.functional.one_hot(target_aggregation, num_classes=config.num_aggregation_labels).type(
        torch.float32
    )
    log_probs = torch.nn.functional.log_softmax(logits_aggregation, dim=-1)

    # torch.FloatTensor[batch_size]
    per_example_aggregation_intermediate = -torch.sum(one_hot_labels * log_probs, dim=-1)
    if config.use_answer_as_supervision:
        # Accumulate loss only for examples requiring cell selection
        # (no aggregation).
        return per_example_aggregation_intermediate * (1 - aggregate_mask)
    else:
        return per_example_aggregation_intermediate


def _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask):
    """Calculates aggregation loss in the case of answer supervision.
    Args:
        logits_aggregation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
    Returns:
        aggregation_loss_unknown (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Aggregation loss (in case of answer supervision) per example.
    """

    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    # Index 0 correponds to "no aggregation".
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    # Predict some aggregation in case of an answer that needs aggregation.
    # This increases the probability of all aggregation functions, in a way
    # similar to MML, but without considering whether the function gives the
    # correct answer.
    return -torch.log(aggregation_ops_total_mass) * aggregate_mask


def _calculate_aggregation_loss(logits_aggregation, aggregate_mask, aggregation_function_id, config):
    """Calculates the aggregation loss per example.
    Args:
        logits_aggregation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_function_id (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`):
            Aggregation function id for every example in the batch.
        config (:class:`~transformers.TapasConfig`):
            Model configuration class with all the parameters of the model.
    Returns:
        aggregation_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Aggregation loss per example.
    """
    per_example_aggregation_loss = _calculate_aggregation_loss_known(
        logits_aggregation, aggregate_mask, aggregation_function_id, config
    )

    if config.use_answer_as_supervision:
        # Add aggregation loss for numeric answers that need aggregation.
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return config.aggregation_loss_importance * per_example_aggregation_loss


def _calculate_expected_result(
    dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
):
    """Calculate the expected result given cell and aggregation probabilities.
    Args:
        dist_per_cell (:obj:`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (:obj: `torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config (:class:`~transformers.TapasConfig`):
            Model configuration class with all the parameters of the model.
    Returns:
        expected_result (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            The expected result per example.
    """
    if config.use_gumbel_for_cells:
        gumbel_dist = torch.distributions.RelaxedBernoulli(
            # The token logits where already divided by the temperature and used for
            # computing cell selection errors so we need to multiply it again here
            temperature=config.temperature,
            logits=dist_per_cell.logits * config.temperature,
        )
        scaled_probability_per_cell = gumbel_dist.sample()
    else:
        scaled_probability_per_cell = dist_per_cell.probs

    # <float32>[batch_size, seq_length]
    scaled_probability_per_cell = (scaled_probability_per_cell / numeric_values_scale) * input_mask_float
    count_result = torch.sum(scaled_probability_per_cell, dim=1)
    numeric_values_masked = torch.where(
        torch.isnan(numeric_values), torch.zeros_like(numeric_values), numeric_values
    )  # Mask non-numeric table values to zero.
    sum_result = torch.sum(scaled_probability_per_cell * numeric_values_masked, dim=1)
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        # The sum of all probabilities except that correspond to other cells
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell / ex, dim=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        # The sum of all probabilities exept that correspond to other cells
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        var = torch.sum(pointwise_var, dim=1, keepdim=True) - pointwise_var

        multiplier = (var / torch.square(ex) + 1) / ex
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell * multiplier, dim=1)
    else:
        raise ValueError(f"Invalid average_approximation_function: {config.average_approximation_function}")

    if config.use_gumbel_for_agg:
        gumbel_dist = torch.distributions.RelaxedOneHotCategorical(
            config.agg_temperature, logits=logits_aggregation[:, 1:]
        )
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        # <float32>[batch_size, num_aggregation_labels - 1]
        aggregation_op_only_probs = torch.nn.functional.softmax(
            logits_aggregation[:, 1:] / config.agg_temperature, dim=-1
        )

    all_results = torch.cat(
        [
            torch.unsqueeze(sum_result, dim=1),
            torch.unsqueeze(average_result, dim=1),
            torch.unsqueeze(count_result, dim=1),
        ],
        dim=1,
    )

    expected_result = torch.sum(all_results * aggregation_op_only_probs, dim=1)
    return expected_result


# PyTorch does not currently support Huber loss with custom delta so we define it ourself
def huber_loss(input, target, delta: float = 1.0):
    errors = torch.abs(input - target)  # shape (batch_size,)
    return torch.where(errors < delta, 0.5 * errors ** 2, errors * delta - (0.5 * delta ** 2))


def _calculate_regression_loss(
    answer,
    aggregate_mask,
    dist_per_cell,
    numeric_values,
    numeric_values_scale,
    input_mask_float,
    logits_aggregation,
    config,
):
    """Calculates the regression loss per example.
    Args:
        answer (:obj: `torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        aggregate_mask (:obj: `torch.FloatTensor` of shape :obj:`(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
        dist_per_cell (:obj:`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (:obj: `torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (:obj: `torch.FloatTensor` of shape :obj:`(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config (:class:`~transformers.TapasConfig`):
            Model configuration class with all the parameters of the model.
    Returns:
        per_example_answer_loss_scaled (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            Scales answer loss for each example in the batch.
        large_answer_loss_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`):
            A mask which is 1 for examples for which their answer loss is larger than the answer_loss_cutoff.
    """
    # <float32>[batch_size]
    expected_result = _calculate_expected_result(
        dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config
    )

    # <float32>[batch_size]
    answer_masked = torch.where(torch.isnan(answer), torch.zeros_like(answer), answer)

    if config.use_normalized_answer_loss:
        normalizer = (torch.max(torch.abs(expected_result), torch.abs(answer_masked)) + EPSILON_ZERO_DIVISION).detach()

        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        per_example_answer_loss = huber_loss(
            normalized_expected_result * aggregate_mask, normalized_answer_masked * aggregate_mask
        )
    else:
        per_example_answer_loss = huber_loss(
            expected_result * aggregate_mask, answer_masked * aggregate_mask, delta=config.huber_loss_delta
        )

    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = torch.ones_like(per_example_answer_loss, dtype=torch.float32)

    else:
        large_answer_loss_mask = torch.where(
            per_example_answer_loss > config.answer_loss_cutoff,
            torch.zeros_like(per_example_answer_loss, dtype=torch.float32),
            torch.ones_like(per_example_answer_loss, dtype=torch.float32),
        )
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)

    return per_example_answer_loss_scaled, large_answer_loss_mask
