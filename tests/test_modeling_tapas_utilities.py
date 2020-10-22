import unittest

import numpy as np
import torch

from transformers import modeling_tapas_utilities as utils


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


if __name__ == "__main__":
    unittest.main()
