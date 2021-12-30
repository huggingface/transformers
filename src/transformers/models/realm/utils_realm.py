# coding=utf-8
# Copyright 2021 The REALM authors and The HuggingFace Inc. team.
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
"""Utilities for REALM."""

import torch


class ScaNNSearcher:
    def __init__(
        self,
        db,
        num_neighbors,
        dimensions_per_block=2,
        num_leaves=1000,
        num_leaves_to_search=100,
        training_sample_size=100000,
    ):
        """Build scann searcher."""

        from scann.scann_ops.py.scann_ops_pybind import builder as Builder

        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure="dot_product")
        builder = builder.tree(
            num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size
        )
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

        self.searcher = builder.build()

    def search_batched(self, question_projection):
        retrieved_block_ids, _ = self.searcher.search_batched(question_projection.detach().cpu())
        # Must return cpu tensor for subsequent numpy operations
#        return torch.tensor(retrieved_block_ids.astype("int64"), device=torch.device("cpu"))
        return retrieved_block_ids.astype("int64")


class BruteForceSearcher:
    def __init__(self, db, num_neighbors):
        """Build brute force searcher."""
        self.db = db
        self.num_neighbors = num_neighbors

    def search_batched(self, question_projection):
        batch_scores = torch.einsum("BD,QD->QB", self.db, question_projection)
        _, retrieved_block_ids = torch.topk(batch_scores, k=self.num_neighbors, dim=-1)
        # Must return cpu tensor for subsequent numpy operations
        return retrieved_block_ids.cpu()


def convert_tfrecord_to_np(block_records_path, num_block_records):
    import tensorflow.compat.v1 as tf

    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())

    return np_record
