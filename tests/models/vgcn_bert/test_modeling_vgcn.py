from typing import List, Optional
import unittest
import torch
import numpy as np
import scipy.sparse as sp

import transformers as tfr
from torch import nn

from transformers.models.vgcn_bert.modeling_vgcn_bert import VocabGraphConvolution


def _rand_adj(g_size):
    adj = torch.sparse_coo_tensor(
        indices=torch.randint(0, g_size, (60,)).view(2, -1),
        values=torch.randint(1, 100, (30,)) / 10,
        size=(g_size, g_size),
    )
    dense_adj = adj.to_dense()
    dense_adj[dense_adj > 10] = 0
    dense_adj.fill_diagonal_(1.0)
    adj = dense_adj.to_sparse_coo()
    return adj


def _zero_padding(adj):
    if adj.layout is not torch.sparse_coo:
        adj = adj.to_sparse_coo()
    indices = adj.indices() + 1
    values = adj.values()
    padded_adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(adj.shape[0] + 1, adj.shape[1] + 1))
    return padded_adj.coalesce()


def _sub_dense_graph(dense_adj, input_ids):
    batch_size = input_ids.shape[0]
    batch_dense_mask = torch.zeros((batch_size,) + dense_adj.shape, dtype=torch.float32)
    row_idx = input_ids.unsqueeze(-1).repeat(1, 1, batch_dense_mask.size(-1))
    # Set values in G corresponding to row and column indices to 1
    batch_dense_mask.scatter_(1, row_idx, 1)
    # col_idx = x.unsqueeze(-2).repeat(1, subgraph_mask.size(-2), 1)
    col_idx = row_idx.transpose(-1, -2)
    batch_dense_mask.scatter_(2, col_idx, 1)
    ground_truth_dense_subadj = dense_adj * batch_dense_mask
    return ground_truth_dense_subadj


class TestVocabGraphConvolution(unittest.TestCase):
    def setUp(self):
        self.g_size = 20
        self.wgraphs = [
            _zero_padding(_rand_adj(self.g_size - 1)),
        ]
        self.wgraph_id_to_tokenizer_id_maps = [
            {i: i for i in range(self.g_size)},
        ]
        self.model = VocabGraphConvolution(
            hid_dim=10,
            out_dim=4,
            wgraphs=self.wgraphs,
            wgraph_id_to_tokenizer_id_maps=self.wgraph_id_to_tokenizer_id_maps,
        )

    def test_forward(self):
        word_embeddings = nn.Embedding(self.g_size, 5)
        input_ids = torch.tensor([[3, 2, 1, 4, 0]], dtype=torch.long)
        output = self.model(word_embeddings, input_ids)
        self.assertEqual(output.shape, torch.Size([1, 4, 5]))

    def test_subgraph(self):
        word_embeddings = nn.Embedding(self.g_size, 5)
        input_ids = torch.tensor([[3, 1, 1, 4, 0], [4, 2, 4, 0, 0]], dtype=torch.long)
        ground_truth_dense_subadj = _sub_dense_graph(self.wgraphs[0].to_dense(), input_ids)
        subgraphs = self.model.get_subgraphs(self.model.wgraphs[0], input_ids)
        self.assertTrue(torch.allclose(subgraphs[0].to_dense(), ground_truth_dense_subadj[0]))
        self.assertTrue(torch.allclose(subgraphs[1].to_dense(), ground_truth_dense_subadj[1]))
