import unittest
import torch
import numpy as np
import scipy.sparse as sp

import transformers as tfr

from transformers.models.vgcn_bert.modeling_graph import WordGraph


class PmiWordGraphTest(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "moore is like a progressive bull in a china shop , a provocateur crashing into ideas and special-interest groups as he slaps together his own brand of liberalism . ",
            "idiotic and ugly . ",
            "even if the naipaul original remains the real masterpiece , the movie possesses its own languorous charm . ",
        ]
        self.tokenizer = tfr.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.window_size = 20
        self.algorithm = "npmi"
        self.edge_threshold = 0.0
        self.remove_stopwords = False
        self.min_freq_to_keep = 0

    def test_init(self):
        graph = WordGraph(
            self.texts,
            self.tokenizer,
            self.window_size,
            self.algorithm,
            self.edge_threshold,
            self.remove_stopwords,
            self.min_freq_to_keep,
        )
        self.assertIsInstance(graph.adjacency_matrix, sp.csr_matrix)
        self.assertEqual(len(graph.wgraph_id_to_tokenizer_id_map), graph.adjacency_matrix.shape[0])

    def test_zero_padding(self):
        graph = WordGraph(
            self.texts,
            self.tokenizer,
            self.window_size,
            self.algorithm,
            self.edge_threshold,
            self.remove_stopwords,
            self.min_freq_to_keep,
        )
        self.assertTrue(graph.wgraph_id_to_tokenizer_id_map[0] == 0)
        self.assertTrue(graph.adjacency_matrix.todense()[0, :].sum() == 0)
        self.assertTrue(graph.adjacency_matrix.todense()[:, 0].sum() == 0)

    def test_adj_matrix(self):
        graph = WordGraph(
            self.texts,
            self.tokenizer,
            self.window_size,
            self.algorithm,
            self.edge_threshold,
            self.remove_stopwords,
            self.min_freq_to_keep,
        )
        self.assertTrue(all(graph.adjacency_matrix[1:, 1:].diagonal() == np.ones(graph.adjacency_matrix.shape[0] - 1)))
        self.assertTrue(all(graph.adjacency_matrix.data >= 0))

    def test_to_torch_sparse(self):
        graph = WordGraph(
            self.texts,
            self.tokenizer,
            self.window_size,
            self.algorithm,
            self.edge_threshold,
            self.remove_stopwords,
            self.min_freq_to_keep,
        )
        torch_sparse = graph.to_torch_sparse()
        self.assertTrue(torch_sparse.layout == torch.sparse_coo)
        self.assertTrue(torch_sparse.is_coalesced())


class PredefinedWordGraphTest(unittest.TestCase):
    def setUp(self):
        self.entity_relations = [
            ("dog", "labrador", 0.6),
            ("cat", "garfield", 0.7),
            ("city", "montreal", 0.8),
            ("weather", "rain", 0.3),
        ]
        self.tokenizer = tfr.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.remove_stopwords = False

    def test_init(self):
        graph = WordGraph(
            self.entity_relations,
            self.tokenizer,
            self.remove_stopwords,
        )
        self.assertIsInstance(graph.adjacency_matrix, sp.csr_matrix)
        self.assertEqual(len(graph.wgraph_id_to_tokenizer_id_map), graph.adjacency_matrix.shape[0])

    def test_zero_padding(self):
        graph = WordGraph(
            self.entity_relations,
            self.tokenizer,
            self.remove_stopwords,
        )
        self.assertTrue(graph.wgraph_id_to_tokenizer_id_map[0] == 0)
        self.assertTrue(graph.adjacency_matrix.todense()[0, :].sum() == 0)
        self.assertTrue(graph.adjacency_matrix.todense()[:, 0].sum() == 0)

    def test_adj_matrix(self):
        graph = WordGraph(
            self.entity_relations,
            self.tokenizer,
            self.remove_stopwords,
        )
        self.assertTrue(all(graph.adjacency_matrix[1:, 1:].diagonal() == np.ones(graph.adjacency_matrix.shape[0] - 1)))
        self.assertTrue(graph.adjacency_matrix[1, 2] == np.float32(0.3))
        print(graph.adjacency_matrix[2, 1])
        self.assertTrue(graph.adjacency_matrix[2, 1] == np.float32(0.3))
