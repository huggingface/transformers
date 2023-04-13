# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

"""Construct the Word/Entity Graph from text samples or pre-defined word-pairs relations

Approaches: NPMI, PMI, pre-defined word-pairs relations.

You may (or not) first preprocess the text before build the graph,
e.g. Stopword removal, String cleaning, Stemming, Nomolization, Lemmatization

"""

from collections import Counter
from math import log
from typing import List, Tuple
import torch

import numpy as np
import scipy.sparse as sp
from transformers.tokenization_utils import PreTrainedTokenizerBase


class WordGraph:
    """
    Word graph based on adjacency matrix, construct from text samples or pre-defined word-pair relations

    Params:
        `rows`: List[str] of text samples, or pre-defined word-pair relations: List[Tuple[str, str, float]]
        `tokenizer`: The same pretrained tokenizer that is used for the model late.
        `window_size`:  Available only for rows is text samples.
            Size of the sliding window for collecting the pieces of text
            and further calculate the NPMI value, default is 20.
        `algorithm`:  Available only for rows is text samples. "npmi" or "pmi", default is "npmi".

    Properties:
        `adj_matrix`: scipy.sparse.csr_matrix, the word graph in sparse adjacency matrix form.
        `vocab`: List of words in the graph.
        `vocab_indices`: indices of vocabulary words.

    """

    def __init__(
        self, rows: list, tokenizer: PreTrainedTokenizerBase, window_size=20, algorithm="npmi", threshold=0.0
    ):
        if type(rows[0]) == tuple:
            (
                self.adjacency_matrix,
                self.vocab,
                self.vocab_indices,
                self.wgraph_id_to_tokenizer_id_map,
                # self.tokenizer_id_to_wgraph_id_map,
                # self.tokenizer_id_to_wgraph_id_array,
            ) = _build_predefined_graph(rows, tokenizer)
        else:
            (
                self.adjacency_matrix,
                self.vocab,
                self.vocab_indices,
                self.wgraph_id_to_tokenizer_id_map,
                # self.tokenizer_id_to_wgraph_id_map,
                # self.tokenizer_id_to_wgraph_id_array,
            ) = _build_pmi_graph(rows, tokenizer, window_size, algorithm, threshold)

    def normalized(self):
        return _normalize_adj(self.adjacency_matrix) if self.adjacency_matrix is not None else None

    def to_torch_sparse(self):
        if self.adjacency_matrix is None:
            return None
        adj = _normalize_adj(self.adjacency_matrix)
        return _scipy_to_torch(adj)


def _normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def _scipy_to_torch(sparse):
    sparse = sparse.tocoo() if sparse.getformat() != "coo" else sparse
    i = torch.LongTensor(np.vstack((sparse.row, sparse.col)))
    v = torch.from_numpy(sparse.data)
    return torch.sparse_coo_tensor(i, v, torch.Size(sparse.shape)).coalesce()


def _delete_special_terms(words: list, terms: list):
    return [w for w in words if w not in terms]


def _build_pmi_graph(
    texts: List[str], tokenizer: PreTrainedTokenizerBase, window_size=20, algorithm="npmi", threshold=0.0
):  # -> Tuple[sp.csr_matrix, list, dict]:
    """
    Build PMI or NPMI adjacency based on text samples

    Params:
        `texts`: List of text sample
        `tokenizer`: The same pretrained tokenizer that is used for the model late.
        `window_size`: Size of the sliding window for collecting the pieces of text
            and further calculate the NPMI value, default is 20.
        `algorithm`: "npmi" or "pmi", default is "npmi".

    Return:
        `vocab_adj`: scipy.sparse.csr_matrix, the graph in sparse adjacency matrix form.
        `vocab`: List of words in the graph.
        `vocab_indices`: indices of vocabulary words.

    """

    # Tokenize the text samples, the tokenizer should be same as that in the combined Bert-like model.
    # Get vocabulary and the word frequency
    vocab_counter = Counter({"[PAD]": 0})
    new_texts = []
    for t in texts:
        words = tokenizer.tokenize(t)
        words = _delete_special_terms(words, ["[CLS]", "[SEP]"])
        if len(words) > 0:
            vocab_counter.update(Counter(words))
            new_texts.append(" ".join(words).strip())

    # TODO: question, sort vocab_counter?
    # TODO: delete stopwords
    # TODO: remove word with freq<n and re generate texts
    texts = new_texts
    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    assert vocab[0] == "[PAD]"
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    # Get the pieces from sliding windows
    windows = []
    for t in texts:
        words = t.split()
        word_ids = [vocab_indices[w] for w in words]
        length = len(word_ids)
        if length <= window_size:
            windows.append(word_ids)
        else:
            for j in range(length - window_size + 1):
                word_ids = word_ids[j : j + window_size]
                windows.append(word_ids)

    # Get the window-count that every word appeared (count 1 for the same window).
    vocab_window_counter = Counter()
    # Get window-count that every word-pair appeared (count 1 for the same window).
    word_pair_window_counter = Counter()
    for word_ids in windows:
        word_ids = list(set(word_ids))
        vocab_window_counter.update(Counter(word_ids))
        word_pair_window_counter.update(
            Counter(
                [
                    f(i, j)
                    # (word_ids[i], word_ids[j])
                    for i in range(1, len(word_ids))
                    for j in range(i)
                    # adding inverse pair
                    for f in (lambda x, y: (word_ids[x], word_ids[y]), lambda x, y: (word_ids[y], word_ids[x]))
                ]
            )
        )

    # Calculate NPMI
    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []

    total_windows = len(windows)
    for wid_pair in word_pair_window_counter.keys():
        i, j = wid_pair
        pair_count = word_pair_window_counter[wid_pair]
        i_count = vocab_window_counter[i]
        j_count = vocab_window_counter[j]
        value = (
            (log(1.0 * i_count * j_count / (total_windows**2)) / log(1.0 * pair_count / total_windows) - 1)
            if algorithm == "npmi"
            else (log((1.0 * pair_count / total_windows) / (1.0 * i_count * j_count / (total_windows**2))))
        )
        if value > threshold:
            vocab_adj_row.append(i)
            vocab_adj_col.append(j)
            vocab_adj_weight.append(value)

    # Build vocabulary adjacency matrix
    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)
    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))
    # tokenizer_id_to_wgraph_id_map = {v: k for k, v in wgraph_id_to_tokenizer_id_map.items()}
    # tokenizer_id_to_wgraph_id_map = dict(sorted(tokenizer_id_to_wgraph_id_map.items()))
    # assert len(wgraph_id_to_tokenizer_id_map) == len(tokenizer_id_to_wgraph_id_map)

    # tokenizer_id_to_wgraph_id_array = np.zeros(max(tokenizer_id_to_wgraph_id_map.keys()) + 1, dtype=np.int64)
    # for tok_id, graph_id in tokenizer_id_to_wgraph_id_map.items():
    #     tokenizer_id_to_wgraph_id_array[tok_id] = graph_id

    return (
        vocab_adj,
        vocab,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
        # tokenizer_id_to_wgraph_id_map,
        # tokenizer_id_to_wgraph_id_array,
    )


def _build_predefined_graph(
    words_relations: List[Tuple[str, str, float]], tokenizer: PreTrainedTokenizerBase
):  # -> Tuple[sp.csr_matrix, list, dict]:
    vocab_counter = Counter({"[PAD]": 0})
    word_pairs = {}
    for w1, w2, v in words_relations:
        w1_subwords = tokenizer.tokenize(w1)
        w1_subwords = _delete_special_terms(w1_subwords, ["[CLS]", "[SEP]"])
        w2_subwords = tokenizer.tokenize(w2)
        w2_subwords = _delete_special_terms(w2_subwords, ["[CLS]", "[SEP]"])
        vocab_counter.update(Counter(w1_subwords))
        vocab_counter.update(Counter(w2_subwords))
        for sw1 in w1_subwords:
            for sw2 in w2_subwords:
                if sw1 != sw2:
                    word_pairs.setdefault((sw1, sw2), v)

    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    assert vocab[0] == "[PAD]"
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    # bulid adjacency matrix
    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []
    for (w1, w2), v in word_pairs.items():
        vocab_adj_row.append(vocab_indices[w1])
        vocab_adj_col.append(vocab_indices[w2])
        vocab_adj_weight.append(v)
        # adding inverse
        vocab_adj_row.append(vocab_indices[w2])
        vocab_adj_col.append(vocab_indices[w1])
        vocab_adj_weight.append(v)

    # Build vocabulary adjacency matrix
    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)
    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    # tokenizer_id_to_wgraph_id_map = {v: k for k, v in wgraph_id_to_tokenizer_id_map.items()}

    # tokenizer_id_to_wgraph_id_array = np.zeros(max(tokenizer_id_to_wgraph_id_map.keys()) + 1, dtype=np.int64)
    # for tok_id, graph_id in tokenizer_id_to_wgraph_id_map.items():
    #     tokenizer_id_to_wgraph_id_array[tok_id] = graph_id

    return (
        vocab_adj,
        vocab,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
        # tokenizer_id_to_wgraph_id_map,
        # tokenizer_id_to_wgraph_id_array,
    )


def _build_knowledge_graph(
    rdf_list: List[str], tokenizer: PreTrainedTokenizerBase
) -> Tuple[sp.csr_matrix, List, dict]:
    """
    Build word level adjacency matrix from a knowledge graph

    Params:

    Return:

    """
    pass


if __name__ == "__main__":
    import os

    import transformers as tfr

    def print_matrix(m):
        for r in m:
            print(" ".join(["%.1f" % v for v in np.ravel(r)]))

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = "/tmp/local-huggingface-models/hf-maintainers_distilbert-base-uncased"

    # DistilBertTokenizerFast
    tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)

    words_relations = [
        ("I", "you", 0.3),
        ("here", "there", 0.7),
        ("city", "montreal", 0.8),
        ("comeabc", "gobefbef", 0.2),
    ]
    wgraph = WordGraph(words_relations, tokenizer)
    # vocab_adj, vocab, vocab_indices = wgraph.adjacency_matrix, wgraph.vocab, wgraph.vocab_indices
    print(len(wgraph.vocab))
    # print(wgraph.tokenizer_id_to_wgraph_id_array)
    print_matrix(wgraph.adjacency_matrix.todense())

    # texts = [" I am here", "He is here", "here i am, gobefbef"]
    texts = [" I am here!", "He is here", "You are also here, gobefbef!", "What is interpribility"]
    wgraph = WordGraph(texts, tokenizer, window_size=4)
    # vocab_adj, vocab, vocab_indices = wgraph.adjacency_matrix, wgraph.vocab, wgraph.vocab_indices

    print(len(wgraph.vocab))
    # print(wgraph.tokenizer_id_to_wgraph_id_array)
    print_matrix(wgraph.adjacency_matrix.todense())
    print()
    norm_adj = _normalize_adj(wgraph.adjacency_matrix)
    print_matrix(norm_adj.todense())

    # print(vocab_indices[vocab[3]])
    print("---end---")
