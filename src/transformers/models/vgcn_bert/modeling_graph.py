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
from typing import Dict, List, Tuple
import torch

import numpy as np
import scipy.sparse as sp
from transformers.tokenization_utils import PreTrainedTokenizerBase

ENGLISH_STOP_WORDS = frozenset(
    {
        "herself",
        "each",
        "him",
        "been",
        "only",
        "yourselves",
        "into",
        "where",
        "them",
        "very",
        "we",
        "that",
        "re",
        "too",
        "some",
        "what",
        "those",
        "me",
        "whom",
        "have",
        "yours",
        "an",
        "during",
        "any",
        "nor",
        "ourselves",
        "has",
        "do",
        "when",
        "about",
        "same",
        "our",
        "then",
        "himself",
        "their",
        "all",
        "no",
        "a",
        "hers",
        "off",
        "why",
        "how",
        "more",
        "between",
        "until",
        "not",
        "over",
        "your",
        "by",
        "here",
        "most",
        "above",
        "up",
        "of",
        "is",
        "after",
        "from",
        "being",
        "i",
        "as",
        "other",
        "so",
        "her",
        "ours",
        "on",
        "because",
        "against",
        "and",
        "out",
        "had",
        "these",
        "at",
        "both",
        "down",
        "you",
        "can",
        "she",
        "few",
        "the",
        "if",
        "it",
        "to",
        "but",
        "its",
        "be",
        "he",
        "once",
        "further",
        "such",
        "there",
        "through",
        "are",
        "themselves",
        "which",
        "in",
        "now",
        "his",
        "yourself",
        "this",
        "were",
        "below",
        "should",
        "my",
        "myself",
        "am",
        "or",
        "while",
        "itself",
        "again",
        "with",
        "they",
        "will",
        "own",
        "than",
        "before",
        "under",
        "was",
        "for",
        "who",
    }
)


class WordGraph:
    """
    Word graph based on adjacency matrix, construct from text samples or pre-defined word-pair relations

    Params:
        `rows`: List[str] of text samples, or pre-defined word-pair relations: List[Tuple[str, str, float]]
        `tokenizer`: The same pretrained tokenizer that is used for the model late.
        `window_size`:  Available only for statistics generation (rows is text samples).
            Size of the sliding window for collecting the pieces of text
            and further calculate the NPMI value, default is 20.
        `algorithm`:  Available only for statistics generation (rows is text samples) -- "npmi" or "pmi", default is "npmi".
        `edge_threshold`: Available only for statistics generation (rows is text samples). Graph edge value threshold, default is 0. Edge value is between -1 to 1.
        `remove_stopwords`: Build word graph with the words that are not stopwords, default is False.
        `min_freq_to_keep`: Available only for statistics generation (rows is text samples). Build word graph with the words that occurred at least n times in the corpus, default is 2.

    Properties:
        `adjacency_matrix`: scipy.sparse.csr_matrix, the word graph in sparse adjacency matrix form.
        `vocab_indices`: indices of word graph vocabulary words.
        `wgraph_id_to_tokenizer_id_map`: map from word graph vocabulary word id to tokenizer vocabulary word id.

    """

    def __init__(
        self,
        rows: list,
        tokenizer: PreTrainedTokenizerBase,
        window_size=20,
        algorithm="npmi",
        edge_threshold=0.0,
        remove_stopwords=False,
        min_freq_to_keep=2,
    ):
        if type(rows[0]) == tuple:
            (
                self.adjacency_matrix,
                self.vocab_indices,
                self.wgraph_id_to_tokenizer_id_map,
            ) = _build_predefined_graph(rows, tokenizer, remove_stopwords)
        else:
            (self.adjacency_matrix, self.vocab_indices, self.wgraph_id_to_tokenizer_id_map,) = _build_pmi_graph(
                rows, tokenizer, window_size, algorithm, edge_threshold, remove_stopwords, min_freq_to_keep
            )

    def normalized(self):
        return _normalize_adj(self.adjacency_matrix) if self.adjacency_matrix is not None else None

    def to_torch_sparse(self):
        if self.adjacency_matrix is None:
            return None
        adj = _normalize_adj(self.adjacency_matrix)
        return _scipy_to_torch(adj)


def _normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
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


def _delete_special_terms(words: list, terms: set):
    return set([w for w in words if w not in terms])


def _build_pmi_graph(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    window_size=20,
    algorithm="npmi",
    edge_threshold=0.0,
    remove_stopwords=False,
    min_freq_to_keep=2,
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build statistical word graph from text samples using PMI or NPMI algorithm.
    """

    # Tokenize the text samples. The tokenizer should be same as that in the combined Bert-like model.
    # Remove stopwords and special terms
    # Get vocabulary and the word frequency
    words_to_remove = (
        set({"[CLS]", "[SEP]"}).union(ENGLISH_STOP_WORDS) if remove_stopwords else set({"[CLS]", "[SEP]"})
    )
    vocab_counter = Counter()
    texts_words = []
    for t in texts:
        words = tokenizer.tokenize(t)
        words = _delete_special_terms(words, words_to_remove)
        if len(words) > 0:
            vocab_counter.update(Counter(words))
            texts_words.append(words)

    # Set [PAD] as the head of vocabulary
    # Remove word with freq<n and re generate texts
    new_vocab_counter = Counter({"[PAD]": 0})
    new_vocab_counter.update(
        Counter({k: v for k, v in vocab_counter.items() if v >= min_freq_to_keep})
        if min_freq_to_keep > 1
        else vocab_counter
    )
    vocab_counter = new_vocab_counter

    # Generate new texts by removing words with freq<n
    if min_freq_to_keep > 1:
        texts_words = [list(filter(lambda w: vocab_counter[w] >= min_freq_to_keep, words)) for words in texts_words]
    texts = [" ".join(words).strip() for words in texts_words if len(words) > 0]

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
    # Get window-count that every word-pair appeared (count 1 for the same window).
    vocab_window_counter = Counter()
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
        if value > edge_threshold:
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

    # Padding the first row and column, "[PAD]" is the first word in the vocabulary.
    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

    return (
        vocab_adj,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
    )


def _build_predefined_graph(
    words_relations: List[Tuple[str, str, float]], tokenizer: PreTrainedTokenizerBase, remove_stopwords: bool = False
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build pre-defined wgraph from a list of word pairs and their pre-defined relations (edge value).
    """

    # Tokenize the text samples. The tokenizer should be same as that in the combined Bert-like model.
    # Remove stopwords and special terms
    # Get vocabulary and the word frequency
    words_to_remove = (
        set({"[CLS]", "[SEP]"}).union(ENGLISH_STOP_WORDS) if remove_stopwords else set({"[CLS]", "[SEP]"})
    )
    vocab_counter = Counter({"[PAD]": 0})
    word_pairs = {}
    for w1, w2, v in words_relations:
        w1_subwords = tokenizer.tokenize(w1)
        w1_subwords = _delete_special_terms(w1_subwords, words_to_remove)
        w2_subwords = tokenizer.tokenize(w2)
        w2_subwords = _delete_special_terms(w2_subwords, words_to_remove)
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

    # Padding the first row and column, "[PAD]" is the first word in the vocabulary.
    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

    return (
        vocab_adj,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
    )


def _build_knowledge_graph(
    rdf_list: List[str], tokenizer: PreTrainedTokenizerBase
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build word level adjacency matrix from a knowledge graph
    """
    pass
