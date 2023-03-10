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

"""Construct the Vocabulary/Entity Graph from text samples"""

# Build NPMI graph from text samples

# You may (or not) first preprocess the text before build the graph,
# e.g. Stopword removal, String cleaning, Stemming, Nomolization, Lemmatization

from collections import Counter
from math import log
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from transformers.tokenization_utils import PreTrainedTokenizerBase


def build_pmi_graph(
    texts: List[str], tokenizer: PreTrainedTokenizerBase, window_size=20, algorithm="npmi"
) -> Tuple[sp.csr_matrix, List, dict]:
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
    vocab_counter = Counter()
    new_texts = []
    for t in texts:
        words = tokenizer.tokenize(t)
        if len(words) > 0:
            vocab_counter.update(Counter(words))
            new_texts.append(" ".join(words).strip())

    # TODO: question, sort vocab_counter?
    # TODO: question, remove word with freq<n and re generate texts
    texts = new_texts
    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    # windows = []
    # for t in texts:
    #     words = t.split()
    #     length = len(words)
    #     if length <= window_size:
    #         windows.append(words)
    #     else:
    #         for j in range(length - window_size + 1):
    #             window_words = words[j : j + window_size]
    #             windows.append(window_words)
    # vocab_window_counter = Counter()
    # # map(lambda x: vocab_window_counter.update(Counter(set(x))), window_words)
    # for window_words in windows:
    #     vocab_window_counter.update(Counter(set(window_words)))

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
        vocab_window_counter.update(Counter(set(word_ids)))
        word_pair_window_counter.update(
            Counter(
                set(
                    [
                        (word_ids[i], word_ids[j])
                        for i in range(1, len(word_ids))
                        for j in range(i)
                        if word_ids[j] != word_ids[i]
                    ]
                )
            )
        )
        # Need adding inverse pair?
        word_pair_window_counter.update(
            Counter(
                set(
                    [
                        (word_ids[j], word_ids[i])
                        for i in range(1, len(word_ids))
                        for j in range(i)
                        if word_ids[j] != word_ids[i]
                    ]
                )
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
        if value > 0:
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

    return vocab_adj, vocab, vocab_indices


def build_manual_graph(
    words_relations: List[Tuple[str, str, float]], tokenizer: PreTrainedTokenizerBase
) -> Tuple[sp.csr_matrix, List, dict]:
    vocab_counter = Counter()
    word_pairs = {}
    for w1, w2, v in words_relations:
        w1_subwords = tokenizer.tokenize(w1)
        w2_subwords = tokenizer.tokenize(w2)
        vocab_counter.update(Counter(w1_subwords))
        vocab_counter.update(Counter(w2_subwords))
        for sw1 in w1_subwords:
            for sw2 in w2_subwords:
                if sw1 != sw2:
                    word_pairs.setdefault((sw1, sw2), v)

    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    # bulid adjacency matrix
    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []
    for (w1, w2), v in word_pairs.items():
        vocab_adj_row.append(vocab_indices[w1])
        vocab_adj_col.append(vocab_indices[w2])
        # Need adding inverse pair?
        vocab_adj_row.append(vocab_indices[w2])
        vocab_adj_col.append(vocab_indices[w1])
        vocab_adj_weight.append(v)
        vocab_adj_weight.append(v)

    # Build vocabulary adjacency matrix
    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)

    return vocab_adj, vocab, vocab_indices


def build_knowledge_graph(rdf_list: List[str], tokenizer: PreTrainedTokenizerBase) -> Tuple[sp.csr_matrix, List, dict]:
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
    vocab_adj, vocab, vocab_indices = build_manual_graph(words_relations, tokenizer)
    print_matrix(vocab_adj.todense())
    print(len(vocab))

    # texts = [" I am here", "He is here", "here i am, gobefbef"]
    texts = [" I am here!", "He is here", "You are also here, gobefbef!", "What is interpribility"]
    vocab_adj, vocab, vocab_indices = build_pmi_graph(texts, tokenizer, window_size=2)
    print_matrix(vocab_adj.todense())
    print(len(vocab))

    # print(vocab_indices[vocab[3]])
    print("---end---")
