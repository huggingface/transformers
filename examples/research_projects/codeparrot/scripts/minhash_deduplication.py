import json
import multiprocessing as mp
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Type

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator


NUM_PERM = 256
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MIN_NUM_TOKENS = 10


def get_min_hash(tokens: List[str]) -> Optional[MinHash]:
    if len(tokens) < MIN_NUM_TOKENS:
        return None
    min_hash = MinHash(num_perm=NUM_PERM)
    for token in set(tokens):
        min_hash.update(token.encode())
    return min_hash


def get_tokens(code: str) -> Set[str]:
    """
    Tokenize a code snippet.
    """
    return set([t for t in NON_ALPHA.split(code) if len(t.strip()) > 0])


class DuplicationIndex:
    def __init__(
        self,
        *,
        duplication_jaccard_threshold: float = 0.85,
        num_perm: int = 256,
        min_num_tokens: int = 10,
    ):
        self.__duplication_jaccard_threshold = duplication_jaccard_threshold
        self.__num_perm = num_perm
        self.__min_num_tokens = min_num_tokens
        self.__index = MinHashLSH(threshold=self.__duplication_jaccard_threshold, num_perm=self.__num_perm)

        self.__duplicate_clusters = defaultdict(set)

    def add(self, filename: str, min_hash: MinHash) -> None:
        close_duplicates = self.__index.query(min_hash)
        if filename in self.__index.keys:
            print("Duplicate key %s" % filename)
            return

        self.__index.insert(filename, min_hash)
        if len(close_duplicates) > 0:
            # print("`%s` duplicate of: %s" % (filename, close_duplicates))

            for base_duplicate in close_duplicates:
                if base_duplicate in self.__duplicate_clusters:
                    self.__duplicate_clusters[base_duplicate].add(filename)
                    break
            else:
                self.__duplicate_clusters[close_duplicates[0]].add(filename)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        duplicate_clusters = []
        for base, duplicates in self.__duplicate_clusters.items():
            cluster = list(duplicates)
            cluster.append(base)
            # reformat the cluster to be a list of dict
            cluster = [{"base_index": el[0], "repo_name": el[1], "path": el[2]} for el in cluster]
            duplicate_clusters.append(cluster)
        return duplicate_clusters

    def save(self, filepath) -> None:
        duplicate_clusters = self.get_duplicate_clusters()
        with open(filepath, "w") as f:
            json.dump(duplicate_clusters, f)


def compute_min_hash(element):
    index, data = element
    min_hash = get_min_hash([t for t in NON_ALPHA.split(data["content"]) if len(t.strip()) > 0])
    if min_hash is not None:
        return (index, data["repo_name"], data["path"]), min_hash


def minhash_iter(dataset_iterator: Type[Dataset]):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
            compute_min_hash,
            ThreadedIterator(dataset_iterator, max_queue_size=10000),
            chunksize=100,
        ):
            if data is not None:
                yield data


def make_duplicate_clusters(dataset_iterator: Type[Dataset]):
    """This function will be rewritten with dataset map"""
    di = DuplicationIndex()

    for filename, min_hash in tqdm(ThreadedIterator(minhash_iter(enumerate(dataset_iterator)), max_queue_size=100)):
        di.add(filename, min_hash)

    # Returns a List[Cluster] where Cluster is List[str] with the filenames.
    return di.get_duplicate_clusters()


def jaccard_similarity(code1: str, code2: str) -> float:
    """Compute the Jaccard similarity of two code snippets."""
    tokens1 = get_tokens(code1)
    tokens2 = get_tokens(code2)
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def find_cluster_extremes(cluster: List[Dict], dataset: Type[Dataset]) -> List[Dict]:
    """
    Find a reduced cluster such that each code in the origin cluster is similar to at least one code in the reduced cluster.
    """
    extremes = []
    for element1 in cluster:
        code1 = dataset[element1["base_index"]]["content"]
        for element2 in extremes:
            code2 = dataset[element2["base_index"]]["content"]
            if jaccard_similarity(code1, code2) >= 0.85:
                element2["copies"] += 1
                break
        else:
            element1["is_extreme"] = True
            element1["copies"] = 1
            extremes.append(element1)
    return extremes


def deduplicate_dataset(dataset: Type[Dataset]) -> Tuple[Type[Dataset], List[List[Dict]]]:
    """Deduplicate the dataset using minhash and jaccard similarity.
    This function first generate duplicate clusters, then each cluster
    is reduced to the extremes that are similar to the other elements in the cluster.
    Codes are called similar if their Jaccard similarity is greater than 0.85.

    Parameters
    ----------
    dataset : Type[Dataset]
        The dataset to deduplicate.

    Returns
    -------
    ds_dedup : Type[Dataset]
        The deduplicated dataset.
    duplicate_clusters : List[List[Dict]]
        The list of duplicate clusters.
        Each cluster is a list of dicts with the following keys:
        - base_index : int
            The index of the code in the original dataset.
        - repo_name : str
        - path : str
        - copies : int
            The number of copies of the code in the cluster. (find_cluster_extremes)
        - is_extreme : bool
            Whether the code is an extreme in the cluster.
        All the codes in the cluster are removed from the dataset except the extremes.
    """
    duplicate_clusters = make_duplicate_clusters(dataset)
    duplicate_indices = set(x["base_index"] for cluster in duplicate_clusters for x in cluster)
    # todo: make this multiprocessing, pay attention to memory usage
    extreme_indices = set()
    for cluster in tqdm(duplicate_clusters):
        for el in find_cluster_extremes(cluster, dataset):
            extreme_indices.add(el["base_index"])
    remove_indices = duplicate_indices - extreme_indices
    ds_filter = dataset.filter(lambda x, idx: idx not in remove_indices, with_indices=True)

    print("Orginal dataset size: %d" % len(dataset))
    print("Duplicate cluster: %d" % len(duplicate_clusters))
    print("Files in duplicate cluster: %d" % len(duplicate_indices))
    print("Unique files in duplicate cluster: %d" % len(extreme_indices))
    print("Filtered dataset size: %d" % len(ds_filter))

    return ds_filter, duplicate_clusters
