import json
import multiprocessing as mp
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Type

from datasets import Dataset
from tqdm import tqdm

from datasketch import MinHash, MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator


NON_ALPHA = re.compile("[^A-Za-z_0-9]")
# parameters used in DuplicationIndex
MIN_NUM_TOKENS = 10
NUM_PERM = 256


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
    ):
        self._duplication_jaccard_threshold = duplication_jaccard_threshold
        self._num_perm = NUM_PERM
        self._index = MinHashLSH(threshold=self._duplication_jaccard_threshold, num_perm=self._num_perm)

        self._duplicate_clusters = defaultdict(set)

    def add(self, filename: str, min_hash: MinHash) -> None:
        close_duplicates = self._index.query(min_hash)
        if filename in self._index.keys:
            print("Duplicate key %s" % filename)
            return

        self._index.insert(filename, min_hash)
        if len(close_duplicates) > 0:

            for base_duplicate in close_duplicates:
                if base_duplicate in self._duplicate_clusters:
                    self._duplicate_clusters[base_duplicate].add(filename)
                    break
            else:
                self._duplicate_clusters[close_duplicates[0]].add(filename)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        duplicate_clusters = []
        for base, duplicates in self._duplicate_clusters.items():
            cluster = [base] + list(duplicates)
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


_shared_dataset = None


def _find_cluster_extremes_shared(cluster):
    """
    Find a reduced cluster such that each code in the origin cluster is similar to at least one code in the reduced cluster.
    """
    extremes = []
    for element1 in cluster:
        code1 = _shared_dataset[element1["base_index"]]["content"]
        for element2 in extremes:
            code2 = _shared_dataset[element2["base_index"]]["content"]
            if jaccard_similarity(code1, code2) >= 0.85:
                element2["copies"] += 1
                break
        else:
            element1["copies"] = 1
            extremes.append(element1)
    return extremes


def find_extremes(cluster_list, dataset):
    global _shared_dataset
    _shared_dataset = dataset
    extremes_list = []
    with mp.Pool() as pool:
        for extremes in tqdm(
            pool.imap_unordered(
                _find_cluster_extremes_shared,
                cluster_list,
            ),
            total=len(cluster_list),
        ):
            extremes_list.append(extremes)
    return extremes_list


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
    extreme_dict = {}
    extremes_clusters = find_extremes(duplicate_clusters, dataset)
    for extremes in extremes_clusters:
        for element in extremes:
            extreme_dict[element["base_index"]] = element
    remove_indices = duplicate_indices - set(extreme_dict.keys())
    ds_filter = dataset.filter(lambda x, idx: idx not in remove_indices, with_indices=True)

    # update duplicate_clusters
    for cluster in duplicate_clusters:
        for element in cluster:
            element["is_extreme"] = element["base_index"] in extreme_dict
            if element["is_extreme"]:
                element["copies"] = extreme_dict[element["base_index"]]["copies"]

    print(f"Original dataset size: {len(dataset)}")
    print(f"Number of duplicate clusters: {len(duplicate_clusters)}")
    print("Files in duplicate cluster: %d" % len(duplicate_indices))
    print("Unique files in duplicate cluster: %d" % len(extreme_dict))
    print("Filtered dataset size: %d" % len(ds_filter))

    return ds_filter, duplicate_clusters
