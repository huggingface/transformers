import gzip
import json
import multiprocessing as mp
import os
import re
import shutil
from collections import defaultdict
from typing import List, Optional, Set

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

    def get_duplicate_clusters(self):
        duplicate_clusters = []
        for base, duplicates in self.__duplicate_clusters.items():
            duplicate_clusters.append(list(duplicates) + [base])
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


def minhash_iter(dataset_iterator):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
            compute_min_hash,
            ThreadedIterator(dataset_iterator, max_queue_size=10000),
            chunksize=100,
        ):
            if data is not None:
                yield data


def make_duplicate_clusters(dataset_iterator):
    """This function will be rewritten with dataset map"""
    di = DuplicationIndex()

    for filename, min_hash in tqdm(ThreadedIterator(minhash_iter(enumerate(dataset_iterator)), max_queue_size=100)):
        di.add(filename, min_hash)

    # Returns a List[Cluster] where Cluster is List[str] with the filenames.
    return di.get_duplicate_clusters()


def jaccard_similarity(code1, code2) -> float:
    """
    Compute the Jaccard similarity of two code snippets.

    """
    tokens1 = get_tokens(code1)
    tokens2 = get_tokens(code2)
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def find_cluster_extremes(cluster: List[tuple], dataset) -> List[tuple]:
    """
    Find a reduced cluster such that each code in the origin cluster is similar to at least one code in the reduced cluster.
    """
    extremes = []
    for element1 in cluster:
        code1 = dataset[element1[0]]["content"]
        for element2 in extremes:
            code2 = dataset[element2[0]]["content"]
            if jaccard_similarity(code1, code2) >= 0.85:
                break
        else:
            extremes.append(code1)
    return extremes


def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, "rb") as f_in:
        with gzip.open(file_path + ".gz", "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)


def main(dataset, output_dir, samples_per_file):
    dataset_iterator = iter(dataset)
    duplicate_clusters = make_duplicate_clusters(dataset_iterator)
    duplicate_indices = set(x[0] for cluster in duplicate_clusters for x in cluster)
    # todo: make this multiprocessing
    extreme_indices = set()
    for cluster in duplicate_clusters:
        for el in find_cluster_extremes(cluster, dataset):
            extreme_indices.add(el[0])
    remove_indices = duplicate_indices - extreme_indices
    ds_filter = dataset.filter(lambda x, idx: idx not in remove_indices, with_indices=True)

    print("Orginal dataset size: %d" % len(dataset))
    print("Duplicate cluster: %d" % len(duplicate_clusters))
    print("Files in duplicate cluster: %d" % len(duplicate_indices))
    print("Unique files in duplicate cluster: %d" % len(extreme_indices))
    print("Filtered dataset size: %d" % len(ds_filter))

    for file_number, index in enumerate(range(0, len(ds_filter), samples_per_file)):
        file_path = f"{output_dir}/file-{file_number+1:012}.json"
        end_index = min(len(ds_filter), index + samples_per_file)
        ds_filter.select(list(range(index, end_index))).to_json(file_path)
        compress_file(file_path)
