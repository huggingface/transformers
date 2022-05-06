from unittest import TestCase

from datasets import Dataset

from minhash_deduplication import make_duplicate_clusters
from minhash_deduplication import deduplicate_dataset


def get_dataset():
    data_dict = {
        "repo_name": ["test_repo1", "test_repo2"],
        "path": ["test_1.py", "test_2.py"],
        "content": ["a " * 20, "a " * 30],
    }
    dataset = Dataset.from_dict(data_dict)
    return dataset


class MakeDuplicateClustersTest(TestCase):
    def test_make_duplicate_clusters(self):
        ds = get_dataset()
        duplicate_clusters = make_duplicate_clusters(ds)
        self.assertEqual(len(duplicate_clusters[0]), 2)

    def test_main(self):
        ds = get_dataset()
        ds_filter = deduplicate_dataset(ds)
