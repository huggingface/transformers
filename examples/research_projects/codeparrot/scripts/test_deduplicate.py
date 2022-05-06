from unittest import TestCase

from datasets import Dataset

from minhash_deduplication import deduplicate_dataset, make_duplicate_clusters


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

    def test_deduplicate_dataset(self):
        ds = get_dataset()
        ds_filter, duplicate_clusters = deduplicate_dataset(ds)
        self.assertEqual(len(ds_filter), 1)
        self.assertEqual(duplicate_clusters[0][0]["copies"], 2)
        self.assertEqual(duplicate_clusters[0][0]["is_extreme"], True)
