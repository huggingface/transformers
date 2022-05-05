from unittest import TestCase

from datasets import Dataset

from deduplicate import make_duplicate_clusters


class MakeDuplicateClustersTest(TestCase):
    def test_make_duplicate_clusters(self):
        data_dict = {
            "repo_name": ["test_repo1", "test_repo2"],
            "path": ["test_1.py", "test_2.py"],
            "content": ["a " * 20, "a " * 30],
        }
        dataset = Dataset.from_dict(data_dict)
        duplicate_clusters = make_duplicate_clusters(dataset)
        self.assertEqual(len(duplicate_clusters), 2)
