# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import json
import os
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from requests.exceptions import HTTPError

from transformers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    cached_file,
    get_file_from_repo,
    has_file,
)


RANDOM_BERT = "hf-internal-testing/tiny-random-bert"
CACHE_DIR = os.path.join(TRANSFORMERS_CACHE, "models--hf-internal-testing--tiny-random-bert")
FULL_COMMIT_HASH = "9b8c223d42b2188cb49d29af482996f9d0f3e5a6"


class GetFromCacheTests(unittest.TestCase):
    def test_cached_file(self):
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        # Should have downloaded the file in here
        self.assertTrue(os.path.isdir(CACHE_DIR))
        # Cache should contain at least those three subfolders:
        for subfolder in ["blobs", "refs", "snapshots"]:
            self.assertTrue(os.path.isdir(os.path.join(CACHE_DIR, subfolder)))
        with open(os.path.join(CACHE_DIR, "refs", "main")) as f:
            main_commit = f.read()
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, "snapshots", main_commit, CONFIG_NAME))
        self.assertTrue(os.path.isfile(archive_file))

        # File is cached at the same place the second time.
        new_archive_file = cached_file(RANDOM_BERT, CONFIG_NAME)
        self.assertEqual(archive_file, new_archive_file)

        # Using a specific revision to test the full commit hash.
        archive_file = cached_file(RANDOM_BERT, CONFIG_NAME, revision="9b8c223")
        self.assertEqual(archive_file, os.path.join(CACHE_DIR, "snapshots", FULL_COMMIT_HASH, CONFIG_NAME))

    def test_cached_file_errors(self):
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            _ = cached_file("tiny-random-bert", CONFIG_NAME)

        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            _ = cached_file(RANDOM_BERT, CONFIG_NAME, revision="aaaa")

        with self.assertRaisesRegex(EnvironmentError, "does not appear to have a file named"):
            _ = cached_file(RANDOM_BERT, "conf")

    def test_non_existence_is_cached(self):
        with self.assertRaisesRegex(EnvironmentError, "does not appear to have a file named"):
            _ = cached_file(RANDOM_BERT, "conf")

        with open(os.path.join(CACHE_DIR, "refs", "main")) as f:
            main_commit = f.read()
        self.assertTrue(os.path.isfile(os.path.join(CACHE_DIR, ".no_exist", main_commit, "conf")))

        path = cached_file(RANDOM_BERT, "conf", _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)

        path = cached_file(RANDOM_BERT, "conf", local_files_only=True, _raise_exceptions_for_missing_entries=False)
        self.assertIsNone(path)

        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Under the mock environment we get a 500 error when trying to reach the tokenizer.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            path = cached_file(RANDOM_BERT, "conf", _raise_exceptions_for_connection_errors=False)
            self.assertIsNone(path)
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_has_file(self):
        self.assertTrue(has_file("hf-internal-testing/tiny-bert-pt-only", WEIGHTS_NAME))
        self.assertFalse(has_file("hf-internal-testing/tiny-bert-pt-only", TF2_WEIGHTS_NAME))
        self.assertFalse(has_file("hf-internal-testing/tiny-bert-pt-only", FLAX_WEIGHTS_NAME))

    def test_get_file_from_repo_distant(self):
        # `get_file_from_repo` returns None if the file does not exist
        self.assertIsNone(get_file_from_repo("bert-base-cased", "ahah.txt"))

        # The function raises if the repository does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            get_file_from_repo("bert-base-case", CONFIG_NAME)

        # The function raises if the revision does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            get_file_from_repo("bert-base-cased", CONFIG_NAME, revision="ahaha")

        resolved_file = get_file_from_repo("bert-base-cased", CONFIG_NAME)
        # The name is the cached name which is not very easy to test, so instead we load the content.
        config = json.loads(open(resolved_file, "r").read())
        self.assertEqual(config["hidden_size"], 768)

    def test_get_file_from_repo_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "a.txt"
            filename.touch()
            self.assertEqual(get_file_from_repo(tmp_dir, "a.txt"), str(filename))

            self.assertIsNone(get_file_from_repo(tmp_dir, "b.txt"))
