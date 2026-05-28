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

from huggingface_hub import constants, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, cached_file, has_file, list_repo_templates


RANDOM_BERT = "hf-internal-testing/tiny-random-bert"
TINY_BERT_PT_ONLY = "hf-internal-testing/tiny-bert-pt-only"
CACHE_DIR = os.path.join(constants.HF_HUB_CACHE, "models--hf-internal-testing--tiny-random-bert")
FULL_COMMIT_HASH = "9b8c223d42b2188cb49d29af482996f9d0f3e5a6"

GATED_REPO = "hf-internal-testing/dummy-gated-model"
README_FILE = "README.md"


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

        # Under the mock environment, hf_hub_download will always raise an HTTPError
        with mock.patch(
            "transformers.utils.hub.hf_hub_download",
            side_effect=HfHubHTTPError("failed", response=mock.Mock(status_code=404)),
        ) as mock_head:
            path = cached_file(RANDOM_BERT, "conf", _raise_exceptions_for_connection_errors=False)
            self.assertIsNone(path)
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_has_file(self):
        self.assertTrue(has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME))
        self.assertFalse(has_file(TINY_BERT_PT_ONLY, "tf_model.h5"))
        self.assertFalse(has_file(TINY_BERT_PT_ONLY, "flax_model.msgpack"))

    def test_has_file_in_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Empty cache dir + offline mode => return False
            assert not has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME, local_files_only=True, cache_dir=tmp_dir)

            # Populate cache dir
            hf_hub_download(TINY_BERT_PT_ONLY, WEIGHTS_NAME, cache_dir=tmp_dir)

            # Cache dir + offline mode => return True
            assert has_file(TINY_BERT_PT_ONLY, WEIGHTS_NAME, local_files_only=True, cache_dir=tmp_dir)

    def test_get_file_from_repo_distant(self):
        # should return None if the file does not exist
        self.assertIsNone(
            cached_file(
                "google-bert/bert-base-cased",
                "ahah.txt",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        )

        # The function raises if the repository does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            cached_file(
                "bert-base-case",
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

        # The function raises if the revision does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            cached_file(
                "google-bert/bert-base-cased",
                CONFIG_NAME,
                revision="ahaha",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

        resolved_file = cached_file(
            "google-bert/bert-base-cased",
            CONFIG_NAME,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        # The name is the cached name which is not very easy to test, so instead we load the content.
        config = json.loads(open(resolved_file).read())
        self.assertEqual(config["hidden_size"], 768)

    def test_get_file_from_repo_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "a.txt"
            filename.touch()
            self.assertEqual(
                cached_file(
                    tmp_dir,
                    "a.txt",
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                ),
                str(filename),
            )

            self.assertIsNone(
                cached_file(
                    tmp_dir,
                    "b.txt",
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
            )

    def test_get_file_gated_repo(self):
        """Test download file from a gated repo fails with correct message when not authenticated."""
        with self.assertRaisesRegex(EnvironmentError, "You are trying to access a gated repo."):
            # All files except README.md are protected on a gated repo.
            cached_file(GATED_REPO, "gated_file.txt", token=False)

    def test_has_file_gated_repo(self):
        """Test check file existence from a gated repo fails with correct message when not authenticated."""
        with self.assertRaisesRegex(EnvironmentError, "is a gated repository"):
            # All files except README.md are protected on a gated repo.
            has_file(GATED_REPO, "gated_file.txt", token=False)

    def test_cached_files_exception_raised(self):
        """Test that unhadled exceptions, e.g. ModuleNotFoundError, is properly re-raised by cached_files when hf_hub_download fails."""
        with mock.patch(
            "transformers.utils.hub.hf_hub_download", side_effect=ModuleNotFoundError("No module named 'MockModule'")
        ):
            with self.assertRaises(ModuleNotFoundError):
                # The error should be re-raised by cached_files, not caught in the exception handling block
                cached_file(RANDOM_BERT, "nonexistent.json")


class OfflineModeTests(unittest.TestCase):
    def test_list_repo_templates_w_offline(self):
        with mock.patch("transformers.utils.hub.list_repo_tree", side_effect=OfflineModeIsEnabled()):
            with mock.patch(
                "transformers.utils.hub.snapshot_download", side_effect=LocalEntryNotFoundError("no snapshot found")
            ):
                self.assertEqual(list_repo_templates(RANDOM_BERT, local_files_only=False), [])


class TestGetCheckpointShardFilesPathTraversal(unittest.TestCase):
    """Regression tests for path-traversal in get_checkpoint_shard_files.

    A malicious index.json could carry ``weight_map`` entries with
    ``..`` components or absolute paths that would escape the model
    directory.  See https://github.com/huggingface/transformers/issues/46097
    """

    def _make_index(self, tmp_path, shard_name):
        import json as _json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        index = {"metadata": {"total_size": 1}, "weight_map": {"weight": shard_name}}
        index_file = model_dir / "model.safetensors.index.json"
        index_file.write_text(_json.dumps(index))
        return str(model_dir), str(index_file)

    def test_path_traversal_dotdot_raises(self):
        import tempfile, pathlib
        from transformers.utils.hub import get_checkpoint_shard_files
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, index_file = self._make_index(pathlib.Path(tmp), "../../etc/passwd")
            with self.assertRaises(ValueError):
                get_checkpoint_shard_files(model_dir, index_file)

    def test_absolute_path_raises(self):
        import tempfile, pathlib
        from transformers.utils.hub import get_checkpoint_shard_files
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, index_file = self._make_index(pathlib.Path(tmp), "/etc/passwd")
            with self.assertRaises(ValueError):
                get_checkpoint_shard_files(model_dir, index_file)

    def test_valid_shard_filename_accepted(self):
        import tempfile, pathlib, json as _json
        from transformers.utils.hub import get_checkpoint_shard_files
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "model"
            d.mkdir()
            shard = "model-00001-of-00001.safetensors"
            (d / shard).write_bytes(b"")
            idx = {"metadata": {"total_size": 1}, "weight_map": {"w": shard}}
            f = d / "model.safetensors.index.json"
            f.write_text(_json.dumps(idx))
            files, _ = get_checkpoint_shard_files(str(d), str(f))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith(str(d)))
