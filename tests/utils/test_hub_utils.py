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
import errno
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
            # TODO: only necessary for read-only cache systems; replace with a shared helper
            with unittest.mock.patch.dict(os.environ, {"HF_XET_CACHE": tmp_dir}):
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


class ReadOnlyCacheFallbackTest(unittest.TestCase):
    """Guards the read-only cache fallback defined in the repo-root ``conftest.py``.

    In CI the shared HF cache is read-only, so downloads of models not already present
    fail with EROFS. ``conftest._with_tmpdir_cache_fallback`` wraps ``cached_files`` to
    retry such failures against a writable tmp dir with Xet disabled. Both the plain
    ``OSError``/EROFS path and the ``hf_xet`` ``RuntimeError`` path must be handled --
    the latter was the regression that slipped past the original errno-only check.
    """

    def setUp(self):
        import conftest

        self.conftest = conftest
        # Reset the module-level session cache dir so each test starts from a clean slate.
        self._saved_cache_dir = conftest._ci_fallback_cache_dir
        conftest._ci_fallback_cache_dir = None

    def tearDown(self):
        self.conftest._ci_fallback_cache_dir = self._saved_cache_dir

    def test_is_readonly_fs_error_classification(self):
        is_ro = self.conftest._is_readonly_fs_error
        # Plain download path: OSError with EROFS errno.
        self.assertTrue(is_ro(OSError(errno.EROFS, "Read-only file system")))
        # hf_xet path: bare RuntimeError, matched on message (no .errno set).
        self.assertTrue(is_ro(RuntimeError("I/O error: Read-only file system (os error 30)")))
        self.assertTrue(is_ro(RuntimeError("Data processing error: I/O error: OS ERROR 30")))
        # Negatives: unrelated errors must propagate untouched.
        self.assertFalse(is_ro(OSError(errno.EACCES, "Permission denied")))
        self.assertFalse(is_ro(RuntimeError("some unrelated runtime error")))
        self.assertFalse(is_ro(ValueError("nope")))

    def test_passthrough_on_success(self):
        fn = mock.Mock(return_value="resolved")
        wrapped = self.conftest._with_tmpdir_cache_fallback(fn)
        self.assertEqual(wrapped("repo", filenames=["f"]), "resolved")
        fn.assert_called_once_with("repo", filenames=["f"])

    def test_reraises_non_readonly_error(self):
        fn = mock.Mock(side_effect=OSError(errno.EACCES, "Permission denied"))
        wrapped = self.conftest._with_tmpdir_cache_fallback(fn)
        with self.assertRaises(OSError):
            wrapped()
        fn.assert_called_once()

    def _assert_recovers(self, first_error):
        """The first call raises ``first_error``; the retry must be given a writable
        ``cache_dir`` with Xet disabled, and its result returned."""
        import huggingface_hub.constants as hf_constants

        original_disable_xet = hf_constants.HF_HUB_DISABLE_XET
        calls = []

        def side_effect(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise first_error
            # On the retry Xet must be disabled and a writable cache_dir supplied.
            self.assertTrue(hf_constants.HF_HUB_DISABLE_XET)
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")
            return "recovered"

        fn = mock.Mock(side_effect=side_effect)
        wrapped = self.conftest._with_tmpdir_cache_fallback(fn)

        self.assertEqual(wrapped(path_or_repo_id="repo"), "recovered")
        self.assertEqual(len(calls), 2)
        # The first attempt is untouched; the retry gets the fallback cache dir.
        self.assertNotIn("cache_dir", calls[0])
        retry_cache_dir = calls[1]["cache_dir"]
        self.assertEqual(retry_cache_dir, self.conftest._ci_fallback_cache_dir)
        self.assertTrue(os.path.isdir(retry_cache_dir))
        # Xet-disable patch is scoped to the retry and restored afterwards.
        self.assertEqual(hf_constants.HF_HUB_DISABLE_XET, original_disable_xet)

    def test_retry_on_xet_runtime_error(self):
        # The exact error raised by the hf_xet Rust layer against a read-only cache.
        self._assert_recovers(RuntimeError("Data processing error: I/O error: Read-only file system (os error 30)"))

    def test_retry_on_oserror_erofs(self):
        # The plain (non-Xet) download path raises this.
        self._assert_recovers(OSError(errno.EROFS, "Read-only file system"))


class OfflineModeTests(unittest.TestCase):
    def test_list_repo_templates_w_offline(self):
        with mock.patch("transformers.utils.hub.HfApi.list_repo_tree", side_effect=OfflineModeIsEnabled()):
            with mock.patch(
                "transformers.utils.hub.HfApi.snapshot_download",
                side_effect=LocalEntryNotFoundError("no snapshot found"),
            ):
                self.assertEqual(list_repo_templates(RANDOM_BERT, local_files_only=False), [])
