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
"""Tests for the read-only cache fallback defined in the repo-root ``conftest.py``.

These exercise the *test runner* itself rather than the ``transformers`` library, so they
live in their own directory and are scheduled by the tests fetcher whenever ``conftest.py``
is modified (see ``utils/tests_fetcher.py``).
"""

import errno
import os
import unittest
import unittest.mock as mock


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
        # Reset the module-level session cache dir so each test starts from a clean slate;
        # the patcher restores whatever value was there before on cleanup.
        patcher = mock.patch.object(conftest, "_ci_fallback_cache_dir", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_is_readonly_fs_error_classification(self):
        is_ro = self.conftest._is_readonly_fs_error
        # Plain download path: OSError with EROFS errno.
        self.assertTrue(is_ro(OSError(errno.EROFS, "Read-only file system")))
        # A wrapped OSError is detected through the exception chain.
        wrapped_erofs = RuntimeError("wrapped")
        wrapped_erofs.__cause__ = OSError(errno.EROFS, "Read-only file system")
        self.assertTrue(is_ro(wrapped_erofs))
        # hf_xet path: bare RuntimeError carrying the raw OS errno as "(os error N)".
        self.assertTrue(is_ro(RuntimeError("I/O error: Read-only file system (os error 30)")))
        self.assertTrue(is_ro(RuntimeError("Data processing error: I/O error: OS ERROR 30")))
        # Negatives: unrelated errors must propagate untouched.
        self.assertFalse(is_ro(OSError(errno.EACCES, "Permission denied")))
        self.assertFalse(is_ro(RuntimeError("some unrelated runtime error")))
        self.assertFalse(is_ro(RuntimeError("I/O error (os error 13)")))  # EACCES, not EROFS
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


if __name__ == "__main__":
    unittest.main()
